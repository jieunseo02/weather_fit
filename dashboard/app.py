"""weather_fit EDA 대시보드 — 서울 기상 × 패션 수요.

실행: 저장소 루트에서 `streamlit run dashboard/app.py`

현재까지의 EDA 두 개를 한 화면에 모은다.
- notebooks/preview_eda.ipynb : W1 전처리, T1 점유율·단기 변동·성수기 상관·요일 보정
- notebooks/01_w3_eda.ipynb   : T3 배치 정렬(level)·희소 키워드 T4 대체·MSTL·한파 -10℃·예보 기간 구분
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.seasonal import MSTL

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# ── 색: dataviz 기본 팔레트 (범주형은 고정 순서로만 배정) ─────────────
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"
MUTED, GRID, NEUTRAL = "#898781", "#e1e0d9", "#f0efec"
SERIES = [BLUE, ORANGE, AQUA]
DIVERGING = [[0, BLUE], [0.5, NEUTRAL], [1, RED]]

WEATHER_VARS = {
    "t_min": "최저기온",
    "t_max": "최고기온",
    "dtr": "일교차",
    "dt_min": "전일 대비 최저기온 변화(Δt)",
    "dt_min_3d": "3일 누적 최저기온 변화",
    "log_precip": "log(1+강수량)",
}
COLD_THRESHOLD = -10          # 한파 D0: t_min ≤ -10℃ 가 2일 이상 지속되는 첫날
LEAD_CHANGE = pd.Timestamp("2024-12-02")   # 중기예보 리드타임 3일 → 4일 전환일

st.set_page_config(page_title="weather_fit EDA", page_icon="🌦️", layout="wide")


# ═════════════════════════════════════════════════════════════
# 데이터 로딩
# ═════════════════════════════════════════════════════════════
W1_COLUMNS = [
    "TM", "STN", "WS_AVG", "WR_DAY", "WD_MAX", "WS_MAX", "WS_MAX_TM", "WD_INS", "WS_INS", "WS_INS_TM",
    "TA_AVG", "TA_MAX", "TA_MAX_TM", "TA_MIN", "TA_MIN_TM", "TD_AVG", "TS_AVG", "TG_MIN", "HM_AVG",
    "HM_MIN", "HM_MIN_TM", "PV_AVG", "EV_S", "EV_L", "FG_DUR", "PA_AVG", "PS_AVG", "PS_MAX", "PS_MAX_TM",
    "PS_MIN", "PS_MIN_TM", "CA_TOT", "SS_DAY", "SS_DUR", "SS_CMB", "SI_DAY", "SI_60M_MAX", "SI_60M_MAX_TM",
    "RN_DAY", "RN_D99", "RN_DUR", "RN_60M_MAX", "RN_60M_MAX_TM", "RN_10M_MAX", "RN_10M_MAX_TM",
    "RN_POW_MAX", "RN_POW_MAX_TM", "SD_NEW", "SD_NEW_TM", "SD_MAX", "SD_MAX_TM",
    "TE_05", "TE_10", "TE_15", "TE_30", "TE_50",
]


@st.cache_data
def load_weather() -> pd.DataFrame:
    """W1 관측(서울 108) → 일별 기상 테이블. 결측 규칙은 preview_eda.ipynb 와 같다."""
    raw = pd.read_csv(ROOT / "data" / "01_raw" / "raw_w1_asos.csv", header=0, encoding="utf-8-sig")
    raw.columns = W1_COLUMNS[: raw.shape[1]]
    raw = raw[raw["TM"] > 0].copy()
    raw["date"] = pd.to_datetime(raw["TM"].astype(int).astype(str), format="%Y%m%d")
    raw = raw.set_index("date").sort_index()

    def clean(col: str, rule: str) -> pd.Series:
        s = raw[col].astype(float)
        if rule == "temp":          # 기온: -99 계열만 결측 (-9℃ 는 실제 기온일 수 있다)
            return s.mask(s <= -99)
        if rule == "positive":      # 음수가 될 수 없는 값: -9, -99 모두 결측
            return s.mask(s < 0)
        return s.mask(s <= -99).mask(s == -9, 0.0)   # 강수·적설: -9 는 '없음'(0), -99 는 결측

    w = pd.DataFrame({
        "t_max": clean("TA_MAX", "temp"), "t_min": clean("TA_MIN", "temp"), "t_avg": clean("TA_AVG", "temp"),
        "humidity": clean("HM_AVG", "positive"), "wind": clean("WS_AVG", "positive"),
        "cloud": clean("CA_TOT", "positive"), "precip_mm": clean("RN_DAY", "zero"), "snow_cm": clean("SD_NEW", "zero"),
    }).asfreq("D")
    w["wind"] = w["wind"].interpolate(limit=3)
    w["dtr"] = w["t_max"] - w["t_min"]
    w["dt_min"] = w["t_min"].diff()
    w["dt_min_3d"] = w["t_min"].diff(3)
    w["log_precip"] = np.log1p(w["precip_mm"])
    return w


def _align_batches(df: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    """배치 차이는 날짜와 무관한 상수다 → 앵커 평균 비율 하나로 정렬한 level 을 만든다."""
    out = []
    groups = df.groupby(group_col, dropna=False) if group_col else [(None, df)]
    for _, g in groups:
        anc = g[g["is_anchor"]].pivot_table(index="date", columns="batch_id", values="ratio")
        factor = (anc[anc.columns[0]].mean() / anc.mean()).rename("factor")
        g = g.merge(factor, left_on="batch_id", right_index=True, how="left")
        g["level"] = g["ratio"] * g["factor"]
        out.append(g[~g["is_anchor"]])
    return pd.concat(out)


@st.cache_data
def load_demand() -> dict:
    t1 = pd.read_csv(RAW / "datalab_category" / "all_years.csv", parse_dates=["date"])
    t3 = pd.read_csv(RAW / "datalab_keyword" / "all_years.csv", parse_dates=["date"], dtype={"cid": str})
    t4 = pd.read_csv(RAW / "datalab_search" / "all_years.csv", parse_dates=["date"])
    idx = pd.date_range(t1["date"].min(), t1["date"].max(), freq="D")

    # T1: 중분류 ratio 와 패션의류 전체(앵커) → 점유율(%) (preview_eda T1-6)
    total = t1[t1["category_l2"].isna()].groupby("date")["ratio"].first().reindex(idx)
    cat = t1[t1["category_l2"].notna()].pivot_table(index="date", columns="category_l2", values="ratio").reindex(idx)
    share = cat.div(total, axis=0) * 100

    # T3·T4: 배치 상수 정렬 level. 커버리지 95% 미만 T3 키워드는 T4 로 대체 (01_w3_eda §2.2)
    def wide(aligned: pd.DataFrame) -> pd.DataFrame:
        w = aligned.pivot_table(index="date", columns="keyword", values="level").reindex(idx)
        return w[w.columns[w.notna().mean() >= 0.95]].fillna(0.0)

    t3w, t4w = wide(_align_batches(t3, "cid")), wide(_align_batches(t4, None))
    t3_all = set(t3.loc[~t3["is_anchor"], "keyword"])
    dropped = sorted(t3_all - set(t3w.columns))
    subs = [k for k in dropped if k in t4w.columns]
    keywords = pd.concat([t3w, t4w[subs]], axis=1)
    source = {k: ("T4 검색" if k in subs else "T3 쇼핑클릭") for k in keywords.columns}
    return {"total": total, "cat_ratio": cat, "cat_share": share, "keywords": keywords, "kw_source": source,
            "t4": t4w, "dropped": dropped, "subs": subs}


@st.cache_data
def load_warnings() -> pd.DataFrame | None:
    files = sorted((RAW / "kma_warning").glob("*.csv"))
    if not files:
        return None
    w = pd.concat([pd.read_csv(f, parse_dates=["tm_ef"]) for f in files])
    return w[w["lvl"].isin([2, 3])].sort_values("tm_ef")   # lvl=1 은 예비특보라 제외


@st.cache_data
def load_forecast_error() -> pd.DataFrame | None:
    files = sorted((RAW / "kma_forecast").glob("*.csv"))
    if not files:
        return None
    fc = pd.concat([pd.read_csv(f, parse_dates=["tmfc", "target_date"]) for f in files])
    t = fc[(fc["src"] == "temp") & (fc["tmfc"] < fc["target_date"])]
    latest = t.sort_values("tmfc").groupby("target_date").tail(1).set_index("target_date")[["t_min", "t_max"]]
    return latest


# ═════════════════════════════════════════════════════════════
# 변환: 단기 변동(=날씨 효과가 남는 부분)
# ═════════════════════════════════════════════════════════════
def log_safe(s: pd.Series) -> pd.Series:
    eps = max(s[s > 0].quantile(0.01), 1e-3) if (s > 0).any() else 1e-3
    return np.log(s + eps)


@st.cache_data
def demand_short_term(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """로그 수요에서 '평소 수준'을 뺀 값. method 에 따라 기준선이 다르다."""
    out = {}
    for k in df.columns:
        y = log_safe(df[k])
        if method == "ma31":
            st_ = y - y.rolling(31, center=True, min_periods=16).mean()
            st_ = st_ - st_.groupby(st_.index.dayofweek).transform("mean")    # 요일 보정
        else:
            st_ = MSTL(y.interpolate().bfill(), periods=(7, 365)).fit().resid
        out[k] = st_
    return pd.DataFrame(out)


def fourier_baseline(y: pd.Series, K: int = 3) -> pd.Series:
    doy = 2 * np.pi * y.index.dayofyear.values / 365.25
    cols = [np.ones(len(y)), np.arange(len(y)) / 365.25] + [f(k * doy) for k in range(1, K + 1) for f in (np.sin, np.cos)]
    X = np.column_stack(cols)
    ok = y.notna().values
    beta = np.linalg.lstsq(X[ok], y.values[ok], rcond=None)[0]
    return pd.Series(X @ beta, index=y.index)


@st.cache_data
def weather_short_term(w: pd.DataFrame, method: str) -> pd.DataFrame:
    x = w[list(WEATHER_VARS)].copy()
    for v in ["t_min", "t_max", "dtr"]:     # Δt 류와 강수는 계절 성분이 작아 그대로 쓴다
        if method == "ma31":
            x[v] = w[v] - w[v].rolling(31, center=True, min_periods=16).mean()
        else:
            x[v] = w[v] - fourier_baseline(w[v])
    return x


def active_months(df: pd.DataFrame, pct: float) -> dict[str, list[int]]:
    """성수기 = 추세를 걷어낸 월평균이 피크월의 pct% 이상인 달."""
    si = df / df.rolling(365, center=True, min_periods=180).mean()
    mm = si.groupby(si.index.month).mean()
    return {k: [m for m in range(1, 13) if mm.loc[m, k] >= pct / 100 * mm[k].max()] for k in df.columns}


def ccf(y: pd.Series, x: pd.Series, max_lag: int = 7) -> pd.Series:
    return pd.Series({k: y.corr(x.shift(k)) for k in range(-max_lag, max_lag + 1)})


def rule_first_day(cond: pd.Series, min_len: int = 2) -> pd.DatetimeIndex:
    run_end = cond.rolling(min_len).sum().eq(min_len)
    in_run = run_end.copy()
    for s in range(1, min_len):
        in_run |= run_end.shift(-s, fill_value=False)
    return in_run.index[in_run & ~in_run.shift(1, fill_value=False)]


def warning_days(warn: pd.DataFrame, wrn: str, idx: pd.DatetimeIndex, max_days: int = 10) -> pd.Series:
    """권역별 발표~해제를 날짜로 펼쳐 4개 권역 any 로 합친다."""
    days = pd.Series(False, index=idx)
    for _, g in warn[warn["wrn"] == wrn].groupby("reg_id"):
        start = None
        for _, r in g.iterrows():
            if r["cmd"] == 1 and start is None:
                start = r["tm_ef"]
            elif r["cmd"] == 3 and start is not None:
                days.loc[start.normalize():r["tm_ef"].normalize()] = True
                start = None
        if start is not None:
            days.loc[start.normalize():start.normalize() + pd.Timedelta(days=max_days)] = True
    return days


# ═════════════════════════════════════════════════════════════
# 차트 공통
# ═════════════════════════════════════════════════════════════
def style(fig: go.Figure, height: int = 360, ytitle: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
        font=dict(family="Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif", size=13),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title=ytitle)
    return fig


def diverging_cell(v: float, vmax: float = 0.8) -> str:
    """상관표 셀 배경: 음수는 파랑, 양수는 빨강, 0 근처는 중립 회색 (matplotlib 없이)."""
    if pd.isna(v):
        return ""
    t = min(abs(v) / vmax, 1.0)
    end = (42, 120, 214) if v < 0 else (227, 73, 72)
    mid = (240, 239, 236)
    r, g, b = (round(m + (e - m) * t) for m, e in zip(mid, end))
    ink = "#ffffff" if t > 0.6 else "#0b0b0b"
    return f"background-color: rgb({r},{g},{b}); color: {ink}"


def ccf_bar(c: pd.Series, n: int, title: str) -> go.Figure:
    ci = 1.96 / np.sqrt(max(n, 1))
    colors = [BLUE if abs(v) > ci else "#c3c2b7" for v in c.values]
    fig = go.Figure(go.Bar(x=c.index, y=c.values, marker_color=colors,
                           hovertemplate="lag %{x}일<br>r = %{y:.2f}<extra></extra>"))
    for yv in (ci, -ci):
        fig.add_hline(y=yv, line_dash="dot", line_color=MUTED, line_width=1)
    fig = style(fig, 300, "상관계수 r")
    fig.update_layout(title=title, hovermode="closest")
    fig.update_xaxes(title="lag k (날씨가 k일 먼저)", dtick=1)
    return fig


# ═════════════════════════════════════════════════════════════
# 화면
# ═════════════════════════════════════════════════════════════
wd = load_weather()
D = load_demand()
warn = load_warnings()

st.title("🌦️ weather_fit — 서울 기상 × 패션 수요 EDA")
st.caption("기간 2023-01-01 ~ 2026-07-31 · 서울(관측소 108) 기상 × 네이버 데이터랩 쇼핑인사이트(전국) · "
           "근거 노트북: `preview_eda.ipynb`(T1) · `01_w3_eda.ipynb`(T3·이벤트)")

with st.sidebar:
    st.header("분석 설정")
    baseline_label = st.radio(
        "평소 수준(기준선)",
        ["31일 이동평균 + 요일 보정", "MSTL(7·365일)"],
        help="수요에서 계절·장기 하락을 걷어내는 방법. 31일 이동평균은 preview_eda(T1), MSTL은 01_w3_eda(T3)에서 쓴 방식이다.",
    )
    method = "ma31" if baseline_label.startswith("31") else "mstl"
    pct = st.slider("성수기 기준 (피크월 대비 %)", 20, 80, 40, step=10,
                    help="월평균(추세 제거)이 가장 높은 달의 이 비율 이상인 달만 상관 계산에 쓴다.")
    st.divider()
    st.markdown("**데이터 상태**")
    st.markdown(f"- 기상 관측: {wd.index.min():%Y-%m-%d} ~ {wd.index.max():%Y-%m-%d} ({len(wd):,}일)")
    st.markdown(f"- T1 카테고리 {D['cat_share'].shape[1]}개 · 키워드 {D['keywords'].shape[1]}개")
    st.markdown(f"- 기상특보: {'있음' if warn is not None else '없음 (`data/raw/kma_warning` 수집 필요)'}")

x_st = weather_short_term(wd, method)

# 탭(st.tabs)은 모든 화면의 차트를 한꺼번에 그려 첫 로딩이 느리다 → 고른 화면만 그린다
PAGES = ["요약", "날씨", "수요 추이", "날씨 × 수요", "한파 이벤트", "처리 기준"]
page = st.radio("화면", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()

# 여러 화면에서 함께 쓰는 값
y_cat = demand_short_term(D["cat_share"], method)
y_kw = demand_short_term(D["keywords"], method)
act_cat, act_kw = active_months(D["cat_share"], pct), active_months(D["keywords"], pct)
cold_d0 = rule_first_day(wd["t_min"] <= COLD_THRESHOLD)

# ── 요약 ─────────────────────────────────────────────────────
if page == "요약":
    def in_season_r(y: pd.DataFrame, act: dict, item: str, var: str) -> float:
        m = y.index.month.isin(act[item])
        return y[item][m].corr(x_st[var][m])

    tot = D["total"]
    jj = tot[tot.index.month <= 7]
    yearly = jj.groupby(jj.index.year).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("장갑(T1) × 최저기온", f"{in_season_r(y_cat, act_cat, '장갑', 't_min'):.2f}", help="성수기 단기 변동 상관")
    c2.metric("우산(T1) × 강수", f"{in_season_r(y_cat, act_cat, '우산', 'log_precip'):.2f}", help="성수기 단기 변동 상관")
    c3.metric("한파 이벤트 (≤-10℃, 2일)", f"{len(cold_d0)}건", help="3.6개 겨울. -12℃ 기준이면 4건")
    c4.metric("패션의류 전체 지수 (1~7월)", f"{yearly.iloc[-1] / yearly.iloc[0]:.2f}배",
              help=f"{yearly.index[0]}년 대비 {yearly.index[-1]}년. 쇼핑인사이트 지수 전체의 구조적 하락")

    st.subheader("지금까지 확인한 것")
    st.markdown(
        """
| # | 발견 | 근거 |
|---|---|---|
| 1 | 쇼핑인사이트 지수는 3년간 **전체가 크게 하락**했다. 카테고리끼리·연도끼리 비교하려면 점유율(T1)이나 배치 정렬(T3)이 필요하다 | 수요 추이 |
| 2 | 날씨 효과는 계절을 걷어낸 **단기 변동**에 있다. 원 시계열 상관은 계절 때문에 부풀려진다 | 날씨 × 수요 |
| 3 | 설명력은 **평소 대비 기온 편차**에 있다. **전일 대비 변화(Δt)는 일 단위로는 거의 0**이다 | 날씨 × 수요 → 변수 비교 |
| 4 | 장갑·롱패딩 등 방한 품목은 **추워진 당일·다음 날**, 우산은 **비 온 당일**에 반응한다 | 날씨 × 수요 → 시차 상관 |
| 5 | 한파 -12℃ 기준은 4건뿐이라 **-10℃로 재정의**했다(9건, 모두 특보 기간과 겹침) | 한파 이벤트 |
"""
    )
    st.info("위 지표는 사이드바의 기준선·성수기 설정에 따라 다시 계산된다. 두 기준선에서 결론이 같은지 확인하는 용도로 쓴다.")

# ── 날씨 ─────────────────────────────────────────────────────
if page == "날씨":
    st.subheader("서울 일별 기온·강수")
    roll = st.checkbox("7일 평균으로 보기", value=False, key="wx_roll")
    w7 = wd.rolling(7, center=True).mean() if roll else wd
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w7.index, y=w7["t_max"], name="최고기온", line=dict(color=ORANGE, width=1.5)))
    fig.add_trace(go.Scatter(x=w7.index, y=w7["t_min"], name="최저기온", line=dict(color=BLUE, width=1.5)))
    fig.add_hline(y=COLD_THRESHOLD, line_dash="dot", line_color=BLUE, line_width=1,
                  annotation_text=f"한파 기준 {COLD_THRESHOLD}℃", annotation_font_color=MUTED)
    fig.add_hline(y=33, line_dash="dot", line_color=ORANGE, line_width=1,
                  annotation_text="폭염 기준 33℃", annotation_font_color=MUTED)
    st.plotly_chart(style(fig, 360, "℃"), use_container_width=True)

    fig = go.Figure(go.Bar(x=wd.index, y=wd["precip_mm"], marker_color=BLUE, name="강수량",
                           hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} mm<extra></extra>"))
    st.plotly_chart(style(fig, 220, "mm"), use_container_width=True)

    st.subheader("월별 일교차와 Δt")
    m = wd.groupby(wd.index.month).agg(일교차_평균=("dtr", "mean"), Δt_min_표준편차=("dt_min", "std"))
    m.index = [f"{i}월" for i in m.index]
    col_a, col_b = st.columns(2)
    for col, c, color, unit in [(col_a, "일교차_평균", ORANGE, "℃"), (col_b, "Δt_min_표준편차", BLUE, "℃")]:
        f = go.Figure(go.Bar(x=m.index, y=m[c], marker_color=color,
                             hovertemplate="%{x}<br>%{y:.1f}℃<extra></extra>"))
        f = style(f, 280, unit)
        f.update_layout(title=c.replace("_", " "), hovermode="closest")
        col.plotly_chart(f, use_container_width=True)
    st.caption("일교차는 봄(3–5월)에 가장 크고 장마철(7–8월)에 가장 작다. 기온 급변(Δt의 변동폭)은 겨울에 몰려 있다. "
               "일교차도 계절성이 있으므로 수요와 비교하기 전에 평소 수준을 빼야 한다.")

    fe = load_forecast_error()
    if fe is not None:
        st.subheader("중기예보 vs 관측 오차 (기간별)")
        cmp = fe.join(wd[["t_min", "t_max"]], rsuffix="_obs").dropna()
        cmp["기간"] = np.where(cmp.index < LEAD_CHANGE, "A: ~2024-12-01 (D-3 발표)", "B: 2024-12-02~ (D-4 발표)")
        err = cmp.groupby("기간")[["t_min", "t_max", "t_min_obs", "t_max_obs"]].apply(lambda d: pd.Series({
            "일수": len(d),
            "최저기온 MAE(℃)": (d.t_min - d.t_min_obs).abs().mean(),
            "최고기온 MAE(℃)": (d.t_max - d.t_max_obs).abs().mean(),
            "예보Δt ~ 관측Δt 상관": d.t_min.diff().corr(d.t_min_obs.diff()),
        })).round(2)
        st.dataframe(err, use_container_width=True)
        st.caption("2024-12-02부터 중기예보 최소 리드타임이 늘어 D-4 발표가 선택된다. 리드타임이 길어진 B 기간은 오차가 크다. "
                   "W5 모델에서 두 기간을 구분해 다룬다.")

# ── 수요 추이 ─────────────────────────────────────────────────
if page == "수요 추이":
    src = st.radio("데이터", ["T1 중분류 카테고리", "키워드 (T3 + T4 대체)"], horizontal=True, key="dem_src")
    if src.startswith("T1"):
        items = st.multiselect("카테고리 (최대 3개)", list(D["cat_share"].columns), default=["장갑", "우산", "수영"],
                               max_selections=3, key="dem_cat")
        view = st.radio("값", ["점유율(% of 패션의류 전체)", "원래 ratio"], horizontal=True, key="dem_view")
        data = D["cat_share"] if view.startswith("점유율") else D["cat_ratio"]
        ylab = "%" if view.startswith("점유율") else "ratio"
        st.caption("원래 ratio로 보면 모든 카테고리가 함께 내려간다(공통 하락). 점유율로 바꾸면 그 하락이 상쇄된다.")
    else:
        items = st.multiselect("키워드 (최대 3개)", list(D["keywords"].columns), default=["패딩", "우산", "반바지"],
                               max_selections=3, key="dem_kw")
        data, ylab = D["keywords"], "level (배치 정렬)"
        st.caption("T4로 대체한 키워드: " + ", ".join(D["subs"]) + " · 완전 제외: "
                   + ", ".join(sorted(set(D["dropped"]) - set(D["subs"]))))

    if items:
        scale = st.toggle("항목별 평균 = 100 으로 맞춰 모양만 비교", value=True, key="dem_scale",
                          help="항목마다 크기가 수십 배 달라 작은 항목이 바닥에 깔린다. 켜면 각 항목을 자기 평균 대비 지수로 바꾼다.")
        sm = data[items].rolling(7, center=True).mean()
        if scale:
            sm = sm / data[items].mean() * 100
            ylab = "지수 (항목 평균 = 100)"
        fig = go.Figure()
        for i, k in enumerate(items):
            fig.add_trace(go.Scatter(x=sm.index, y=sm[k], name=k, line=dict(color=SERIES[i], width=2)))
        fig = style(fig, 380, ylab)
        fig.update_layout(title="7일 이동평균")
        st.plotly_chart(fig, use_container_width=True)

        jj = data[items][data.index.month <= 7]
        yr = jj.groupby(jj.index.year).mean()
        st.markdown("**연도별 1~7월 평균 (첫해 = 1.0)** — 2026년은 7월까지만 있어 같은 달끼리 비교한다")
        st.dataframe((yr / yr.iloc[0]).round(2).T, use_container_width=True)

    st.subheader("월별 계절 패턴")
    st.caption("각 시계열을 365일 중심 이동평균으로 나눠 장기 하락을 걷어낸 뒤, 월평균을 항목별로 z-score화했다. 빨강일수록 그 달에 뜬다.")
    si = data / data.rolling(365, center=True, min_periods=180).mean()
    mm = si.groupby(si.index.month).mean()
    z = ((mm - mm.mean()) / mm.std()).T
    z = z.loc[z.idxmax(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(z=z.values, x=[f"{m}월" for m in z.columns], y=z.index, colorscale=DIVERGING,
                               zmid=0, zmin=-2, zmax=2, hovertemplate="%{y} · %{x}<br>z = %{z:.2f}<extra></extra>"))
    fig = style(fig, max(380, 22 * len(z)), "")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ── 날씨 × 수요 ───────────────────────────────────────────────
if page == "날씨 × 수요":
    c_a, c_b, c_c = st.columns([1.2, 1.5, 1.5])
    src2 = c_a.radio("데이터", ["T1 카테고리(점유율)", "키워드(level)"], key="link_src")
    if src2.startswith("T1"):
        base, y_all, act = D["cat_share"], y_cat, act_cat
        default_item = "장갑"
    else:
        base, y_all, act = D["keywords"], y_kw, act_kw
        default_item = "롱패딩"
    item = c_b.selectbox("항목", list(base.columns), index=list(base.columns).index(default_item))
    var = c_c.selectbox("날씨 변수", list(WEATHER_VARS), format_func=lambda v: WEATHER_VARS[v],
                        index=list(WEATHER_VARS).index("log_precip" if item in ("우산", "레인부츠", "레인코트") else "t_min"))

    mask = pd.Series(y_all.index.month.isin(act[item]), index=y_all.index)
    st.caption(f"성수기: {', '.join(f'{m}월' for m in act[item])} ({int(mask.sum())}일) · "
               f"기준선: {baseline_label}")

    raw_r = np.log1p(base[item]).corr(wd[var])
    st_r = y_all[item][mask].corr(x_st[var][mask])
    k1, k2, k3 = st.columns(3)
    k1.metric("원 시계열 상관 (전 기간)", f"{raw_r:.2f}", help="계절이 섞여 부풀려진 값")
    k2.metric("단기 변동 상관 (성수기)", f"{st_r:.2f}", help="평소 수준을 뺀 뒤의 상관 = 날씨 효과")
    if var in ("t_min", "t_max"):
        d = pd.concat([y_all[item][mask], x_st[var][mask]], axis=1).dropna()
        slope = np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0]
        k3.metric("평소보다 1℃ 추울 때", f"{(np.exp(-slope) - 1) * 100:+.1f}%", help="로그 단기 변동 ~ 기온 편차 기울기")

    col_l, col_r = st.columns(2)
    d = pd.DataFrame({"x": x_st[var], "y": y_all[item]})[mask].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="markers", name=item,
                             marker=dict(color=BLUE, size=8, opacity=0.45, line=dict(color="white", width=1)),
                             text=d.index.strftime("%Y-%m-%d"),
                             hovertemplate="%{text}<br>날씨 %{x:.2f}<br>수요 %{y:.2f}<extra></extra>"))
    if len(d) > 2:
        b1, b0 = np.polyfit(d["x"], d["y"], 1)
        xs = np.linspace(d["x"].min(), d["x"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=b0 + b1 * xs, name="추세선", line=dict(color=ORANGE, width=2)))
    fig = style(fig, 360, "수요 단기 변동 (로그)")
    fig.update_layout(title="성수기 산점도", hovermode="closest")
    fig.update_xaxes(title=f"{WEATHER_VARS[var]} (단기 변동)")
    col_l.plotly_chart(fig, use_container_width=True)

    y_m = y_all[item].where(mask)
    lag_view = col_r.radio("시차 상관", ["수준", "차분 (지속성 제거)"], horizontal=True, key="lag_view")
    c = ccf(y_m, x_st[var]) if lag_view == "수준" else ccf(y_all[item].diff().where(mask), x_st[var].diff())
    col_r.plotly_chart(ccf_bar(c, int(mask.sum()), f"시차 상관 — {lag_view}"), use_container_width=True)
    st.caption("파란 막대만 유의선(±1.96/√n) 밖이다. 수준 상관은 날씨가 며칠씩 이어져서 넓게 퍼지므로, "
               "'며칠째에 반응하나'는 차분 쪽으로 판단한다.")

    st.subheader("겨울·여름별 일관성")
    season_id = (y_all.index.year - (y_all.index.month < 7)).astype(str) if 12 in act[item] or 1 in act[item] \
        else y_all.index.year.astype(str)
    cons = pd.DataFrame({"y": y_all[item], "x": x_st[var], "g": season_id})[mask.values].dropna()
    tbl = cons.groupby("g")[["y", "x"]].apply(lambda g: pd.Series({"일수": len(g), "상관": g["y"].corr(g["x"])})).round(2)
    tbl.index.name = "시즌(시작 연도)"
    st.dataframe(tbl.T, use_container_width=True)
    st.caption("한 시즌이 결과 전체를 끌고 가는지 확인한다. 모든 시즌에서 부호가 같아야 믿을 만하다.")

    st.subheader("변수 비교: 어떤 날씨 변수가 가장 잘 설명하나")
    rows = {}
    for k in base.columns:
        m = y_all.index.month.isin(act[k])
        r = {v: y_all[k][m].corr(x_st[v][m]) for v in WEATHER_VARS}
        r["최강 변수"] = WEATHER_VARS[max(r, key=lambda v: abs(r[v]) if pd.notna(r[v]) else -1)]
        r["성수기 일수"] = int(m.sum())
        if not src2.startswith("T1"):
            r["소스"] = D["kw_source"][k]
        rows[k] = r
    screen = pd.DataFrame(rows).T
    screen = screen.rename(columns=WEATHER_VARS)
    num_cols = list(WEATHER_VARS.values())
    screen[num_cols] = screen[num_cols].astype(float)
    screen = screen.reindex(screen[num_cols].abs().max(axis=1).sort_values(ascending=False).index)
    st.dataframe(screen.style.format("{:.2f}", subset=num_cols).map(diverging_cell, subset=num_cols),
                 use_container_width=True, height=420)
    avg = screen[num_cols].abs().mean().sort_values(ascending=False).round(3)
    st.markdown("**변수별 평균 |r|**: " + " · ".join(f"{k} {v}" for k, v in avg.items()))
    st.caption("핵심 가설은 '절대 기온보다 Δt·일교차'였지만, 데이터는 평소 대비 기온 편차가 가장 강하고 Δt는 거의 0이다. "
               "우산류의 일교차 상관은 '비 오는 날은 일교차가 작다'는 대리 효과로 본다. "
               "|r| 0.2 미만은 여러 조합을 훑은 결과라 우연일 수 있다.")

# ── 한파 이벤트 ───────────────────────────────────────────────
if page == "한파 이벤트":
    st.subheader(f"한파 D0 = 최저기온 ≤ {COLD_THRESHOLD}℃ 가 2일 이상 이어지는 첫날")
    idx = wd.index
    cold_flag = warning_days(warn, "C", idx) if warn is not None else None
    ev = pd.DataFrame({
        "D0": cold_d0.date,
        "D0 최저기온(℃)": wd.loc[cold_d0, "t_min"].values,
        "-12℃ 이하 동반": [bool((wd["t_min"].loc[d:d + pd.Timedelta(days=3)] <= -12).any()) for d in cold_d0],
    })
    if cold_flag is not None:
        ev["특보 발효(±2일)"] = [bool(cold_flag.loc[d - pd.Timedelta(days=2): d + pd.Timedelta(days=2)].any()) for d in cold_d0]

    col_l, col_r = st.columns([1, 1.4])
    col_l.dataframe(ev, use_container_width=True, hide_index=True)
    comp = pd.DataFrame({
        "기준": ["-12℃", "-10℃", "-8℃"],
        "이벤트 수": [len(rule_first_day(wd["t_min"] <= t)) for t in (-12, -10, -8)],
    })
    col_r.dataframe(comp, use_container_width=True, hide_index=True)
    col_r.caption("-12℃ 기준은 3.6개 겨울에 4건(2024-25 겨울 0건)이라 목표(연 2~5회)에 못 미쳐 -10℃로 재정의했다. "
                  "-10℃ 기준 9건은 모두 실제 한파특보 기간과 겹친다.")
    if warn is None:
        st.warning("`data/raw/kma_warning` 이 없어 특보 대조는 생략했다. 수집기를 돌리면 표에 특보 열이 생긴다.")

    st.subheader("이벤트 윈도우 미리보기 (D-7 ~ D+14) — W4 예정")
    src3 = st.radio("데이터", ["키워드(level)", "T1 카테고리(점유율)"], horizontal=True, key="cold_src")
    yb = y_kw if src3.startswith("키워드") else y_cat
    default = "롱패딩" if src3.startswith("키워드") else "장갑"
    item3 = st.selectbox("항목", list(yb.columns), index=list(yb.columns).index(default), key="cold_item")
    offsets = range(-7, 15)
    mat = pd.DataFrame({d.date(): [yb[item3].get(d + pd.Timedelta(days=o), np.nan) for o in offsets] for d in cold_d0},
                       index=list(offsets))
    fig = go.Figure()
    for c in mat.columns:
        fig.add_trace(go.Scatter(x=mat.index, y=(np.exp(mat[c]) - 1) * 100, name=str(c), showlegend=False,
                                 line=dict(color="#c3c2b7", width=1),
                                 hovertemplate=f"{c}<br>D%{{x:+d}}: %{{y:+.0f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=mat.index, y=(np.exp(mat.mean(axis=1)) - 1) * 100, name=f"{len(mat.columns)}건 평균",
                             line=dict(color=BLUE, width=3), hovertemplate="평균 D%{x:+d}: %{y:+.1f}%<extra></extra>"))
    fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=0, line_color=GRID)
    fig = style(fig, 380, "평소 대비 수요 변화 (%)")
    fig.update_xaxes(title="한파 D0 기준 일수", dtick=1)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("회색 선은 이벤트 하나하나, 파란 선은 평균이다. y값은 계절·장기 하락을 뺀 단기 변동이라 "
               "'평소보다 몇 % 더 찾았나'로 읽는다. 사례가 9건뿐이라 개별 선이 평균을 좌우하지 않는지 함께 본다.")

# ── 처리 기준 ─────────────────────────────────────────────────
if page == "처리 기준":
    st.subheader("이 대시보드에 적용한 처리 기준")
    st.markdown(
        f"""
| 항목 | 기준 | 이유 |
|---|---|---|
| 기상 결측 | 기온은 -99만 결측, 습도·풍속·운량은 -9·-99 결측, 강수·적설은 -9 → 0 | 서울 겨울 -9℃는 실제 기온일 수 있다 |
| T1 지표 | 카테고리 ÷ 패션의류 전체 × 100 (점유율) | 지수 전체의 공통 하락을 상쇄한다 |
| T3 지표 | 배치별 상수 배율로 정렬한 `level` | 배치 차이는 날짜와 무관한 상수다. `rescaled`는 앵커(반팔티)의 계절·노이즈를 주입한다 |
| 희소 키워드 | T3 커버리지 95% 미만은 T4로 대체({len(D['subs'])}개), 기모레깅스는 제외 | 조회한 대분류(cid)가 맞지 않아 생긴 결측이다 |
| 평소 수준 | 31일 이동평균+요일 보정 / MSTL 중 선택 (사이드바) | 본 분석 기준은 MSTL, 31일 이동평균은 비교용 |
| 성수기 | 추세 제거 월평균이 피크월의 {pct}% 이상인 달 | 비수기 값은 작아서 노이즈가 크다 |
| 한파 D0 | 최저기온 ≤ -10℃ 2일 이상 지속 첫날 | -12℃는 표본 부족(4건) |
| 예보 | 2024-12-02 전후 A/B 기간 구분 | 중기예보 리드타임이 3일 → 4일로 바뀌었다 |
"""
    )
    st.subheader("한계")
    st.markdown(
        "- 쇼핑 데이터는 **전국** 단위다. 서울 기상 × 전국 수요 구조라는 점을 해석할 때 감안한다.\n"
        "- 성수기 겨울은 3.6번뿐이다. 상관의 정확한 크기는 확정하기 어렵고, 상관이 인과를 증명하지 않는다.\n"
        "- 공휴일·할인 행사 같은 달력 요인은 아직 보정하지 않았다."
    )
