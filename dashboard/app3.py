"""
실행: 저장소 루트에서  python -m streamlit run dashboard/app3.py

[구성] COVER / 01 날씨 / 02 수요 추이 / 03 날씨 × 수요 / 04 한파 이벤트
[템플릿] 이전 app.py 의 디자인(색·글꼴·표지)을 그대로 쓴다.
[화면 내용] 멘토님 앱의 4개 화면과 같은 구성이다. MSTL은 하지 않았으므로 쓰지 않는다.
[데이터] 노트북이 data/dashboard/ 에 저장한 CSV를 읽는다. 성수기 기준과 요일 보정은 앱에서 바꿔 가며 다시 계산한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='WEATHER_FIT · Report', layout='centered')
DATA = Path(__file__).resolve().parents[1] / 'data' / 'dashboard'     # 어디서 실행해도 같은 경로

# ---------- 색 (app.py 템플릿과 같음) ----------
BG, INK, BURGUNDY, GRID = '#FAF7F2', '#2B2B2B', '#8E2C2C', '#E6DED0'
COLD, HEAT, GRAY, GOLD = '#2F5D8A', '#C0502D', '#8C8C8C', '#D9A441'
BLUE, ORANGE, RED, NEUTRAL, MUTED = COLD, HEAT, BURGUNDY, '#F1EADF', GRAY      # 화면 코드에서 쓰는 이름
SERIES = [COLD, HEAT, GOLD]
ANCHORS = ['반팔티', '양말', '립스틱', '등산화']      # 묶음 크기를 맞추려고 넣은 기준 키워드라 분석 대상이 아님

WEATHER_VARS = {
    't_min': '최저기온',
    't_max': '최고기온',
    'dtr': '일교차',
    'dt_min': '전일 대비 최저기온 변화(Δt)',
    'dt_min_3d': '3일 누적 최저기온 변화',
    'log_precip': 'log(1+강수량)',
}

# ---------- 디자인 ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;600;700;900&display=swap');
div.block-container {max-width: 980px !important; padding-top: 2.5rem;}
h1, h2, h3 {font-family: 'Noto Serif KR', serif; letter-spacing: -0.01em;}
.chap-no {color:#8E2C2C; font-size:0.8rem; letter-spacing:0.25em; font-weight:700; margin-bottom:-0.4rem;}
.mast {font-family:'Noto Serif KR', serif; font-weight:900; font-size:3.2rem; letter-spacing:0.08em; line-height:1.1; margin:0;}
.sub {font-family:'Noto Serif KR', serif; font-size:1.15rem; color:#555; margin:0.2rem 0 0.4rem 0;}
.question {font-family:'Noto Serif KR', serif; font-size:1.3rem; line-height:1.6; margin:0.2rem 0 1rem 0;}
.pull {font-family:'Noto Serif KR', serif; font-size:1.2rem; line-height:1.65; border-left:4px solid #8E2C2C; padding:0.4rem 1rem; margin:1.2rem 0;}
.nextup {color:#7A7A7A; font-style:italic; margin-top:2rem;}
[data-testid="stHeaderActionElements"], a.header-anchor {display:none !important;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def read(name: str, **kw) -> pd.DataFrame | None:
    path = DATA / name
    return pd.read_csv(path, **kw) if path.exists() else None


def need_file(name: str, hint: str):
    st.error(f"`data/dashboard/{name}` 이 없습니다. {hint}")
    st.stop()


@st.cache_data
def load_weather() -> pd.DataFrame:
    w = read("weather_daily.csv", index_col="date", parse_dates=True)
    if w is None:
        need_file("weather_daily.csv", "노트북의 '대시보드용 데이터 저장' 셀을 먼저 실행하세요.")
    w = w.asfreq("D")
    if "dtr" not in w:
        w["dtr"] = w["t_max"] - w["t_min"]
    w["dt_min_3d"] = w["t_min"].diff(3)                 # 3일 누적 변화 (노트북 T1-12a)
    w["log_precip"] = np.log1p(w["precip_mm"])
    return w


@st.cache_data
def load_demand() -> dict:
    kw = read("demand_t3.csv", index_col="date", parse_dates=True)
    if kw is None:
        need_file("demand_t3.csv", "노트북의 '대시보드용 데이터 저장' 셀을 먼저 실행하세요.")
    kw = kw.drop(columns=[a for a in ANCHORS if a in kw.columns]).asfreq("D")
    share = read("t1_share.csv", index_col="date", parse_dates=True)     # 선택: 없으면 키워드만 쓴다
    wide = read("t1_wide.csv", index_col="date", parse_dates=True)
    return {"keywords": kw, "t1_share": share, "t1_wide": wide}


@st.cache_data
def load_warning() -> pd.DataFrame | None:
    return read("warning_daily.csv", index_col="date", parse_dates=True)


@st.cache_data
def load_forecast() -> pd.DataFrame | None:
    return read("forecast_error.csv", index_col="date", parse_dates=True)


# ═════════════════════════════════════════════════════════════
# 변환 (노트북과 같은 정의)
# ═════════════════════════════════════════════════════════════
def short_term(s: pd.Series, window: int = 31) -> pd.Series:
    """앞뒤 window일 평균(= 그 시기 평소 수준)을 뺀다. 계절·장기 하락이 사라지고 그날만의 편차가 남는다."""
    return s - s.rolling(window, center=True, min_periods=15).mean()


def log_level(s: pd.Series) -> pd.Series:
    """0이 있어도 log가 되도록 양수 값의 하위 1%를 더한다 (노트북 T3와 같음)."""
    eps = max(s[s > 0].quantile(0.01), 1e-3)
    return np.log(s + eps)


@st.cache_data
def demand_short_term(df: pd.DataFrame, floor: bool, dow: bool) -> pd.DataFrame:
    """수요의 평소 대비 변동. floor: T3 키워드(0 있음) → 로그 바닥값 / T1 점유율 → 그냥 로그. dow: 요일 평균 제거."""
    y = (df.apply(log_level) if floor else np.log(df)).apply(short_term)
    return y - y.groupby(y.index.dayofweek).transform("mean") if dow else y


@st.cache_data
def weather_short_term(w: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=w.index)
    for v in ["t_min", "t_max", "dtr", "log_precip"]:
        x[v] = short_term(w[v])          # 계절성이 있는 변수는 평소 수준을 먼저 뺀다
    x["dt_min"] = w["dt_min"]            # 변화량은 이미 계절 성분이 거의 없다
    x["dt_min_3d"] = w["dt_min_3d"]
    return x[list(WEATHER_VARS)]


def season_index(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """월별 계절지수 (행=월, 열=항목).
    t1: 점유율의 월평균 (노트북 T1-9)  /  kw: 365일 평균 대비 비율의 월평균 (노트북 T3)"""
    if kind == "t1":
        return df.groupby(df.index.month).mean()
    return (df / df.rolling(365, center=True, min_periods=180).mean()).groupby(df.index.month).mean()


def active_months(si: pd.DataFrame, pct: float) -> dict[str, list[int]]:
    """성수기 = 계절지수가 최고 달의 pct% 이상인 달."""
    return {k: [m for m in si.index if si.loc[m, k] >= pct / 100 * si[k].max()] for k in si.columns}


def ccf(y: pd.Series, x: pd.Series, max_lag: int = 7) -> pd.Series:
    return pd.Series({k: y.corr(x.shift(k)) for k in range(-max_lag, max_lag + 1)})


def rule_first_days(cond: pd.Series, min_len: int = 2) -> pd.DatetimeIndex:
    """cond가 min_len일 이상 연속되는 구간의 첫날 (노트북과 같은 함수)."""
    cond = cond.fillna(False).astype(bool)
    run_end = cond.astype(int).rolling(min_len).sum().eq(min_len)
    in_run = run_end.copy()
    for s in range(1, min_len):
        in_run |= run_end.shift(-s, fill_value=False)
    return in_run.index[in_run & ~in_run.shift(1, fill_value=False)]


# ═════════════════════════════════════════════════════════════
# 차트·문단 공통 (app.py 템플릿)
# ═════════════════════════════════════════════════════════════
def style(fig, height=420, ytitle=''):
    # 모든 차트의 배경·글꼴·격자를 같은 톤으로 맞춘다.
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG, hovermode='x unified',
                      font=dict(family='Noto Serif KR, serif', color=INK, size=13),
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation='h', y=-0.22))
    fig.update_xaxes(showline=True, linecolor=INK, gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, title=ytitle)
    return fig


def chapter(no, title, subtitle):
    st.markdown(f"<div class='chap-no'>{no}</div>", unsafe_allow_html=True)
    st.markdown(f'# {title}')
    st.markdown(f"<div class='question'>{subtitle}</div>", unsafe_allow_html=True)


def pull(text):
    st.markdown(f"<div class='pull'>{text}</div>", unsafe_allow_html=True)


def nextup(text):
    st.markdown(f"<div class='nextup'>다음 장 → {text}</div>", unsafe_allow_html=True)


def table(df):
    st.dataframe(df, hide_index=True, use_container_width=True)


def diverging_cell(v, vmax=0.8):
    """상관표 셀 배경: 음수는 파랑, 양수는 주황, 0 근처는 종이색."""
    if pd.isna(v):
        return ''
    t = min(abs(v) / vmax, 1.0)
    end = (47, 93, 138) if v < 0 else (192, 80, 45)
    mid = (250, 247, 242)
    r, g, b = (round(m + (e - m) * t) for m, e in zip(mid, end))
    return f"background-color: rgb({r},{g},{b}); color: {'#ffffff' if t > 0.6 else INK}"


def ccf_bar(c, n, title):
    ci = 1.96 / np.sqrt(max(n, 1))
    colors = [COLD if abs(v) > ci else '#C9C2B4' for v in c.values]
    fig = go.Figure(go.Bar(x=c.index, y=c.values, marker_color=colors,
                           hovertemplate='lag %{x}일<br>r = %{y:.2f}<extra></extra>'))
    for yv in (ci, -ci):
        fig.add_hline(y=yv, line_dash='dot', line_color=MUTED, line_width=1)
    fig = style(fig, 300, '상관계수 r')
    fig.update_layout(title=title, hovermode='closest')
    fig.update_xaxes(title='lag k (날씨가 k일 먼저)', dtick=1)
    return fig


# ======================================================================
# COVER
# ======================================================================
def page_cover():
    st.markdown("<div class='chap-no'>REPORT · SEOUL · 2023.01 – 2026.07</div>", unsafe_allow_html=True)
    st.markdown("<div class='mast'>WEATHER_FIT</div>", unsafe_allow_html=True)

    pull('소비자는 단순 절대 기온보다 체감 온도, 일교차, 전날 대비 기온 변화량 및 한파/폭염/호우 특보 등 기상 이벤트 발생 시 특정 패션 카테고리를 더 적극적으로 검색하고 소비할 것을 가설로 시작했습니다.')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('분석 기간', '1,308일')
    c2.metric('쇼핑 키워드', '30개')
    c3.metric('한파·폭염 특보', '각 20건')
    c4.metric('날씨에 가장 민감한 상품', '롱패딩 -0.85')

    st.markdown('## 목차')
    table(pd.DataFrame({
        '': ['01', '02', '03', '04', '05'],
        '제목': ['날씨', '수요 추이', '날씨 × 수요', '한파 이벤트', '결론'],
        '내용': ['서울 3년 7개월의 기온·강수·예보 오차', '해마다 줄어든 쇼핑 지수와 키워드의 달력',
                     '평소 대비 날씨와 수요의 상관, 시차, 겨울별 일관성', '몇 도부터 한파인가, 한파 전후의 수요',
                     '확인된 것, 틀렸던 가설, 한계']}))
    st.caption('성수기 기준(30·40·50·60%)은 03 날씨 × 수요 화면에서, 요일 보정은 03·04 화면에서 바꿔 볼 수 있습니다.')
    st.markdown('## 날씨에서 계절을 뺀 이유')
    st.markdown("""아이스크림 판매량과 물놀이 사고는 둘 다 여름에 늘어서 같이 움직입니다. 그렇다고 아이스크림이 사고를 일으키는 건 아닙니다. \
장갑 클릭과 최저기온을 그대로 비교하면 상관이 -0.71로 꽤 강해 보이지만, 이 숫자에는 "겨울이라서 춥고, 겨울이라서 장갑을 찾는다"는 계절 효과가 섞여 있습니다.

그래서 이 리포트는 모든 값에서 그 시기의 평소 수준(앞뒤 31일 평균)을 뺐습니다. 남는 것은 "평소보다 얼마나 추웠나"와 "평소보다 얼마나 더 찾았나"입니다. \
계절을 빼자 장갑의 상관은 -0.46으로 줄었고, 수영과 기온의 상관은 0.63에서 0.15로 떨어졌습니다. \
반대로 비는 계절이 아니라 그날의 사건이어서, 계절을 빼도 우산의 반응이 오히려 선명해졌습니다(0.49 → 0.62).""")
    st.markdown('## 두 가지 원칙')
    st.markdown('1. 계절에 의한 가짜 상관을 피하기 위해, 모든 값을 **"평소 수준 대비"** 로 본다.\n'
                '2. 분석 기준은 **실제 날씨(관측)**, 모델 입력 후보는 **예보**로 구분해 "그날 실제로 알 수 있었던 정보"의 오차를 방어한다.')


# ═════════════════════════════════════════════════════════════
# 데이터와 사이드바
# ═════════════════════════════════════════════════════════════
wd = load_weather()
D = load_demand()
warn = load_warning()
fc = load_forecast()
has_t1 = D['t1_share'] is not None

PAGES_ORDER = ['COVER', '01 · 날씨', '02 · 수요 추이', '03 · 날씨 × 수요', '04 · 한파 이벤트', '05 · 결론']
st.sidebar.markdown('### WEATHER_FIT')
st.sidebar.caption('Seoul · Weather & Wardrobe')
if st.session_state.get('menu') not in PAGES_ORDER:
    st.session_state['menu'] = PAGES_ORDER[0]
for _p in PAGES_ORDER:
    if st.sidebar.button(_p, key=f'nav_{_p}', use_container_width=True,
                         type='primary' if st.session_state['menu'] == _p else 'secondary'):
        st.session_state['menu'] = _p
        st.rerun()
menu = st.session_state['menu']

with st.sidebar:
    st.divider()
    st.caption('평소 수준: 앞뒤 31일 이동평균')
    st.caption(f"기상 {len(wd):,}일 · 키워드 {D['keywords'].shape[1]}개" + (f" · T1 {D['t1_share'].shape[1]}개" if has_t1 else ' · T1 없음'))
    st.caption(f"특보 {'있음' if warn is not None else '없음'} · 예보 오차 {'있음' if fc is not None else '없음'}")
    if not has_t1:
        st.warning('T1 카테고리는 t1_share.csv, t1_wide.csv 가 있어야 보입니다.')

x_st = weather_short_term(wd)                        # 날씨의 평소 대비 편차 (설정과 무관)
si_kw = season_index(D['keywords'], 'kw')            # 월별 계절지수 (성수기 기준과 무관, 기준은 화면에서 적용)
if has_t1:
    si_cat = season_index(D['t1_share'], 't1')


# ═════════════════════════════════════════════════════════════
# 장
# ═════════════════════════════════════════════════════════════
def page_ch01():
    chapter('CHAPTER 01', '날씨', '3년 7개월의 서울의 날씨')
    st.subheader("서울 일별 기온·강수")
    roll = st.checkbox("7일 평균으로 보기", value=False, key="wx_roll")
    w7 = wd.rolling(7, center=True).mean() if roll else wd
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w7.index, y=w7["t_max"], name="최고기온", line=dict(color=ORANGE, width=1.5)))
    fig.add_trace(go.Scatter(x=w7.index, y=w7["t_min"], name="최저기온", line=dict(color=BLUE, width=1.5)))
    fig.add_hline(y=-10, line_dash="dot", line_color=BLUE, line_width=1,
                  annotation_text="한파 기준 -10℃", annotation_font_color=MUTED)
    fig.add_hline(y=33, line_dash="dot", line_color=ORANGE, line_width=1,
                  annotation_text="폭염 기준 33℃", annotation_font_color=MUTED)
    st.plotly_chart(style(fig, 360, "℃"), use_container_width=True)

    # 강수: 7일 단위 히스토그램 (7일 칸마다 강수량 합산)
    fig = go.Figure(go.Histogram(
        x=wd.index, y=wd["precip_mm"], histfunc="sum",
        xbins=dict(size=7 * 24 * 3600 * 1000), marker_color=BLUE, name="주간 강수량",
        hovertemplate="%{x|%Y-%m-%d}부터 7일<br>합계 %{y:.1f} mm<extra></extra>"))
    st.plotly_chart(style(fig, 220, "7일 합계 mm"), use_container_width=True)

    st.subheader("월별 일교차와 전날과의 기온 변화")
    NAME_A = "일교차 평균"
    NAME_B = "최저기온 변화의 표준편차"
    m = wd.groupby(wd.index.month).agg(**{
        NAME_A: ("dtr", "mean"),
        NAME_B: ("dt_min", "std"),
    })
    m.index = [f"{i}월" for i in m.index]
    col_a, col_b = st.columns(2)
    for col, c, color in [(col_a, NAME_A, ORANGE), (col_b, NAME_B, BLUE)]:
        f = go.Figure(go.Bar(x=m.index, y=m[c], marker_color=color,
                             hovertemplate="%{x}<br>%{y:.1f}℃<extra></extra>"))
        f = style(f, 280, "℃")
        f.update_layout(title=c, hovermode="closest")
        col.plotly_chart(f, use_container_width=True)
    st.caption("일교차는 봄(3–5월)에 가장 크고 장마철(7–8월)에 가장 작다. 기온 급변(전날 기온의 변동폭)은 겨울에 몰려 있다. "
               "일교차도 계절성이 있으므로 수요와 비교하기 전에 평소 수준을 빼야 한다.")

    if fc is not None:
        st.subheader("중기예보 vs 관측 오차 (기간별)")
        cmp = fc.dropna(subset=["t_min", "t_max", "t_min_obs", "t_max_obs"]).copy()
        cmp["기간"] = np.where(cmp["lead_days"] <= 3, "A: 3일 앞 발표", "B: 4일 앞 발표")
        err = cmp.groupby("기간").apply(lambda d: pd.Series({
            "일수": len(d),
            "최저기온 MAE(℃)": (d.t_min - d.t_min_obs).abs().mean(),
            "최고기온 MAE(℃)": (d.t_max - d.t_max_obs).abs().mean(),
            "예보 기온변화 ~ 관측 기온변화 상관": d.t_min.diff().corr(d.t_min_obs.diff()),
        })).round(2)
        st.dataframe(err, use_container_width=True)
        st.caption("중기예보는 3일 뒤부터 발표되고, 2024-11-29 발표분부터 4일 뒤로 늘었다. 리드타임이 길어진 B 기간은 오차가 크다. "
                   "이 데이터에는 '내일 예보'가 없다.")

def page_ch02():
    chapter('CHAPTER 02', '수요 추이', '쇼핑 지수는 어떻게 움직였나')
    options = (["T1 중분류 카테고리"] if has_t1 else []) + ["키워드 (T3)"]
    src = st.radio("데이터", options, horizontal=True, key="dem_src")
    if src.startswith("T1"):
        pool = list(D["t1_share"].columns)
        items = st.multiselect("카테고리 (최대 3개)", pool, default=[k for k in ["장갑", "우산", "수영"] if k in pool],
                               max_selections=3, key="dem_cat")
        view = st.radio("값", ["점유율(% of 패션의류 전체)", "원래 ratio"], horizontal=True, key="dem_view")
        if view.startswith("점유율"):
            data, ylab, kind = D["t1_share"], "%", "t1"
        else:
            data, ylab, kind = D["t1_wide"].drop(columns="패션의류_전체", errors="ignore"), "ratio", "t1"
        st.caption("원래 ratio로 보면 모든 카테고리가 함께 내려간다(공통 하락). 점유율로 바꾸면 그 하락이 상쇄된다.")
    else:
        pool = list(D["keywords"].columns)
        items = st.multiselect("키워드 (최대 3개)", pool, default=[k for k in ["롱패딩", "우산", "반바지"] if k in pool],
                               max_selections=3, key="dem_kw")
        if has_t1:
            view_k = st.radio("값", ["level (배치 정렬)", "패션의류 전체 대비 (%)"], horizontal=True, key="dem_view_kw")
        else:
            view_k = "level (배치 정렬)"
        if view_k.startswith("패션"):
            # 키워드 level ÷ 패션의류 전체 × 100: 플랫폼 전체가 내려가는 공통 하락이 분자·분모에 같이 들어 있어 상쇄된다
            total = D["t1_wide"]["패션의류_전체"].reindex(D["keywords"].index)
            data = D["keywords"].div(total, axis=0).mul(100).replace([np.inf, -np.inf], np.nan)
            ylab, kind = "% (÷ 패션의류 전체)", "kw"
            st.caption("공통 하락을 걷어낸 보조 보기입니다(노트북에서는 키워드에 쓰지 않았습니다). 키워드 level은 카테고리마다 다른 앵커로 맞춘 값이라 "
                       "이 비율의 크기 자체에는 의미가 없고, 같은 키워드의 시간 흐름과 연도 비교에만 쓰세요.")
        else:
            data, ylab, kind = D["keywords"], "level (배치 정렬)", "kw"
            st.caption("level로 보면 모든 키워드가 함께 내려갑니다(공통 하락). '패션의류 전체 대비'로 바꾸면 상쇄됩니다.")

    if items:
        # 성수기 기준 (03 날씨 × 수요와 같은 정의, 노트북에서 점검한 30~60%)
        pct = st.radio("성수기 기준 (피크월 대비 %)", [30, 40, 50, 60], index=1, horizontal=True,
                       format_func=lambda p: f"{p}%", key="pct_dem",
                       help="계절지수가 가장 높은 달의 이 비율 이상인 달을 성수기로 본다. 노트북 기본값은 40%, 점검 범위는 30~60%.")
        si_src = season_index(D["t1_share"], "t1") if src.startswith("T1") else season_index(D["keywords"], "kw")
        act_dem = active_months(si_src, pct)

        scale = st.toggle("항목별 평균 = 100 으로 맞춰 모양만 비교", value=True, key="dem_scale",
                          help="항목마다 크기가 수십 배 달라 작은 항목이 바닥에 깔린다. 켜면 각 항목을 자기 평균 대비 지수로 바꾼다.")
        sm = data[items].rolling(7, center=True).mean()
        if scale:
            sm = sm / data[items].mean() * 100
            ylab = "지수 (항목 평균 = 100)"
        fig = go.Figure()
        for i, k in enumerate(items):
            peak_mask = sm.index.month.isin(act_dem[k])
            # 성수기는 굵은 실선, 비수기는 가는 옅은 선 (같은 색)
            fig.add_trace(go.Scatter(x=sm.index, y=sm[k].where(peak_mask), name=f"{k} (성수기)", line=dict(color=SERIES[i], width=3)))
            fig.add_trace(go.Scatter(x=sm.index, y=sm[k].where(~peak_mask), name=f"{k} (비수기)", opacity=0.35,
                                     line=dict(color=SERIES[i], width=1.2), showlegend=False))
        fig = style(fig, 380, ylab)
        fig.update_layout(title=f"7일 이동평균 (굵은 선 = 성수기 {pct}% 기준, 옅은 선 = 비수기)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(" · ".join(f"{k}: {', '.join(f'{m}월' for m in act_dem[k])}" for k in items) + " 이 성수기입니다.")

        fig_b = go.Figure()
        for i, k in enumerate(items):
            rel = si_src[k] / si_src[k].max()
            fig_b.add_trace(go.Bar(x=[f"{m}월" for m in rel.index], y=rel.values, name=k, marker_color=SERIES[i],
                                   marker_opacity=[1.0 if m in act_dem[k] else 0.3 for m in rel.index],
                                   hovertemplate=f"{k} " + "%{x}<br>최고 달 대비 %{y:.2f}<extra></extra>"))
        fig_b.add_hline(y=pct / 100, line_dash="dot", line_color=BURGUNDY,
                        annotation_text=f"성수기 기준 {pct}%", annotation_font_color=MUTED)
        fig_b = style(fig_b, 260, "최고 달 = 1.0")
        fig_b.update_layout(title="월별 계절지수와 성수기 기준선 (진한 막대 = 성수기)", barmode="group", hovermode="closest")
        st.plotly_chart(fig_b, use_container_width=True)

        jj = data[items][data.index.month <= 7]
        yr = jj.groupby(jj.index.year).mean()
        st.markdown("**연도별 1~7월 평균 (첫해 = 1.0)** — 2026년은 7월까지만 있어 같은 달끼리 비교한다")
        st.dataframe((yr / yr.iloc[0]).round(2).T, use_container_width=True)

    st.subheader("월별 계절 패턴")
    # 위에서 고른 성수기 기준(pct_dem)을 그대로 따른다. 항목을 하나도 안 골랐으면 radio가 없으므로 기본값 40
    pct_h = st.session_state.get("pct_dem", 40)
    st.caption(f"각 항목의 최고 달 = 1.0 으로 맞춘 월별 계절지수. 성수기 기준 {pct_h}% 미만인 달은 회색(비수기)으로 비워 둔다. "
               "진한 빨강일수록 그 달에 뜬다. (한 줄 안에서만 비교)")
    base_h = D["t1_share"] if src.startswith("T1") else data
    mm = season_index(base_h, "t1" if src.startswith("T1") else "kw")
    z = (mm / mm.max()).T
    z = z.loc[z.idxmax(axis=1).sort_values(kind="stable").index]
    z_show = z.where(z >= pct_h / 100)            # 기준 미만인 달은 NaN → 빈칸(비수기)
    fig = go.Figure(go.Heatmap(z=z_show.values, x=[f"{m}월" for m in z.columns], y=z.index,
                               colorscale=[[0, "#f1b38f"], [1, RED]], zmin=pct_h / 100, zmax=1, hoverongaps=False,
                               hovertemplate="%{y} · %{x}<br>최고 달 대비 %{z:.2f}<extra></extra>"))
    fig = style(fig, max(380, 22 * len(z)), "")
    fig.update_layout(hovermode="closest", plot_bgcolor="#e6e4da")   # 빈칸 = 회색 = 비수기
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def page_ch03():
    chapter('CHAPTER 03', '날씨 × 수요', '평소보다 추우면, 평소보다 더 찾을까')
    # 이 화면에서만 쓰는 설정 (노트북에서 강건성을 확인한 30~60% 범위)
    s1, s2 = st.columns([2.2, 1])
    pct = s1.radio('성수기 기준 (피크월 대비 %)', [30, 40, 50, 60], index=1, horizontal=True,
                   format_func=lambda p: f'{p}%', key='pct_link',
                   help='계절지수가 가장 높은 달의 이 비율 이상인 달만 상관 계산에 쓴다. 노트북 기본값은 40%, 점검 범위는 30~60%.')
    dow = s2.checkbox('요일 보정', value=True, key='dow_link', help='수요에서 요일 평균을 한 번 더 뺀다 (노트북 T1-12c).')
    y_kw = demand_short_term(D['keywords'], True, dow)
    act_kw = active_months(si_kw, pct)
    if has_t1:
        y_cat = demand_short_term(D['t1_share'], False, dow)
        act_cat = active_months(si_cat, pct)

    c_a, c_b, c_c = st.columns([1.2, 1.5, 1.5])
    options = (["T1 카테고리(점유율)"] if has_t1 else []) + ["키워드(T3 level)"]
    src2 = c_a.radio("데이터", options, key="link_src")
    if src2.startswith("T1"):
        base, y_all, act, si_cur, floor_default = D["t1_share"], y_cat, act_cat, si_cat, "장갑"
        raw_log = np.log
    else:
        base, y_all, act, si_cur, floor_default = D["keywords"], y_kw, act_kw, si_kw, "롱패딩"
        raw_log = np.log1p
    names = list(base.columns)
    item = c_b.selectbox("항목", names, index=names.index(floor_default) if floor_default in names else 0)
    rainy = item in ("우산", "레인부츠", "레인코트")
    var = c_c.selectbox("날씨 변수", list(WEATHER_VARS), format_func=lambda v: WEATHER_VARS[v],
                        index=list(WEATHER_VARS).index("log_precip" if rainy else "t_min"))

    mask = pd.Series(y_all.index.month.isin(act[item]), index=y_all.index)
    st.caption(f"성수기: {', '.join(f'{m}월' for m in act[item])} ({int(mask.sum())}일) · 성수기 기준 {pct}% · "
               f"평소 수준: 31일 이동평균{' + 요일 보정' if dow else ''}")

    # 성수기 기준이 어떤 달을 고르는지 눈으로 확인 (슬라이더를 움직이면 파란 막대가 바뀐다)
    mm_item = si_cur[item]
    fig_s = go.Figure(go.Bar(x=[f"{m}월" for m in mm_item.index], y=mm_item.values,
                             marker_color=[BLUE if m in act[item] else "#c3c2b7" for m in mm_item.index],
                             hovertemplate="%{x}<br>지수 %{y:.2f}<extra></extra>"))
    fig_s.add_hline(y=mm_item.max() * pct / 100, line_dash="dot", line_color=ORANGE,
                    annotation_text=f"피크월의 {pct}%", annotation_font_color=MUTED)
    fig_s = style(fig_s, 220, "월별 계절지수")
    fig_s.update_layout(title=f"{item}: 성수기로 고른 달 (파랑)", hovermode="closest")
    st.plotly_chart(fig_s, use_container_width=True)

    raw_r = raw_log(base[item]).corr(wd[var])
    st_r = y_all[item][mask].corr(x_st[var][mask])
    k1, k2, k3 = st.columns(3)
    k1.metric("원 시계열 상관 (전 기간)", f"{raw_r:.2f}", help="계절이 섞여 부풀려진 값")
    k2.metric("단기 변동 상관 (성수기)", f"{st_r:.2f}", help="평소 수준을 뺀 뒤의 상관 = 날씨 효과")
    if var in ("t_min", "t_max"):
        d0 = pd.concat([y_all[item][mask], x_st[var][mask]], axis=1).dropna()
        if len(d0) > 2:
            slope = np.polyfit(d0.iloc[:, 1], d0.iloc[:, 0], 1)[0]
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
    winter_like = 12 in act[item] or 1 in act[item]          # 11~1월이 성수기면 겨울: 1월은 전해 겨울에 묶는다
    season_id = (y_all.index.year - (y_all.index.month < 7)).astype(str) if winter_like else y_all.index.year.astype(str)
    cons = pd.DataFrame({"y": y_all[item], "x": x_st[var], "g": season_id})[mask.values].dropna()
    tbl = cons.groupby("g")[["y", "x"]].apply(lambda g: pd.Series({"일수": len(g), "상관": g["y"].corr(g["x"])})).round(2)
    tbl.index.name = "시즌(시작 연도)"
    st.dataframe(tbl.T, use_container_width=True)
    st.caption("한 시즌이 결과 전체를 끌고 가는지 확인한다. 모든 시즌에서 부호가 같아야 믿을 만하다. 일수가 적은 시즌(데이터가 일부만 있는 해)은 크게 믿지 않는다.")

    st.subheader("날씨 변수별 수요와의 상관 비교")
    scope = st.radio("범위", ["선택한 항목만", "전체 항목"], horizontal=True, key="screen_scope",
                     help=f"선택한 항목: {item}. 위에서 항목을 바꾸면 표도 따라 바뀐다.")
    keys = [item] if scope.startswith("선택") else list(base.columns)
    rows = {}
    for k in keys:
        m_ = y_all.index.month.isin(act[k])
        r = {v: y_all[k][m_].corr(x_st[v][m_]) for v in WEATHER_VARS}
        r["최강 변수"] = WEATHER_VARS[max(r, key=lambda v: abs(r[v]) if pd.notna(r[v]) else -1)]
        r["성수기 일수"] = int(m_.sum())
        rows[k] = r
    screen = pd.DataFrame(rows).T.rename(columns=WEATHER_VARS)
    num_cols = list(WEATHER_VARS.values())
    screen[num_cols] = screen[num_cols].astype(float)
    screen = screen.reindex(screen[num_cols].abs().max(axis=1).sort_values(ascending=False).index)
    st.dataframe(screen.style.format("{:.2f}", subset=num_cols).map(diverging_cell, subset=num_cols),
                 use_container_width=True, height=420 if len(keys) > 8 else "content")
    avg = screen[num_cols].abs().mean().sort_values(ascending=False).round(3)
    if len(keys) > 1:
        st.markdown("**변수별 평균 |r|**: " + " · ".join(f"{k} {v}" for k, v in avg.items()))
    st.caption("핵심 가설은 '절대 기온보다 Δt·일교차'였지만, 데이터는 평소 대비 기온 편차가 가장 강하고 Δt는 거의 0이다. "
               "우산류의 일교차 상관은 '비 오는 날은 일교차가 작다'는 대리 효과로 본다. "
               "|r| 0.2 미만은 여러 조합을 훑은 결과라 우연일 수 있다.")


def page_ch04():
    chapter('CHAPTER 04', '한파 이벤트', '한파의 기준, 그날 옷장의 변화')
    dow = st.checkbox('요일 보정', value=True, key='dow_cold', help='수요에서 요일 평균을 한 번 더 뺀다 (노트북 T1-12c).')
    y_kw = demand_short_term(D['keywords'], True, dow)
    if has_t1:
        y_cat = demand_short_term(D['t1_share'], False, dow)

    e1, e2, e3 = st.columns(3)
    thr = e1.slider("한파 임계값 (최저기온 ≤ ℃)", -14, -4, -10, key="cold_thr")
    min_len = e2.slider("지속 일수 (이상)", 1, 4, 2, key="cold_len")
    tol = e3.slider("특보와 맞추는 허용 오차 (±일)", 0, 4, 2, key="cold_tol")
    st.subheader(f"한파 D0 = 최저기온 ≤ {thr}℃ 가 {min_len}일 이상 이어지는 첫날")

    cold_d0 = rule_first_days(wd["t_min"] <= thr, min_len)
    has_warn = warn is not None and "cold_warn" in warn.columns
    if has_warn:
        wflag = warn["cold_warn"].fillna(0).astype(bool).reindex(wd.index, fill_value=False)
        warn_eps = rule_first_days(wflag, 1)                       # 실제 특보 에피소드 첫날
    ev = pd.DataFrame({
        "D0": cold_d0.date,
        "D0 최저기온(℃)": wd.loc[cold_d0, "t_min"].values,
        "-12℃ 이하 동반": [bool((wd["t_min"].loc[d:d + pd.Timedelta(days=3)] <= -12).any()) for d in cold_d0],
    })
    if has_warn:
        ev[f"특보 발효(±{tol}일)"] = [bool(wflag.loc[d - pd.Timedelta(days=tol): d + pd.Timedelta(days=tol)].any()) for d in cold_d0]

    def prec_rec(days):
        if not has_warn or len(days) == 0:
            return np.nan, np.nan
        p = np.mean([wflag.loc[d - pd.Timedelta(days=tol): d + pd.Timedelta(days=tol)].any() for d in days])
        r = np.mean([any(abs((d - w).days) <= tol for d in days) for w in warn_eps])
        return round(float(p), 2), round(float(r), 2)

    col_l, col_r = st.columns([1, 1.4])
    col_l.dataframe(ev, use_container_width=True, hide_index=True)
    rows = []
    for t in (-12, -10, -8, -6):
        days = rule_first_days(wd["t_min"] <= t, min_len)
        p, r = prec_rec(days)
        rows.append({"기준": f"{t}℃", "이벤트 수": len(days), "정밀도": p, "재현율": r})
    col_r.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    col_r.caption("정밀도 = 규칙 이벤트 중 실제 특보가 있던 비율, 재현율 = 실제 특보 중 규칙이 잡아낸 비율. "
                  "-12℃ 기준은 3.6개 겨울에 4건이라 표본이 부족해 노트북에서는 -10℃를 골랐다. "
                  "특보는 서울 4개 구역 중 한 곳만 해당돼도 발표돼서 규칙 이벤트보다 자주 나온다.")
    if not has_warn:
        st.warning("`warning_daily.csv` 에 `cold_warn` 컬럼이 없어 특보 대조는 생략했다.")

    st.subheader("한파 전후 3주의 수요 변화")
    st.caption("한파 시작일(D0) 기준 7일 전부터 14일 후까지")
    src3 = st.radio("데이터", ["키워드(T3 level)"] + (["T1 카테고리(점유율)"] if has_t1 else []), horizontal=True, key="cold_src")
    yb = y_kw if src3.startswith("키워드") else y_cat
    names3 = list(yb.columns)
    default = "롱패딩" if src3.startswith("키워드") else "장갑"
    item3 = st.selectbox("항목", names3, index=names3.index(default) if default in names3 else 0, key="cold_item")
    if len(cold_d0) == 0:
        st.info("이 기준에서는 한파 이벤트가 없습니다. 임계값을 올려 보세요.")
    else:
        offsets = list(range(-7, 15))
        mat = pd.DataFrame({d.date(): [yb[item3].get(d + pd.Timedelta(days=o), np.nan) for o in offsets] for d in cold_d0},
                           index=offsets)
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
                   "'평소보다 몇 % 더 찾았나'로 읽는다. 사례가 적어서 개별 선이 평균을 좌우하지 않는지 함께 본다.")
    nextup('05 · 결론')


def page_ch05():
    chapter('CHAPTER 05', '결론', '서울의 하루하루 날씨는 옷장을 움직였다. 다만 모든 옷을 움직인 것은 아니었다')
    st.caption('이 장의 숫자는 노트북 기본 설정(성수기 40%, 요일 보정 켬) 기준')
    pull('평소보다 추운 날에는 방한 용품이, 비 오는 날에는 우산이 더 찾아졌습니다. 하지만 코트나 니트처럼 날씨를 따르지 않는 옷도 있었습니다.')

    st.markdown('## 1 · 평소보다 추운 날, 방한 용품의 검색 증가')
    st.markdown('계절을 걷어내고도 롱패딩, 패딩, 어그, 장갑, 목도리의 클릭은 최저기온과 뚜렷하게 반대로 움직였습니다.')
    table(pd.DataFrame({
        '키워드': ['롱패딩', '패딩', '어그', '장갑', '목도리'],
        '최저기온과의 상관': [-0.85, -0.74, -0.70, -0.68, -0.65],
        '평소보다 1℃ 추울 때': ['약 +9.0%', '약 +4.6%', '약 +4.0%', '약 +7.7%', '약 +4.8%']}))
    st.markdown('장갑은 겨울별로 나눠 봐도 모두 같은 방향이었고(-0.53 ~ -0.88), 추워진 당일에 가장 크게 반응해 다음 날까지 일부 이어졌습니다. '
                '우산은 비 온 당일에만 반응했습니다(+0.63 ~ +0.68).')

    st.markdown('## 2 · 겨울옷 안에서도 갈리는 날씨 민감도')
    st.markdown('옷이 팔리는 이유는 크게 두 가지로 나눌 수 있습니다.\n\n'
                '- **계절(달력) 요인**: "지금이 어느 시기인가"가 정하는 수요입니다. 11월이니 패딩, 여름이니 수영복.\n'
                '- **날씨 요인**: "오늘 날씨가 평소와 얼마나 다른가"가 정하는 수요입니다.')
    table(pd.DataFrame({
        '': ['무엇이 정하나', '움직이는 속도', '이 리포트의 예', '확인한 방법', '활용 방향'],
        '계절(달력) 요인': ['시기 (몇 월인가)', '몇 주~몇 달에 걸쳐 천천히', '수영, 코트, 가디건', '계절을 빼면 상관이 크게 줄어듦', '시즌 일정에 맞춰 준비'],
        '날씨 요인': ['그날 날씨가 평소와 얼마나 다른가', '하루~이틀 사이에 갑자기', '롱패딩, 장갑, 우산', '계절을 빼도 상관이 남음', '예보에 맞춰 노출·재고 조정']}))
    st.markdown('롱패딩은 날씨에 민감했고(-0.85), 코트는 거의 반응하지 않았습니다(-0.09). 니트, 가디건, 맨투맨도 마찬가지였습니다.')
    pull('패딩은 "오늘 추워서" 사고, 코트는 "겨울이 왔으니" 산다고 볼 수 있습니다. (데이터로 확인한 것이 아니라 모양에서 나온 추측입니다.)')

    st.markdown('## 3 · 틀렸던 가설')
    st.markdown('처음에는 *소비자는 절대 기온보다 체감온도, 일교차, 전일 대비 변화에 더 반응한다*고 생각했습니다.')
    table(pd.DataFrame({
        '가설': ['계절을 빼야 진짜 효과가 보인다', '방한·여름 상품은 기온에 반응한다', '전일 대비 기온 변화에 반응한다',
               '체감온도가 기온보다 더 잘 설명한다', '일교차에 반응한다', '검색이 클릭보다 먼저 움직인다', '내일 예보로 추천할 수 있다'],
        '결과': ['✅', '✅', '❌', '❌', '△', '❌', '❌'],
        '근거': ['장갑 -0.71 → -0.46, 수영 0.63 → 0.15', '롱패딩 -0.85, 반바지 +0.48', '평균 상관 0.03 (3일 누적도 0.10)',
               '5개 키워드 모두 평균기온이 조금 우세', '우산류에서만 강하고 비의 대리 변수로 보임',
               '12개 키워드 모두 같은 날 가장 강함', '가장 가까운 예보가 3~4일 앞이고 오차가 효과와 비슷함']}))
    pull('사람들은 변화의 크기가 아니라 "평소보다 춥다는 사실"에 반응한 것으로 보입니다. ')

    st.markdown('## 4 · 이 분석의 한계')
    st.markdown('- 겨울이 3번뿐이라 정확한 크기(9%, 7.7%)는 범위로 읽어야 합니다. 상관은 인과를 증명하지 않습니다.\n'
                '- 서울 날씨와 전국 쇼핑 데이터를 맞춰 본 결과입니다.\n'
                '- 공휴일과 할인 행사 같은 달력 요인은 보정하지 못했습니다.\n'
                '- 예보 오차(1.1~1.8℃)가 방한 용품의 반응(1℃당 약 3~9%)과 비슷한 크기여서 "내일 추천"은 다루지 않았습니다.\n'
                '- 31일 평균과 성수기 40%는 제가 정한 값입니다. 30~60% 범위에서 방향이 유지되는지 확인했습니다.')

PAGES = {
    'COVER': page_cover,
    '01 · 날씨': page_ch01,
    '02 · 수요 추이': page_ch02,
    '03 · 날씨 × 수요': page_ch03,
    '04 · 한파 이벤트': page_ch04,
    '05 · 결론': page_ch05,
}
PAGES[menu]()