"""W1. 지상관측(ASOS) 일자료 수집기 — 서울 지점 108.

    GET kma_sfcdd3.php?tm1=20230101&tm2=20230131&stn=108&disp=1&help=1&authKey=...

이 데이터가 `weather_daily` 의 실체다. 예보(W2·W3)가 아니라 **관측(W1)이 EDA와
이벤트 정의의 기준**이므로, 여기서는 원본을 최대한 손대지 않고 그대로 남긴다.
일교차·Δt·체감온도 같은 파생은 전부 src/transform 단계에서 만든다.

왜 월 단위로 쪼개 호출하나
    kma_sfcdd3.php 는 **기간 조회 최대 31일** 제약이 있다. 월 단위로 자르면
    (1) 31일을 넘길 일이 없고 (2) 실패한 달만 골라 재실행할 수 있으며
    (3) 저장 파티션(`{yyyy-mm}.csv`)과 1:1로 대응돼 재개 지점이 명확하다.

사용법
    python -m src.collect.weather_asos                                # 어제 1일 (증분)
    python -m src.collect.weather_asos --start 2023-01-01 --end 2026-07-31   # 백필
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta

import pandas as pd

from ._common import (
    DATA_RAW,
    KMA_BASE,
    SEOUL,
    ApiError,
    get,
    kma_key,
    kma_rows,
    month_range,
    write_csv,
    raw_path,
    read_raw,
    as_text_frame,
)

SOURCE = "kma_asos"
ENDPOINT = f"{KMA_BASE}/kma_sfcdd3.php"

# 응답 컬럼 순서 (fallback).
# 출처: https://apihub.kma.go.kr/apiList.do?apiSeq=9 의 "일자료(기간 조회)" 항목.
# 실제 파싱은 응답 안의 `#` 헤더 줄을 우선 사용하고, 헤더를 못 찾을 때만 이 목록을 쓴다.
# TODO: 활용신청 후 실제 응답으로 컬럼 검증 필요.
#       확인 방법 — help=1 로 한 번 호출해 `#` 주석 헤더 줄을 아래 목록과 대조.
#       `curl "…/kma_sfcdd3.php?tm1=20230101&tm2=20230103&stn=108&disp=1&help=1&authKey=$KEY"`
ASOS_COLUMNS: list[str] = [
    "TM",            # 관측일 (KST)
    "STN",           # 국내 지점번호
    "WS_AVG",        # 일 평균 풍속 (m/s)
    "WR_DAY",        # 일 풍정 (m)
    "WD_MAX",        # 최대풍향
    "WS_MAX",        # 최대풍속 (m/s)
    "WS_MAX_TM",     # 최대풍속 시각
    "WD_INS",        # 최대순간풍향
    "WS_INS",        # 최대순간풍속 (m/s)
    "WS_INS_TM",     # 최대순간풍속 시각
    "TA_AVG",        # 일 평균기온 (℃)
    "TA_MAX",        # 일 최고기온 (℃)
    "TA_MAX_TM",     # 최고기온 시각
    "TA_MIN",        # 일 최저기온 (℃)
    "TA_MIN_TM",     # 최저기온 시각
    "TD_AVG",        # 일 평균 이슬점온도 (℃)
    "TS_AVG",        # 일 평균 지면온도 (℃)
    "TG_MIN",        # 일 최저 초상온도 (℃)
    "HM_AVG",        # 일 평균 상대습도 (%)
    "HM_MIN",        # 최소 상대습도 (%)
    "HM_MIN_TM",     # 최소습도 시각
    "PV_AVG",        # 일 평균 증기압 (hPa)
    "EV_S",          # 소형증발량 (mm)
    "EV_L",          # 대형증발량 (mm)
    "FG_DUR",        # 안개계속시간 (hr)
    "PA_AVG",        # 일 평균 현지기압 (hPa)
    "PS_AVG",        # 일 평균 해면기압 (hPa)
    "PS_MAX",        # 최고 해면기압 (hPa)
    "PS_MAX_TM",     # 최고 해면기압 시각
    "PS_MIN",        # 최저 해면기압 (hPa)
    "PS_MIN_TM",     # 최저 해면기압 시각
    "CA_TOT",        # 일 평균 전운량 (1/10)
    "SS_DAY",        # 일조시간 (hr)
    "SS_DUR",        # 가조시간 (hr)
    "SS_CMB",        # 캠벨 일조 (hr)
    "SI_DAY",        # 일사량 (MJ/m2)
    "SI_60M_MAX",    # 최대 1시간 일사량 (MJ/m2)
    "SI_60M_MAX_TM",
    "RN_DAY",        # 일 강수량 (mm)
    "RN_D99",        # 9-9 강수량 (mm)
    "RN_DUR",        # 강수계속시간 (hr)
    "RN_60M_MAX",    # 1시간 최다강수량 (mm)
    "RN_60M_MAX_TM",
    "RN_10M_MAX",    # 10분 최다강수량 (mm)
    "RN_10M_MAX_TM",
    "RN_POW_MAX",    # 최대 강우강도 (mm/h)
    "RN_POW_MAX_TM",
    "SD_NEW",        # 일 최심 신적설 (cm)
    "SD_NEW_TM",
    "SD_MAX",        # 일 최심 적설 (cm)
    "SD_MAX_TM",
    "TE_05",         # 0.5m 지중온도 (℃)
    "TE_10",         # 1.0m 지중온도 (℃)
    "TE_15",         # 1.5m 지중온도 (℃)
    "TE_30",         # 3.0m 지중온도 (℃)
    "TE_50",         # 5.0m 지중온도 (℃)
]

# docs/datasets.md §7 weather_daily 로 올려보낼 컬럼 매핑 (단순 이름 변경일 뿐 정규화가 아니다)
#
# ★ precip_mm(RN_DAY) 해석 주의 — transform 단계에서 반드시 처리할 것 ★
#   ASOS 는 **무강수일의 RN_DAY 를 -9.0(결측)으로 표기**한다. 실측 3년치에서
#   1,308일 중 754일(57.6%)이 여기 해당하고, 실제로 비가 온 날은 554일이다.
#   즉 이 컬럼의 NaN 은 "측정 실패"가 아니라 대부분 **"비가 안 옴 = 0.0mm"** 이다.
#   raw 에서는 원문 그대로 NaN 으로 두고, processed 단계에서 fillna(0.0) 한다.
#   (여기서 0으로 채우면 "결측"과 "무강수"를 영영 구분할 수 없게 되므로 raw 에서는 안 한다)
SCHEMA_MAP = {
    "t_max": "TA_MAX",
    "t_min": "TA_MIN",
    "t_avg": "TA_AVG",
    "precip_mm": "RN_DAY",      # #39 일 강수량(mm)
    "humidity": "HM_AVG",
    "wind": "WS_AVG",
}

# ── 결측값 처리 — 이 모듈에서 가장 조심해야 할 부분 ──────────────────────
#
# ASOS 응답은 결측을 `-9`, `-9.0`, `-9.00`, `-99.0` 으로 표기한다. 실측 예:
#     20230101 108  2.7 … 1.6 -9.00 1019.8 … 9.6 -9.0 …  -9.0  -9.0 -9.00 …
#
# 그런데 **-9.0 을 무조건 결측으로 바꾸면 실제 관측값이 파괴된다.**
#     20230103 108 … TA_AVG=-5.0  TA_MAX=0.6  TA_MIN=-9.0  ← 진짜 -9.0℃ 다
# 서울의 1월 최저기온 -9.0℃ 는 흔한 값이다. 반대로 강수량·습도·풍속·적설은
# 물리적으로 음수가 될 수 없으므로 -9.0 이 나오면 100% 결측이다.
#
# 그래서 **컬럼의 물리적 정의역에 따라 sentinel 집합을 다르게** 적용한다.
#   · 음수가 불가능한 컬럼(강수/습도/풍속/적설/시간/일사 등) → -9 계열도 결측
#   · 음수가 가능한 컬럼(기온류 TA_/TD_/TS_/TG_/TE_)        → -99 계열만 결측
#
# 절충의 대가: 기온이 진짜로 결측이라 -9.0 으로 온 경우를 놓친다. 지점 108(서울)은
# 결측이 거의 없는 기간관측 지점이라 이쪽 위험이 훨씬 작다고 판단했다.
MISSING_SIGNED = {"", "-", "-99", "-99.0", "-99.00", "-999", "-999.0",
                  "-9999", "-9999.0"}                       # 기온류: -9 는 유효값
MISSING_UNSIGNED = MISSING_SIGNED | {"-9", "-9.0", "-9.00"}  # 그 외: -9 도 결측

# 음수가 물리적으로 가능한(=기온) 컬럼 접두사
SIGNED_PREFIXES = ("TA_", "TD_", "TS_", "TG_", "TE_")

PERMISSION_GUIDE = """
────────────────────────────────────────────────────────────────
  기상청 API 허브 권한 오류 (403) — 인증키는 맞지만 API 사용 권한이 없습니다.
────────────────────────────────────────────────────────────────
  해결: https://apihub.kma.go.kr 로그인 →
        [지상관측] > [종관기상관측(ASOS)] > "일자료(기간 조회)" (kma_sfcdd3.php)
        해당 API 상세 페이지에서 **활용신청** 버튼을 누르세요.
  * 기상청 API 허브는 API 하나하나마다 활용신청이 따로 필요합니다.
  * 승인 반영까지 수 분 걸릴 수 있습니다. 이후 같은 명령을 다시 실행하세요.
────────────────────────────────────────────────────────────────"""


# ── 파싱 ──────────────────────────────────────────────────
def _redact(msg: object) -> str:
    """오류 메시지에서 authKey 를 가린다.

    requests 의 연결 오류 메시지에는 **요청 URL 이 통째로** 들어간다.
    그대로 출력하면 authKey 가 로그·터미널에 그대로 남으므로 반드시 지운다.
    """
    return re.sub(r"(authKey=)[^&\s'\"]+", r"\1***", str(msg))


def _header_candidates(text: str, must_have: tuple[str, ...]) -> list[list[str]]:
    """`help=1` 응답에서 컬럼명 후보를 **우선순위 순서로** 뽑는다.

    기상청 허브 help 블록은 컬럼 정보를 두 군데에 준다. 실제 응답으로 확인한 결과
    **둘이 서로 다르다**. fct_afs_wl.php 예:

        #  1. REG_ID   : 예보구역코드          ← ① 번호 매긴 "항목 카탈로그" (14개)
        ...
        #  7. MAN_ID   : 예보관ID                 ↑ mode 옵션에 따라 출력되지 않는 항목까지 포함
        #  8. MAN_FC   : 예보관명
        #  9. REG_NAME : 예보구역명
        ...
        # REG_ID TM_FC  TM_EF  MOD STN C SKY PRE CONF WF RN_ST   ← ② 실제 출력 헤더 (11개)
        11B00000,202301010600,202301040000,A02,109,2,WB01,WB00,없음,맑음,0,=

    ①의 MAN_ID/MAN_FC/REG_NAME 은 **실제 데이터 행에 없다.** ① 순서로 파싱하면
    SKY 자리에 MAN_ID 가 들어가는 식으로 컬럼이 통째로 밀린다.
    따라서 **②(실제 출력 헤더)를 최우선**으로 쓰고, ①은 ②가 없을 때만 쓴다.
    최종 선택은 to_frame() 이 실제 필드 수와 대조해 결정한다.
    """
    out: list[list[str]] = []

    # ② 실제 출력 헤더 행. 데이터 바로 위에 오므로 뒤에서부터 찾는다.
    #    고정폭 정렬용 대시가 컬럼명에 붙어 있다: `REG_KO----------------`
    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s.startswith("#"):
            continue
        toks = [t.strip("-").strip() for t in re.split(r"[,\s]+", s.lstrip("#").strip())]
        # 헤더 행 끝에도 종료 마커 "=" 가 붙어 나온다(wrn_met_data.php). 컬럼명이 아니므로 제외.
        toks = [t for t in toks if t and t != "="]
        if len(toks) < 5:
            continue
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t) for t in toks):
            continue
        up = [t.upper() for t in toks]
        if all(m in up for m in must_have):
            out.append(up)
            break

    # ① 번호 매긴 항목 카탈로그 (출력되지 않는 항목이 섞일 수 있음)
    numbered: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*#\s*(\d+)\s*\.\s*([A-Za-z][A-Za-z0-9_]*)\s*[:：]", line)
        if m:
            numbered.setdefault(int(m.group(1)), m.group(2).upper())
    if numbered and set(numbered) == set(range(1, len(numbered) + 1)):
        cols = [numbered[i] for i in range(1, len(numbered) + 1)]
        if all(m in cols for m in must_have):
            out.append(cols)

    return out


def _uniquify(names: list[str]) -> list[str]:
    """같은 컬럼명이 두 번 나오면 뒤엣것에 _2, _3 을 붙인다.

    이름이 겹치면 df["X"] 가 Series 가 아니라 DataFrame 을 돌려줘 조용히 깨진다.
    """
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def to_frame(text: str, fallback: list[str], must_have: tuple[str, ...]) -> pd.DataFrame:
    """기상청 허브 텍스트 응답 → DataFrame. 모든 값은 원문 문자열 그대로 둔다."""
    raw_lines = kma_rows(text)
    if not raw_lines:
        return pd.DataFrame(columns=fallback)

    # 구분자는 **응답을 보고 판정**한다. 문서도 disp 파라미터도 믿을 수 없다:
    #   · kma_sfcdd3.php 는 disp=1 을 줘도 무시하고 항상 공백 구분(고정폭)으로 응답한다
    #   · fct_afs_* / wrn_met_data 는 disp=1 이면 쉼표 구분
    # 데이터 행에 쉼표가 있으면 쉼표, 없으면 공백으로 자른다.
    comma = any("," in ln for ln in raw_lines)
    rows = [ln.split(",") if comma else ln.split() for ln in raw_lines]
    rows = [[c.strip() for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows:
        return pd.DataFrame(columns=fallback)

    # 기상청 허브는 각 데이터 행 끝에 종료 마커 "=" 를 붙인다.
    #   11B00000,202301010600,…,맑음,0,=
    # 이걸 값으로 읽으면 마지막 컬럼이 통째로 밀려 오염되므로 먼저 떼어낸다.
    # 모든 행이 "=" 로 끝날 때만 제거해서, 실제 값이 잘려나가는 일이 없게 한다.
    while rows and all(len(r) > 1 and r[-1] == "=" for r in rows):
        rows = [r[:-1] for r in rows]

    width = max(len(r) for r in rows)

    # 후보(② 실제헤더 → ① 카탈로그 → 문서 목록) 중 **실제 필드 수와 일치하는 첫 번째**를 쓴다.
    # 길이 대조가 컬럼 밀림을 막는 마지막 안전장치다.
    candidates = _header_candidates(text, must_have) + [list(fallback)]
    header = next((c for c in candidates if len(c) == width), None)
    if header is None:
        header = candidates[0]
        print(f"    ! 컬럼 수 불일치: 후보 {[len(c) for c in candidates]} vs 데이터 {width}개"
              f" → {header[:3]}… 로 진행하고 남는 자리는 COL_n 으로 채움")
    if len(header) < width:
        header += [f"COL_{i}" for i in range(len(header), width)]
    header = _uniquify(header[:width])

    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=header, dtype="string")


def _num(s: pd.Series, name: str | None = None) -> pd.Series:
    """숫자 컬럼으로 변환. 컬럼 이름에 따라 결측 sentinel 집합을 고른다.

    name 을 주지 않으면 안전한 쪽(기온류 규칙 = -9 를 값으로 유지)을 쓴다.
    """
    col = name or str(s.name or "")
    sentinels = (MISSING_SIGNED if col.startswith(SIGNED_PREFIXES) else MISSING_UNSIGNED)
    cleaned = s.astype("string").str.strip().replace(dict.fromkeys(sentinels, pd.NA))
    return pd.to_numeric(cleaned, errors="coerce")


def _tm_to_date(s: pd.Series) -> pd.Series:
    digits = s.astype("string").str.replace(r"\D", "", regex=True).str[:8]
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")


# ── 수집 ──────────────────────────────────────────────────
def fetch_month(tm1: str, tm2: str, stn: str, key: str) -> pd.DataFrame:
    """[tm1, tm2] (YYYYMMDD, 최대 31일) 구간의 일자료를 가져온다."""
    text = get(ENDPOINT, {
        "tm1": tm1, "tm2": tm2, "stn": stn,
        # disp=1(쉼표)을 줘도 이 API는 무시하고 공백 구분으로 응답한다.
        # 그래서 to_frame() 이 응답을 보고 구분자를 판정한다.
        "disp": "1",
        "help": "1",     # 컬럼 정의를 같이 받아 문서 대신 응답으로 컬럼을 확정
        "authKey": key,
    })
    df = to_frame(text, ASOS_COLUMNS, ("TM", "STN"))
    if df.empty:
        return df

    # docs/datasets.md §7 weather_daily 컬럼을 앞쪽에 붙인다. 이름 변경 + 숫자 캐스팅뿐이고
    # 조인·정규화는 하지 않는다 (raw 규약).
    out = pd.DataFrame(index=df.index)
    out["date"] = _tm_to_date(df["TM"])
    out["region"] = "서울"
    out["stn"] = df["STN"]
    for dst, src in SCHEMA_MAP.items():
        out[dst] = _num(df[src], src) if src in df.columns else pd.NA
    # weather_code(맑음/흐림/비/눈)는 CA_TOT·RN_DAY·SD_NEW 로부터 **파생**되는 값이라
    # raw 단계에서 만들지 않는다(= src/transform 의 몫). 스키마 자리만 비워 둔다.
    out["weather_code"] = pd.NA

    dup = [c for c in df.columns if c in out.columns]
    return pd.concat([out, df.drop(columns=dup)], axis=1)


def _merge_partition(df: pd.DataFrame, partition: str, keys: list[str]) -> pd.DataFrame:
    """기존 파티션과 합쳐 멱등성을 보장한다.

    부분 기간(예: 1/1~1/5)만 다시 돌려도 이미 받아둔 나머지 날짜가 날아가지 않고,
    같은 날짜를 두 번 받으면 새 응답이 이긴다(keep="last").
    """
    path = raw_path(SOURCE, partition)
    if path.exists():
        # CSV 는 타입을 보존하지 않는다. 양쪽을 문자열 표현으로 맞춰야
        # 키 비교가 어긋나지 않는다 (저장된 "108" vs 새 108).
        df = pd.concat([read_raw(path), as_text_frame(df)], ignore_index=True)
    return (df.drop_duplicates(subset=keys, keep="last")
              .sort_values(keys)
              .reset_index(drop=True))


def fetch_daily_fallback(tm1: str, tm2: str, stn: str, key: str) -> pd.DataFrame:
    """대체 경로: `kma_sfcdd.php` 로 **하루씩** 조회해 이어붙인다.

    `kma_sfcdd3.php`(기간 조회) 활용신청이 아직 승인되지 않은 계정을 위한 우회로다.
    `kma_sfcdd.php` 는 기간 파라미터를 무시하고 **tm 하루치 1행**만 돌려주므로
    3년치면 약 1,300회 호출이 된다. 그래서 기본 비활성이고 `--fallback-daily` 로만 켠다.

    TODO: kma_sfcdd.php 의 컬럼 구성이 kma_sfcdd3.php 와 동일한지는 미검증이다.
          (to_frame 이 응답 헤더로 컬럼을 잡으므로 달라도 밀리지는 않는다)
          이 경로를 실제로 쓰게 되면 하루치를 kma_sfcdd3 결과와 대조해 볼 것.
    """
    days = pd.date_range(tm1, tm2, freq="D")
    frames = []
    for d in days:
        text = get(f"{KMA_BASE}/kma_sfcdd.php",
                   {"tm": d.strftime("%Y%m%d"), "stn": stn,
                    "disp": "1", "help": "1", "authKey": key})
        f = to_frame(text, ASOS_COLUMNS, ("TM", "STN"))
        if not f.empty:
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ASOS_COLUMNS)


def _to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """원본 프레임 앞에 docs/datasets.md §7 weather_daily 컬럼을 붙인다."""
    out = pd.DataFrame(index=df.index)
    out["date"] = _tm_to_date(df["TM"])
    out["region"] = "서울"
    out["stn"] = df["STN"]
    for dst, src in SCHEMA_MAP.items():
        out[dst] = _num(df[src], src) if src in df.columns else pd.NA
    out["weather_code"] = pd.NA
    dup = [c for c in df.columns if c in out.columns]
    return pd.concat([out, df.drop(columns=dup)], axis=1)


def missing_report(df: pd.DataFrame) -> dict[str, int]:
    """원본 컬럼별 결측 건수. 백필 후 '무엇이 얼마나 비었나'를 남기기 위한 것."""
    skip = {"date", "region", "stn", "weather_code", "TM", "STN"}
    rep: dict[str, int] = {}
    for c in df.columns:
        if c in skip or c.endswith("_TM"):
            continue
        n = int(_num(df[c], c).isna().sum())
        if n:
            rep[c] = n
    return dict(sorted(rep.items(), key=lambda kv: -kv[1]))


def backfill(start: str, end: str, stn: str = SEOUL["asos_stn"],
             *, fallback_daily: bool = False) -> list[str]:
    """월 단위 백필. 실패한 달 목록을 반환한다(전체 중단하지 않음)."""
    key = kma_key()
    failed: list[str] = []

    for s, e in month_range(start, end):
        partition = s.strftime("%Y-%m")
        label = f"[{partition}] {s:%Y-%m-%d} ~ {e:%Y-%m-%d}"
        try:
            if fallback_daily:
                df = _to_schema(fetch_daily_fallback(s, e, stn, key))
            else:
                df = fetch_month(s.strftime("%Y%m%d"), e.strftime("%Y%m%d"), stn, key)
            if df.empty:
                print(f"{label} … 데이터 0건 (건너뜀)")
                continue
            df = _merge_partition(df, partition, ["date", "stn"])
            path = write_csv(df, SOURCE, partition)
            print(f"{label} … {len(df):>3}행 → {path.name}")
        except ApiError as exc:
            if "403" in str(exc) or "활용신청" in str(exc):
                raise                       # 권한 문제는 모든 달에서 똑같이 난다 → 즉시 중단
            print(f"{label} … 실패: {_redact(exc)[:160]}")
            failed.append(partition)
        except Exception as exc:            # noqa: BLE001 - 한 달 실패로 백필 전체를 죽이지 않는다
            print(f"{label} … 실패: {type(exc).__name__}: {_redact(exc)[:160]}")
            failed.append(partition)

    return failed


# ── CLI ───────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="기상청 ASOS 일자료 수집 (서울 지점 108)")
    p.add_argument("--start", help="시작일 YYYY-MM-DD (기본: 어제)")
    p.add_argument("--end", help="종료일 YYYY-MM-DD (기본: 어제)")
    p.add_argument("--stn", default=SEOUL["asos_stn"], help=f"지점번호 (기본 {SEOUL['asos_stn']})")
    p.add_argument("--fallback-daily", action="store_true",
                   help="kma_sfcdd3.php(기간조회) 미승인 시 kma_sfcdd.php 로 하루씩 조회 "
                        "(3년이면 ~1,300회 호출 — 정말 필요할 때만)")
    a = p.parse_args(argv)

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    start = a.start or yesterday
    end = a.end or (a.start or yesterday)      # --start 만 주면 그날 하루

    print(f"ASOS 일자료 수집: {start} ~ {end} (지점 {a.stn})"
          + (" [일 단위 대체 경로]" if a.fallback_daily else ""))
    try:
        failed = backfill(start, end, a.stn, fallback_daily=a.fallback_daily)
    except ApiError as exc:
        if "403" in str(exc) or "활용신청" in str(exc):
            print(PERMISSION_GUIDE, file=sys.stderr)
            return 2
        print(f"\n[중단] API 오류: {_redact(exc)[:300]}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"\n[중단] {exc}", file=sys.stderr)
        return 1

    if failed:
        print(f"\n실패한 월 {len(failed)}개: {', '.join(failed)}")
        print("→ 해당 월만 --start/--end 로 다시 실행하세요.")
        return 1
    print("\n완료. 실패한 월 없음.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
