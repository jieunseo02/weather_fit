"""W4. 기상특보 발표 이력 수집기 — 서울.

    GET wrn_reg.php?disp=1&help=1&authKey=…                                   # 특보구역 코드
    GET wrn_met_data.php?tmfc1=202301010000&tmfc2=202312312359&reg=…&disp=1   # 특보 이력

왜 필요한가
    PRD의 룰 기반 이벤트 정의(한파 `t_min ≤ -12℃` 2일 지속 등)가 **얼마나 맞는지**
    대조할 정답지다. "우리 룰이 실제 특보와 몇 % 일치하는가"는 원안에 없던 분석이고,
    리포트에서 가장 설득력 있는 장면이 된다.

서울 특보구역 코드는 하드코딩하지 않는다
    `wrn_reg.php` 로 구역 목록을 받아 이름으로 서울을 찾는다(`resolve_seoul_regs`).
    실제로 조회해 보면 하드코딩이 왜 위험한지 바로 드러난다 — '서울'이라는 이름의
    구역이 **두 개**이고, 그중 `L1010100` 은 2020-05-14 에 만료된 옛 코드다.
    2023~2026 구간에서 쓸 코드는 `L1100000`(+ 하위 4개 권역)이다. 자세한 내용은
    `resolve_seoul_regs` docstring 참조.

저장은 연 파티션
    특보는 연간 수십 건 수준이라 월 파티션이면 파일이 너무 잘게 쪼개진다.

사용법
    python -m src.collect.weather_warning --list-regions          # 특보구역 코드 조회
    python -m src.collect.weather_warning                          # 어제 1일 (증분)
    python -m src.collect.weather_warning --start 2023-01-01 --end 2026-07-31
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
    ApiError,
    get,
    kma_key,
    kma_rows,
    write_csv,
    raw_path,
    read_raw,
    as_text_frame,
)

SOURCE = "kma_warning"
REG_ENDPOINT = f"{KMA_BASE}/wrn_reg.php"
DATA_ENDPOINT = f"{KMA_BASE}/wrn_met_data.php"

# wrn_met_data.php 응답 컬럼 — ✅ 실제 응답으로 검증 완료
#
#   문서(apiList.do?seqApi=10&seqApiSub=288)가 나열한 순서와 **완전히 다르다.**
#   문서는 REG_ID/TM_ST/TM_ED/REG_SP/REG_KO… 로 시작하는 구역 마스터 컬럼을 앞에 두지만,
#   실제 출력은 TM_FC 부터 시작하고 구역 마스터 컬럼은 아예 없다. 실측 헤더:
#     # TM_FC, TM_EF, TM_IN, STN, REG_ID, WRN, LVL, CMD, GRD, CNT, RPT, T01…T18, =
#     202301231000,202301232100,202301230925,109,L1100100,C,3,1,00,4,101,,,…
#
#   ★ TM_ED(해제시각) 컬럼은 **존재하지 않는다.** ★
#   docs/datasets.md §7 의 `tm_ed` 는 사실과 다르다. 특보 해제는 별도
#   컬럼이 아니라 **CMD=3(해제) 인 별개의 행**으로 표현된다. 따라서 "특보 발효 구간"은
#   발표행(CMD=1/5/6)과 해제행(CMD=3)을 짝지어 transform 단계에서 만들어야 한다.
WRN_COLUMNS: list[str] = [
    "TM_FC",     # 발표시각 (KST)
    "TM_EF",     # 발효시각 (KST)
    "TM_IN",     # 입력시각 (KST)
    "STN",       # 발표관서
    "REG_ID",    # 특보구역코드
    "WRN",       # 특보종류코드
    "LVL",       # 특보수준
    "CMD",       # 특보명령
    "GRD",       # 태풍경보시 등급 (2011-09-05 이후 생산 중지)
    "CNT",       # 작업상태 (4:통보완료)
    "RPT",       # 통보문 발송구분
] + [f"T{i:02d}" for i in range(1, 19)]      # T01~T18 비고
MUST_HAVE = ("REG_ID", "TM_FC")

# wrn_reg.php (특보구역 마스터) — ✅ 실제 응답으로 검증 완료 (이 API는 활용신청 불필요)
#   1.REG_ID 특보구역코드 / 2.TM_ST 시작시각 / 3.TM_ED 종료시각 / 4.REG_SP 특성
#   5.REG_UP 상위 특보구역코드 / 6.REG_KO 특보구역명(약어) / 7.REG_NAME 특보구역명
#   + 문서화되지 않은 8번째 꼬리 컬럼(값이 항상 "=")이 실제로 붙어 나온다 → COL_7 로 들어감
REG_COLUMNS: list[str] = ["REG_ID", "TM_ST", "TM_ED", "REG_SP", "REG_UP",
                          "REG_KO", "REG_NAME"]

# 특보종류 코드.
# 출처: apiList.do?seqApi=10&seqApiSub=288 및 docs/datasets.md (두 곳이 일치)
# 실측 교차검증: 2023-01-23 서울 한파 특보가 WRN="C" 로 나옴 → C=한파 확인.
# (static/html/attach/wrn_table.html 은 C=대설, D=한파 로 적혀 있으나 **그쪽이 틀렸다**)
WRN_NAME = {
    "W": "강풍", "R": "호우", "C": "한파", "D": "건조", "O": "해일", "V": "풍랑",
    "T": "태풍", "S": "대설", "Y": "황사", "H": "폭염", "F": "안개", "K": "열대야",
}
# LVL / CMD — ✅ wrn_met_data.php 의 help=1 주석에서 그대로 옮김 (실측)
LVL_NAME = {"1": "예비", "2": "주의보", "3": "경보", "4": "중대경보"}   # 중대경보는 폭염만
CMD_NAME = {"1": "발표", "2": "대치", "3": "해제", "4": "대치해제(자동)",
            "5": "연장", "6": "변경", "7": "변경해제"}

PERMISSION_GUIDE = """
────────────────────────────────────────────────────────────────
  기상청 API 허브 권한 오류 (403) — 인증키는 맞지만 API 사용 권한이 없습니다.
────────────────────────────────────────────────────────────────
  해결: https://apihub.kma.go.kr 로그인 → [특보] 카테고리로 이동해
        아래 **두 API 각각** 상세 페이지에서 "활용신청" 버튼을 누르세요.
          · 특보구역   wrn_reg.php
          · 특보자료   wrn_met_data.php
  * 기상청 API 허브는 API 하나하나마다 활용신청이 따로 필요합니다.
  * 승인 반영까지 수 분 걸릴 수 있습니다. 이후 같은 명령을 다시 실행하세요.
────────────────────────────────────────────────────────────────"""


# ── 파싱 ──────────────────────────────────────────────────
# NOTE: weather_asos.py / weather_forecast.py 와 거의 같다. 세 수집기가 안정되면
#       _common.py 로 올릴 것 (지금은 _common.py 를 다른 작업이 함께 쓰고 있어 손대지 않는다).
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

    특보 응답은 TM_ED 처럼 이름이 겹칠 가능성이 있고, 중복 컬럼명이 있으면
    df["TM_ED"] 가 Series 가 아니라 DataFrame 을 돌려줘 조용히 깨진다.
    """
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def to_frame(text: str, fallback: list[str], must_have: tuple[str, ...]) -> pd.DataFrame:
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


def _tm(s: pd.Series) -> pd.Series:
    """YYYYMMDD / YYYYMMDDHH / YYYYMMDDHHMM 아무거나 → datetime64."""
    digits = (s.astype("string").str.replace(r"\D", "", regex=True)
               .str.slice(0, 12).str.ljust(12, "0"))
    return pd.to_datetime(digits, format="%Y%m%d%H%M", errors="coerce")


# ── 특보구역 조회 ─────────────────────────────────────────
def fetch_regions(key: str | None = None) -> pd.DataFrame:
    """wrn_reg.php 로 특보구역 목록을 받아온다. (서울 코드 하드코딩 방지용)"""
    text = get(REG_ENDPOINT, {"disp": "1", "help": "1", "authKey": key or kma_key()})
    return to_frame(text, REG_COLUMNS, ("REG_ID",))


def resolve_seoul_regs(key: str | None = None, *, include_sub: bool = True,
                       at: pd.Timestamp | None = None) -> list[tuple[str, str]]:
    """특보구역 목록에서 서울 구역을 찾아 [(REG_ID, 구역명), …] 반환.

    왜 이렇게까지 하나 — 실제 wrn_reg.php 응답을 보면 함정이 두 개 있다.

    1) **만료된 구역 레코드가 그대로 들어 있다.** '서울'이라는 이름은 두 번 나오는데
       `L1010100`(경기도 하위, TM_ED=2020-05-14 만료)과 `L1100000`(광역, 2100년까지 유효)이다.
       이름만 보고 앞엣것을 집으면 2023~2026년 구간에서 **조용히 0건**을 받는다.
       그래서 TM_ST ≤ 기준시각 ≤ TM_ED 로 유효한 레코드만 남긴다.

    2) **2020-05-14부터 서울은 4개 권역으로 쪼개졌다.**
       L1100100 서울동남권 / L1100200 서울동북권 / L1100300 서울서남권 / L1100400 서울서북권.
       특보가 권역 단위로 발표되면 상위 `L1100000` 만 조회해서는 잡히지 않을 수 있으므로
       기본적으로 하위 권역까지 함께 수집한다(`include_sub=False` 로 끌 수 있음).

    못 찾으면 예외를 던진다. 틀린 코드로 빈 결과를 받는 것보다 낫다.
    """
    regs = fetch_regions(key)
    name_col = next((c for c in ("REG_KO", "REG_NAME", "REG_EN") if c in regs.columns), None)
    if name_col is None or regs.empty:
        raise ApiError("wrn_reg.php 응답에서 구역명 컬럼(REG_KO/REG_NAME)을 찾지 못했습니다. "
                       "--reg 로 특보구역코드를 직접 지정하세요.")

    at = at or pd.Timestamp.now()
    live = regs
    if {"TM_ST", "TM_ED"} <= set(regs.columns):
        st, ed = _tm(regs["TM_ST"]), _tm(regs["TM_ED"])
        live = regs[(st.isna() | (st <= at)) & (ed.isna() | (ed >= at))]

    names = live[name_col].astype("string").fillna("")
    parent = live[names == "서울"]
    if parent.empty:                                   # 이름 규칙이 바뀐 경우 대비
        parent = live[names.str.startswith("서울", na=False)]
    if parent.empty:
        raise ApiError("특보구역 목록에서 현재 유효한 '서울' 구역을 찾지 못했습니다. "
                       "`--list-regions` 로 목록을 확인하고 --reg 로 직접 지정하세요.")

    row = parent.iloc[0]
    pid = str(row["REG_ID"])
    found = [(pid, str(row[name_col]))]

    if include_sub and "REG_UP" in live.columns:
        sub = live[(live["REG_UP"].astype("string") == pid) & (live["REG_ID"].astype("string") != pid)]
        found += [(str(r["REG_ID"]), str(r[name_col])) for _, r in sub.iterrows()]
    return found


# ── 수집 ──────────────────────────────────────────────────
def fetch_year(tmfc1: str, tmfc2: str, reg: str, key: str) -> pd.DataFrame:
    """[tmfc1, tmfc2] (YYYYMMDDHHMM) 발표분 특보 이력."""
    text = get(DATA_ENDPOINT, {
        "tmfc1": tmfc1, "tmfc2": tmfc2, "reg": reg,
        "disp": "1",     # 0=기본, 1=+내용, 2=+담당자
        "help": "1",
        "authKey": key,
    })
    df = to_frame(text, WRN_COLUMNS, MUST_HAVE)
    if df.empty:
        return df

    tm_fc = _tm(df["TM_FC"])
    if tm_fc.isna().all():
        raise ApiError(
            f"TM_FC 파싱 실패 — 컬럼 순서가 코드의 가정과 다릅니다. "
            f"응답 첫 줄: {kma_rows(text)[0][:200]!r} / 사용한 컬럼: {list(df.columns)}")

    # docs/datasets.md §7 weather_warning 컬럼
    out = pd.DataFrame(index=df.index)
    out["region"] = "서울"
    out["reg_id"] = df.get("REG_ID", pd.NA)
    out["wrn"] = df.get("WRN", pd.NA)
    out["lvl"] = df.get("LVL", pd.NA)
    out["tm_fc"] = tm_fc                                       # 발표
    out["tm_ef"] = _tm(df["TM_EF"]) if "TM_EF" in df else pd.NaT   # 발효
    # 해제시각 — 이 API에는 TM_ED 컬럼이 **없다**(위 WRN_COLUMNS 주석 참조).
    # docs/datasets.md §7 스키마 자리만 유지하고 NaT 로 둔다. 실제 해제는 CMD=3 인 별도 행이며,
    # "발효 구간(발표~해제)" 조립은 transform 단계에서 발표행↔해제행을 짝지어 만든다.
    tm_ed_cols = [c for c in df.columns if re.fullmatch(r"TM_ED(_\d+)?", c)]
    out["tm_ed"] = _tm(df[tm_ed_cols[-1]]) if tm_ed_cols else pd.NaT
    out["cmd"] = df.get("CMD", pd.NA)
    # 코드 → 한글. 원본 코드 컬럼(WRN/LVL/CMD)도 그대로 남으므로 되돌릴 수 있다.
    out["wrn_name"] = out["wrn"].map(WRN_NAME)
    out["lvl_name"] = out["lvl"].map(LVL_NAME)
    out["cmd_name"] = df["CMD"].map(CMD_NAME) if "CMD" in df else pd.NA

    dup = [c for c in df.columns if c in out.columns]
    return pd.concat([out, df.drop(columns=dup)], axis=1)


def _merge_partition(df: pd.DataFrame, partition: str, keys: list[str]) -> pd.DataFrame:
    """기존 파티션과 합쳐 멱등성을 보장한다(같은 키는 새 응답이 이긴다)."""
    path = raw_path(SOURCE, partition)
    if path.exists():
        # CSV 는 타입을 보존하지 않는다. 양쪽을 문자열 표현으로 맞춰야
        # 키 비교가 어긋나지 않는다 (저장된 "108" vs 새 108).
        df = pd.concat([read_raw(path), as_text_frame(df)], ignore_index=True)
    return (df.drop_duplicates(subset=keys, keep="last")
              .sort_values(keys)
              .reset_index(drop=True))


# WRN2_MET_DATA 의 PK 는 (TM_FC, REG_ID, WRN_TP). 같은 특보의 연장·해제는
# CMD/TM_EF 가 달라지므로 함께 키에 넣어야 이력이 덮이지 않는다.
DEDUP_KEYS = ["reg_id", "tm_fc", "wrn", "lvl", "tm_ef"]


def _year_range(start: str, end: str):
    """[start, end] 를 연 단위 (첫날, 마지막날) 쌍으로 쪼갠다."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cur = s.replace(month=1, day=1)
    while cur <= e:
        nxt = cur + pd.offsets.YearBegin(1)
        yield max(cur, s), min(nxt - pd.Timedelta(days=1), e)
        cur = nxt


def backfill(start: str, end: str, reg: str | None = None,
             *, include_sub: bool = True) -> list[str]:
    """연 단위 백필. 실패한 연도 목록을 반환한다(전체 중단하지 않음)."""
    key = kma_key()
    if reg is None:
        targets = resolve_seoul_regs(key, include_sub=include_sub)
        print("특보구역 자동 확정: " + ", ".join(f"{c}({n})" for c, n in targets))
    else:
        targets = [(reg, "(--reg 직접 지정)")]

    failed: list[str] = []
    for s, e in _year_range(start, end):
        partition = s.strftime("%Y")
        label = f"[{partition}] {s:%Y-%m-%d} ~ {e:%Y-%m-%d}"
        tmfc1, tmfc2 = s.strftime("%Y%m%d") + "0000", e.strftime("%Y%m%d") + "2359"

        frames, errs = [], []
        for code, name in targets:
            try:
                df = fetch_year(tmfc1, tmfc2, code, key)
                if not df.empty:
                    df["region"] = name          # 광역/권역 구분 보존
                    frames.append(df)
                print(f"{label} {code}({name}) … {len(df):>4}행")
            except ApiError as exc:
                if "403" in str(exc) or "활용신청" in str(exc):
                    raise                        # 권한 문제는 모든 연도에서 동일 → 즉시 중단
                errs.append(f"{code}: {_redact(exc)[:120]}")
            except Exception as exc:             # noqa: BLE001
                errs.append(f"{code}: {type(exc).__name__}: {_redact(exc)[:120]}")

        if errs:
            print(f"{label} … 실패: {' | '.join(errs)}")
            failed.append(partition)
        if not frames:
            continue

        merged = _merge_partition(pd.concat(frames, ignore_index=True), partition, DEDUP_KEYS)
        path = write_csv(merged, SOURCE, partition)
        print(f"{label} … 저장 {len(merged):>4}행 → {path.name}")

    return failed


# ── CLI ───────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="기상청 기상특보 이력 수집 (서울)")
    p.add_argument("--start", help="발표일 시작 YYYY-MM-DD (기본: 어제)")
    p.add_argument("--end", help="발표일 종료 YYYY-MM-DD (기본: 어제)")
    p.add_argument("--reg", help="특보구역코드 직접 지정 (기본: wrn_reg.php 로 서울 자동 확정)")
    p.add_argument("--no-subregions", action="store_true",
                   help="서울 하위 권역(동남/동북/서남/서북) 제외하고 광역만 수집")
    p.add_argument("--list-regions", action="store_true",
                   help="특보구역 목록만 출력하고 종료 (--grep 로 필터)")
    p.add_argument("--grep", help="--list-regions 결과를 구역명으로 필터")
    a = p.parse_args(argv)

    try:
        if a.list_regions:
            regs = fetch_regions()
            if a.grep:
                regs = regs[regs.apply(
                    lambda r: r.astype("string").str.contains(a.grep, na=False).any(), axis=1)]
            print(f"특보구역 {len(regs)}건")
            print(regs.to_string(max_rows=600))
            print("\n[참고] TM_ED 가 과거인 행은 **만료된 구역**입니다. "
                  "수집기는 만료 레코드를 자동으로 걸러냅니다.")
            return 0

        yesterday = (date.today() - timedelta(days=1)).isoformat()
        start = a.start or yesterday
        end = a.end or (a.start or yesterday)
        print(f"기상특보 이력 수집: 발표일 {start} ~ {end}")
        failed = backfill(start, end, a.reg, include_sub=not a.no_subregions)
    except ApiError as exc:
        if "403" in str(exc) or "활용신청" in str(exc):
            print(PERMISSION_GUIDE, file=sys.stderr)
            return 2
        print(f"\n[중단] API 오류: {_redact(exc)[:400]}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"\n[중단] {exc}", file=sys.stderr)
        return 1

    if failed:
        print(f"\n실패한 연도 {len(failed)}개: {', '.join(failed)}")
        print("→ 해당 연도만 --start/--end 로 다시 실행하세요.")
        return 1
    print("\n완료. 실패한 연도 없음.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
