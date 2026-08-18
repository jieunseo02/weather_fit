"""W2·W3. 중기예보 "발표분 아카이브" 수집기 — 서울.

    GET fct_afs_wc.php?tmfc1=2023010100&tmfc2=2023013123&reg=11B10101&disp=1&help=1   # 기온
    GET fct_afs_wl.php?tmfc1=…&tmfc2=…&reg=11B00000&disp=1&help=1                     # 육상

왜 이 데이터셋이 필요한가
    관측치(W1)로 학습한 모델은 "내일의 실제 기온"을 미리 아는 셈이라 실전에서 못 쓴다.
    발표분 아카이브를 쓰면 **예보 오차까지 포함한 현실적인 모델**이 된다.

그래서 이 모듈의 핵심 규칙은 하나다
    **발표시각(tmfc)과 예보 대상일(target_date)을 절대 섞지 않는다.**
    두 값을 별도 컬럼으로 저장해 두어야 나중에 "발표일 ≤ D-1, 대상일 = D" 로
    조인해 데이터 누수를 막을 수 있다. 한 컬럼으로 뭉개면 이 데이터셋은 의미가 없다.

    ※ 중기예보는 통상 **+3~10일** 앞을 예보한다. 즉 대상일 D 에 대해 가장 가까운
      발표분은 D-3 근처이지 D-1 이 아니다. 조인 시에는 "대상일 D 에 대한 발표 중
      가장 최신이면서 tmfc < D 인 것"을 고르는 식으로 처리한다 (transform 단계).

발표 주기
    1일 2회 06시 / 18시 KST. 월 구간을 tmfc1=YYYYMMDD00, tmfc2=YYYYMMDD23 으로
    잡으면 두 발표분을 모두 포함한다.

사용법
    python -m src.collect.weather_forecast                                    # 어제 발표분
    python -m src.collect.weather_forecast --start 2023-01-01 --end 2026-07-31
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
    write_meta,
    write_parquet,
)

SOURCE = "kma_forecast"

# 두 API 정의. src 컬럼으로 구분해 **한 파일에 union** 으로 넣는다(조인하지 않는다).
#   kind : 저장 시 src 컬럼 값
#   reg  : 서울 예보구역 (기온구역과 육상구역이 다르다)
ENDPOINTS = {
    "temp": {                                   # W2 중기 기온예보
        "url": f"{KMA_BASE}/fct_afs_wc.php",
        "reg": SEOUL["reg_temp"],               # 11B10101 서울
        "name": "중기기온예보",
    },
    "land": {                                   # W3 중기 육상예보
        "url": f"{KMA_BASE}/fct_afs_wl.php",
        "reg": SEOUL["reg_land"],               # 11B00000 서울·인천·경기
        "name": "중기육상예보",
    },
}

# 응답 컬럼 순서 (fallback). 실제 파싱은 응답의 `#` 실제 출력 헤더 행을 우선한다.
#
# ── land (fct_afs_wl.php) : ✅ 실제 응답으로 검증 완료 ──────────────────────
#   help=1 의 번호 목록은 14개를 선언하지만 **실제 데이터 행은 11개**다.
#   MAN_ID(7) / MAN_FC(8) / REG_NAME(9) 은 출력되지 않는다(mode 기본값에서 생략).
#   번호 목록 순서로 파싱하면 SKY 자리에 MAN_ID 가 들어가 컬럼이 통째로 밀린다.
#   실측 헤더/행:
#     # REG_ID TM_FC        TM_EF        MOD STN C SKY  PRE  CONF WF   RN_ST
#     11B00000,202301010600,202301040000,A02,109,2,WB01,WB00,없음,맑음,0,=
#   (행 끝 "=" 는 종료 마커 → to_frame 이 제거)
#
# ── temp (fct_afs_wc.php) : ✅ 실제 응답으로 검증 완료 ──────────────────────
#   여기도 번호 목록(11개)과 실제 출력(12개)이 다르다. 게다가 기상청 문서 자체에
#   **번호 오타**가 있다 — MAX / MIN_L / MIN_H / MAX_L / MAX_H 가 전부 "11." 로 매겨져 있다.
#   번호 목록을 믿으면 컬럼이 5개 사라진다. 실측 헤더/행:
#     # REG_ID TM_FC        TM_EF        MOD STN C MIN MAX MIN_L MIN_H MAX_L MAX_H
#     11B10101,202301010600,202301040000,A01,109,2,-7,2,0,1,1,1,=
#   MIN_L/MIN_H/MAX_L/MAX_H 는 기온 **범위(하한/상한)** 이지 절대기온이 아니다.
FALLBACK_COLUMNS = {
    "temp": ["REG_ID", "TM_FC", "TM_EF", "MOD", "STN", "C",
             "MIN", "MAX", "MIN_L", "MIN_H", "MAX_L", "MAX_H"],
    "land": ["REG_ID", "TM_FC", "TM_EF", "MOD", "STN", "C",
             "SKY", "PRE", "CONF", "WF", "RN_ST"],
}
MUST_HAVE = ("REG_ID", "TM_FC", "TM_EF")

# 코드값 — SKY/PRE 는 실제 응답으로 확인됨.
# raw 에는 코드값 그대로 저장하고, 아래 dict 는 한글 매핑을 "노출만" 한다(raw 보존 원칙).
SKY_CODE = {"WB01": "맑음", "WB02": "구름조금", "WB03": "구름많음", "WB04": "흐림"}
PRE_CODE = {"WB00": "없음", "WB09": "비", "WB10": "비", "WB11": "비/눈",
            "WB12": "눈", "WB13": "눈/비"}
# MOD(구간) — ✅ 확인됨: A01=24시간, A02=12시간.
#   즉 A02 는 같은 대상일에 00시(오전)/12시(오후) 두 행이 생긴다. 일 단위로 접을 때
#   "A02 두 행을 min/max 로 합칠지, 12시 행만 쓸지"를 정해야 한다 → transform 단계의 결정사항.
MOD_NAME = {"A01": "24시간", "A02": "12시간"}

MISSING = {"", "-", "-99", "-99.0", "-999", "-999.0", "-9999", "-9999.0"}

PERMISSION_GUIDE = """
────────────────────────────────────────────────────────────────
  기상청 API 허브 권한 오류 (403) — 인증키는 맞지만 API 사용 권한이 없습니다.
────────────────────────────────────────────────────────────────
  해결: https://apihub.kma.go.kr 로그인 → [예보] > [중기예보] 로 이동해
        아래 **두 API 각각** 상세 페이지에서 "활용신청" 버튼을 누르세요.
          · 중기기온예보  fct_afs_wc.php
          · 중기육상예보  fct_afs_wl.php
  * 기상청 API 허브는 API 하나하나마다 활용신청이 따로 필요합니다.
  * 승인 반영까지 수 분 걸릴 수 있습니다. 이후 같은 명령을 다시 실행하세요.
────────────────────────────────────────────────────────────────"""


# ── 파싱 ──────────────────────────────────────────────────
# NOTE: 아래 _header_from / to_frame / _merge_partition 은 weather_asos.py·
#       weather_warning.py 와 거의 같다. 세 수집기가 안정되면 _common.py 로 올릴 것.
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


def _num(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.strip().replace(dict.fromkeys(MISSING, pd.NA))
    return pd.to_numeric(cleaned, errors="coerce")


def _tm(s: pd.Series) -> pd.Series:
    """YYYYMMDD / YYYYMMDDHH / YYYYMMDDHHMM 아무거나 → datetime64."""
    digits = (s.astype("string").str.replace(r"\D", "", regex=True)
               .str.slice(0, 12).str.ljust(12, "0"))
    return pd.to_datetime(digits, format="%Y%m%d%H%M", errors="coerce")


# ── 수집 ──────────────────────────────────────────────────
def fetch(kind: str, tmfc1: str, tmfc2: str, key: str, reg: str | None = None) -> pd.DataFrame:
    """한 엔드포인트의 [tmfc1, tmfc2] 발표분을 가져온다. tmfc 는 YYYYMMDDHH."""
    spec = ENDPOINTS[kind]
    reg = reg or spec["reg"]
    text = get(spec["url"], {
        "tmfc1": tmfc1, "tmfc2": tmfc2, "reg": reg,
        "disp": "1",     # 쉼표 구분
        "help": "1",     # 컬럼 헤더 동봉 → 문서 대신 응답으로 컬럼 확정
        "authKey": key,
    })
    df = to_frame(text, FALLBACK_COLUMNS[kind], MUST_HAVE)
    if df.empty:
        return df

    # 컬럼 정렬이 틀리면 시각이 안 읽힌다. 조용히 깨지는 대신 여기서 터뜨린다.
    tmfc, tmef = _tm(df["TM_FC"]), _tm(df["TM_EF"])
    if tmfc.isna().all() or tmef.isna().all():
        raise ApiError(
            f"[{kind}] TM_FC/TM_EF 파싱 실패 — 컬럼 순서가 코드의 가정과 다릅니다. "
            f"응답 첫 줄: {kma_rows(text)[0][:200]!r} / 사용한 컬럼: {list(df.columns)}")

    out = pd.DataFrame(index=df.index)
    out["src"] = kind                              # temp(기온) | land(육상)
    out["region"] = "서울"
    out["reg_id"] = df.get("REG_ID", pd.NA)
    out["tmfc"] = tmfc                             # ★ 발표시각
    out["tm_ef"] = tmef                            # 예보 유효시각 (00=오전 / 12=오후)
    out["target_date"] = tmef.dt.normalize()       # ★ 예보 대상일
    out["mod"] = df.get("MOD", pd.NA)

    # docs/datasets.md §7 weather_forecast_daily 컬럼. 없는 항목은 NA (temp/land 가 서로 보완)
    out["t_max"] = _num(df["MAX"]) if "MAX" in df else pd.NA
    out["t_min"] = _num(df["MIN"]) if "MIN" in df else pd.NA
    out["rn_st"] = _num(df["RN_ST"]) if "RN_ST" in df else pd.NA
    out["sky"] = df["SKY"] if "SKY" in df else pd.NA
    out["wf"] = df["WF"] if "WF" in df else pd.NA
    # 코드 → 한글은 되돌릴 수 있는 매핑이라 raw 에 같이 둔다 (원본 SKY/PRE 도 보존됨)
    out["sky_name"] = out["sky"].map(SKY_CODE) if "SKY" in df else pd.NA
    out["pre_name"] = df["PRE"].map(PRE_CODE) if "PRE" in df else pd.NA

    dup = [c for c in df.columns if c in out.columns]
    return pd.concat([out, df.drop(columns=dup)], axis=1)


def _merge_partition(df: pd.DataFrame, partition: str, keys: list[str]) -> pd.DataFrame:
    """기존 파티션과 합쳐 멱등성을 보장한다(같은 키는 새 응답이 이긴다)."""
    path = DATA_RAW / SOURCE / f"{partition}.parquet"
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    return (df.drop_duplicates(subset=keys, keep="last")
              .sort_values(keys)
              .reset_index(drop=True))


# 한 발표분 안에서 행을 유일하게 만드는 키.
# (발표시각, 구역, 대상시각, 모드) 조합이면 중복 없이 재수집이 가능하다.
DEDUP_KEYS = ["src", "reg_id", "tmfc", "tm_ef", "mod"]


def backfill(start: str, end: str, kinds: list[str] | None = None) -> tuple[list[str], set[str]]:
    """월 단위 백필. (실패한 월 목록, 권한 없는 엔드포인트 집합) 반환.

    두 엔드포인트의 활용신청이 **따로** 승인되므로, 한쪽만 403 인 상태가 실제로 생긴다.
    그 경우 전체를 멈추지 않고 **승인된 쪽만 계속 수집**하고, 마지막에 미승인 API를 안내한다.
    """
    key = kma_key()
    kinds = kinds or list(ENDPOINTS)
    failed: list[str] = []
    denied: set[str] = set()          # 403 이 뜬 엔드포인트 (이후 달에서는 호출 생략)

    for s, e in month_range(start, end):
        partition = s.strftime("%Y-%m")
        label = f"[{partition}] {s:%Y-%m-%d} ~ {e:%Y-%m-%d}"
        tmfc1, tmfc2 = s.strftime("%Y%m%d") + "00", e.strftime("%Y%m%d") + "23"

        frames, errs = [], []
        for kind in kinds:
            if kind in denied:            # 이미 권한 없음이 확인된 API는 다시 때리지 않는다
                continue
            try:
                df = fetch(kind, tmfc1, tmfc2, key)
                if not df.empty:
                    frames.append(df)
                print(f"{label} {ENDPOINTS[kind]['name']} … {len(df):>4}행")
            except ApiError as exc:
                if "403" in str(exc) or "활용신청" in str(exc):
                    denied.add(kind)      # 모든 달에서 동일하게 실패 → 이 API만 건너뛴다
                    print(f"{label} {ENDPOINTS[kind]['name']} … 403 활용신청 필요 → 이후 건너뜀")
                    continue
                errs.append(f"{kind}: {_redact(exc)[:120]}")
            except Exception as exc:         # noqa: BLE001
                errs.append(f"{kind}: {type(exc).__name__}: {_redact(exc)[:120]}")

        if errs:
            print(f"{label} … 실패: {' | '.join(errs)}")
            failed.append(partition)
        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)
        merged = _merge_partition(merged, partition, DEDUP_KEYS)
        path = write_parquet(merged, SOURCE, partition)
        print(f"{label} … 저장 {len(merged):>4}행 → {path.name}")

    write_meta(SOURCE, {
        "collected_at": date.today().isoformat(),
        "source": "kma_apihub_fct_afs_wc + fct_afs_wl",
        "endpoints": {k: v["url"] for k, v in ENDPOINTS.items()},
        "request": {"tmfc1": start, "tmfc2": end, "disp": 1, "help": 1,
                    "reg": {k: v["reg"] for k, v in ENDPOINTS.items()}},
        "partition": "{yyyy-mm}.parquet (tmfc 발표시각 기준 월 파티션)",
        "notes": ("중기예보 발표분 아카이브. src 컬럼으로 temp(기온)/land(육상)을 구분해 "
                  "union 저장(조인 금지). tmfc=발표시각, target_date=예보 대상일. "
                  "학습 조인 시 tmfc < target_date 조건을 반드시 걸어야 누수가 없다."),
        "failed_months": failed,
        "permission_denied": sorted(denied),
    })
    return failed, denied


# ── CLI ───────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="기상청 중기예보(기온·육상) 발표분 아카이브 수집 (서울)")
    p.add_argument("--start", help="발표일 시작 YYYY-MM-DD (기본: 어제)")
    p.add_argument("--end", help="발표일 종료 YYYY-MM-DD (기본: 어제)")
    p.add_argument("--kind", choices=["temp", "land", "both"], default="both",
                   help="temp=기온예보, land=육상예보 (기본 both)")
    a = p.parse_args(argv)

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    start = a.start or yesterday
    end = a.end or (a.start or yesterday)
    kinds = list(ENDPOINTS) if a.kind == "both" else [a.kind]

    print(f"중기예보 발표분 수집: 발표일 {start} ~ {end} / {', '.join(kinds)}")
    try:
        failed, denied = backfill(start, end, kinds)
    except ApiError as exc:
        if "403" in str(exc) or "활용신청" in str(exc):
            print(PERMISSION_GUIDE, file=sys.stderr)
            return 2
        print(f"\n[중단] API 오류: {_redact(exc)[:400]}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"\n[중단] {exc}", file=sys.stderr)
        return 1

    rc = 0
    if denied:
        names = ", ".join(f"{ENDPOINTS[k]['name']}({ENDPOINTS[k]['url'].rsplit('/', 1)[-1]})"
                          for k in sorted(denied))
        print(f"\n활용신청이 필요한 API: {names}", file=sys.stderr)
        print(PERMISSION_GUIDE, file=sys.stderr)
        rc = 2 if len(denied) == len(kinds) else 1
    if failed:
        print(f"\n실패한 월 {len(failed)}개: {', '.join(failed)}")
        print("→ 해당 월만 --start/--end 로 다시 실행하세요.")
        rc = rc or 1
    if not denied and not failed:
        print("\n완료. 실패한 월 없음.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
