"""수집 스크립트 공통 유틸.

모든 수집기가 공유하는 경로·설정·HTTP·저장 규약. 수집기별 로직은 각 모듈에 둔다.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT_ROOT / "configs"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

KMA_BASE = "https://apihub.kma.go.kr/api/typ01/url"
NAVER_BASE = "https://openapi.naver.com"

# 서울 고정 상수 (docs/datasets.md 참조)
SEOUL = {
    "asos_stn": "108",       # 종관기상관측 지점
    "fct_stn": "109",        # 예보관서
    "reg_temp": "11B10101",  # 중기 기온예보 구역
    "reg_land": "11B00000",  # 중기 육상예보 구역 (서울·인천·경기)
}


# ── 설정 ──────────────────────────────────────────────────
def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


# 환경변수 이름 → secrets 구조 상의 위치
ENV_MAP = {
    "KMA_APIHUB_AUTH_KEY":   ("kma_apihub", "auth_key"),
    "NAVER_CLIENT_ID":       ("naver", "client_id"),
    "NAVER_CLIENT_SECRET":   ("naver", "client_secret"),
    "DATA_GO_KR_SERVICE_KEY": ("data_go_kr", "service_key"),
    "KOSIS_API_KEY":         ("kosis", "api_key"),
}


def _load_dotenv(path: Path) -> dict[str, str]:
    """의존성 없이 .env 를 읽는다. `KEY=value`, `export KEY=value`, `#` 주석 지원."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.removeprefix("export ").partition("=")
        val = val.strip().strip('"').strip("'")
        if val:
            out[key.strip()] = val
    return out


def load_secrets() -> dict:
    """`.env` → 환경변수 → `configs/secrets.yaml` 순으로 키를 찾는다.

    셋 중 아무거나 쓰면 된다. 같은 키가 여러 곳에 있으면 앞선 것이 이긴다.
    """
    merged: dict[str, dict[str, str]] = {}

    env = {**_load_dotenv(PROJECT_ROOT / ".env"), **os.environ}
    for var, (section, field) in ENV_MAP.items():
        if env.get(var):
            merged.setdefault(section, {})[field] = env[var]

    path = CONFIGS / "secrets.yaml"
    if path.exists():
        for section, fields in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items():
            for field, value in (fields or {}).items():
                if value:
                    merged.setdefault(section, {}).setdefault(field, value)

    if not merged:
        raise FileNotFoundError(
            "API 키를 찾을 수 없습니다. 다음 중 하나를 준비하세요.\n"
            f"  1) {PROJECT_ROOT / '.env'}  (.env.example 참고)\n"
            f"  2) 환경변수 {', '.join(ENV_MAP)}\n"
            f"  3) {path}  (configs/secrets.example.yaml 복사)")
    return merged


def kma_key() -> str:
    return load_secrets()["kma_apihub"]["auth_key"]


def naver_headers() -> dict[str, str]:
    n = load_secrets()["naver"]
    return {
        "X-Naver-Client-Id": n["client_id"],
        "X-Naver-Client-Secret": n["client_secret"],
        "Content-Type": "application/json",
    }


# ── HTTP ──────────────────────────────────────────────────
class ApiError(RuntimeError):
    pass


def get(url: str, params: dict, *, retries: int = 3, pause: float = 0.4) -> str:
    """GET 후 본문 텍스트 반환. 5xx·타임아웃만 재시도한다."""
    last: Exception | None = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code >= 500:
                raise ApiError(f"{r.status_code}")
            if r.status_code != 200:
                raise ApiError(f"{r.status_code} {r.text[:200]}")
            time.sleep(pause)
            return r.text
        except (requests.RequestException, ApiError) as e:
            last = e
            if isinstance(e, ApiError) and not str(e)[:1].isdigit():
                raise
            time.sleep(2 ** i)
    raise ApiError(f"재시도 {retries}회 실패: {last}")


def post_json(url: str, headers: dict, body: dict,
              *, retries: int = 3, pause: float = 0.4) -> dict:
    last: Exception | None = None
    for i in range(retries):
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            time.sleep(pause)
            return r.json()
        last = ApiError(f"{r.status_code} {r.text[:300]}")
        if r.status_code < 500:      # 4xx는 재시도해도 같다
            raise last
        time.sleep(2 ** i)
    raise last  # type: ignore[misc]


# ── 기상청 API 허브 응답 파싱 ─────────────────────────────
def kma_rows(text: str) -> list[str]:
    """`#START7777` ~ `#7777END` 사이의 데이터 행만 반환.

    인증·권한 오류는 JSON 바디로 오므로 여기서 걸러 예외를 던진다.
    """
    t = text.strip()
    if t.startswith("{"):
        raise ApiError(f"API 오류 응답: {' '.join(t.split())[:200]}")
    return [ln.strip() for ln in t.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


# ── 저장 ──────────────────────────────────────────────────
# raw 는 CSV 로 둔다. 전체가 수 MB 수준이라 Parquet 의 압축·타입보존 이점보다
# "엑셀로도 열리고 diff 도 보이는" 단순함이 낫다.
# 대신 CSV 는 타입을 보존하지 않으므로, 재읽기가 필요한 곳에서는
# 아래 as_text_frame / read_raw 로 **문자열 표현을 통일**해 다뤄야 한다.
# 그러지 않으면 저장된 "108" 이 다시 읽을 때 108 로 추론돼 키 비교가 어긋난다.

def write_csv(df: pd.DataFrame, source: str, partition: str) -> Path:
    """data/raw/{source}/{partition}.csv 로 저장."""
    out = DATA_RAW / source
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{partition}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def raw_path(source: str, partition: str) -> Path:
    return DATA_RAW / source / f"{partition}.csv"


def as_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    """to_csv 가 써낼 텍스트와 같은 표현으로 정규화한다 (결측은 빈 문자열)."""
    return df.astype(object).where(df.notna(), "").astype(str)


def read_raw(path: Path) -> pd.DataFrame:
    """저장된 CSV 를 타입 추론 없이 문자열 그대로 읽는다."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")




# ── 기간 유틸 ─────────────────────────────────────────────
def month_range(start: str, end: str) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end] 를 월 단위 (첫날, 말일) 쌍으로 쪼갠다. YYYY-MM-DD 입력."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cur = s.replace(day=1)
    while cur <= e:
        nxt = (cur + pd.offsets.MonthBegin(1))
        yield max(cur, s), min(nxt - pd.Timedelta(days=1), e)
        cur = nxt
