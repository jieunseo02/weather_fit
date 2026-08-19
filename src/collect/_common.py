"""수집 스크립트 공통 유틸.

모든 수집기가 공유하는 경로·설정·HTTP·저장 규약. 수집기별 로직은 각 모듈에 둔다.
"""
from __future__ import annotations

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


def load_secrets() -> dict:
    path = CONFIGS / "secrets.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. configs/secrets.example.yaml 을 복사해 만드세요.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
def write_parquet(df: pd.DataFrame, source: str, partition: str) -> Path:
    """data/raw/{source}/{partition}.parquet 로 저장."""
    out = DATA_RAW / source
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{partition}.parquet"
    df.to_parquet(path, index=False)
    return path




# ── 기간 유틸 ─────────────────────────────────────────────
def month_range(start: str, end: str) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end] 를 월 단위 (첫날, 말일) 쌍으로 쪼갠다. YYYY-MM-DD 입력."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cur = s.replace(day=1)
    while cur <= e:
        nxt = (cur + pd.offsets.MonthBegin(1))
        yield max(cur, s), min(nxt - pd.Timedelta(days=1), e)
        cur = nxt
