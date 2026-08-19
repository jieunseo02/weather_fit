"""네이버 데이터랩 수집기 3종(T1/T3/T4)이 공유하는 로직.

핵심은 두 가지다.

1. **배치 구성** — 데이터랩은 요청당 대상 수에 상한이 있다(카테고리 3, 키워드 5, 그룹 5).
   상한을 그대로 채우면 안 되고, **슬롯 1개는 앵커에 고정**해야 한다.
2. **앵커 재정규화** — 응답 ``ratio`` 는 "요청에 포함된 대상들 × 요청 기간" 안에서
   최댓값을 100으로 한 상대값이다. 배치가 다르면 분모가 달라 **배치 간 비교가 불가능**하다.
   모든 배치에 공통 앵커를 넣고 ``rescaled = ratio / anchor_ratio`` 로 나누면
   앵커를 1.0으로 하는 공통 축 위에 전체 대상을 올릴 수 있다.

왜 ``_common.py`` 가 아니라 별도 파일인가: ``_common.py`` 는 기상청 수집기와 공유하는
범용 유틸이고, 여기 있는 것은 데이터랩 전용 규칙이다. 특히 재정규화는 틀리면 분석 전체가
조용히 무효가 되는 부분이라 **한 곳에만** 두고 3개 수집기가 같은 구현을 쓰게 한다.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

from ._common import NAVER_BASE, load_yaml, naver_headers, post_json

T = TypeVar("T")

# docs/datasets.md §7 shopping_trend_daily 스키마 + raw 추적용 2개(cid, is_anchor).
# cid/is_anchor 는 processed 단계에서 설정 파일을 다시 읽지 않고도
# "이 행이 어느 카테고리의 앵커였나"를 알 수 있게 하려고 raw에 남긴다.
SCHEMA_COLS = [
    "date", "source",
    "category_l1", "category_l2", "keyword",
    "ratio", "rescaled",
    "batch_id",
    "gender", "age_group",
    "cid", "is_anchor",
]


# ── 설정 ──────────────────────────────────────────────────
def load_categories() -> dict:
    """configs/categories.yaml — 타깃 정의의 단일 진실 원천."""
    return load_yaml("categories.yaml")


def keyword_anchor(cfg: dict, cid: str) -> str:
    """해당 카테고리에서 쓸 앵커 키워드.

    T3의 ratio는 '카테고리 안에서의' 상대값이라, 그 카테고리에서 클릭이 잡히지 않는
    키워드는 앵커가 될 수 없다(ratio 0 → 0으로 나누기). 그래서 cid별 앵커를 둔다.
    """
    by_cid = cfg.get("keyword_anchor_by_cid") or {}
    return by_cid.get(str(cid)) or cfg["anchor"]["keyword"]


# ── 배치 구성 ─────────────────────────────────────────────
def chunk_with_anchor(items: Sequence[T], anchor: T, slots: int,
                      key: Callable[[T], Any] = lambda x: x) -> list[list[T]]:
    """앵커를 모든 배치의 첫 슬롯에 고정하고 나머지를 ``slots-1`` 개씩 나눈다.

    ``items`` 에 앵커와 같은 대상이 들어 있으면 중복 요청이 되므로 제거한다
    (예: 앵커 키워드 '반팔티'는 top_summer 그룹에도 들어 있다).
    """
    if slots < 2:
        raise ValueError("앵커 슬롯 1개가 필요하므로 slots는 2 이상이어야 한다")
    akey = key(anchor)
    rest = [x for x in items if key(x) != akey]
    per = slots - 1
    if not rest:                       # 앵커만 있는 경우도 1회는 요청한다
        return [[anchor]]
    return [[anchor, *rest[i:i + per]] for i in range(0, len(rest), per)]


# ── 응답 파싱 ─────────────────────────────────────────────
def iter_results(payload: dict) -> Iterable[tuple[str, list[dict]]]:
    """데이터랩 응답에서 (title, data) 쌍을 뽑는다.

    응답 구조는 T1/T3/T4가 동일하다::

        {"startDate":..., "endDate":..., "timeUnit":"date",
         "results":[{"title":"여성의류", "category":["50000167"],
                     "data":[{"period":"2026-07-01","ratio":56.5}, ...]}]}

    대상 식별 필드명만 다르다(category / keyword / keywords). title로 통일해 쓴다.
    """
    results = payload.get("results")
    if not results:
        raise ValueError(f"results 비어 있음 — 응답: {str(payload)[:300]}")
    for r in results:
        yield r["title"], r.get("data") or []


# ── 앵커 재정규화 (이 모듈의 존재 이유) ───────────────────
def add_rescaled(df: pd.DataFrame) -> pd.DataFrame:
    """``is_anchor`` 행의 ratio를 1.0으로 두고 같은 (batch_id, date) 안에서 재정규화.

    - 앵커 자신의 ``rescaled`` 는 정의상 정확히 1.0이 된다(검증 포인트).
    - **앵커 ratio가 0이거나 그 날 앵커 행이 없으면 ``rescaled`` 는 NaN.**
      0으로 나누면 inf가 섞여 들어가 이후 평균·상관이 전부 오염된다.
      NaN으로 남겨 두면 "이 날은 비교 불가"라는 사실이 데이터에 보존된다.
    """
    if df.empty:
        df["rescaled"] = pd.Series(dtype="float64")
        return df

    anchors = df.loc[df["is_anchor"], ["batch_id", "date", "ratio"]]
    dup = anchors.duplicated(subset=["batch_id", "date"]).sum()
    if dup:
        raise ValueError(f"배치·날짜당 앵커 행이 2개 이상이다 ({dup}건) — 배치 구성 오류")

    lookup = anchors.set_index(["batch_id", "date"])["ratio"]
    idx = pd.MultiIndex.from_frame(df[["batch_id", "date"]])
    denom = lookup.reindex(idx).to_numpy(dtype="float64")
    denom = np.where(denom == 0, np.nan, denom)      # 0 나누기 방지

    df["rescaled"] = df["ratio"].to_numpy(dtype="float64") / denom
    return df


def finalize(rows: list[dict], source: str) -> pd.DataFrame:
    """행 목록 → 스키마 순서를 맞춘 DataFrame + 재정규화."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=SCHEMA_COLS)
    df["source"] = source
    df = add_rescaled(df)
    for c in SCHEMA_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[SCHEMA_COLS].sort_values(["batch_id", "date"]).reset_index(drop=True)
    return df


# ── HTTP ──────────────────────────────────────────────────
def call(path: str, body: dict) -> dict:
    """데이터랩 POST 1회.

    도메인/헤더는 ``_common`` 의 ``NAVER_BASE`` / ``naver_headers()`` 만 쓴다.
    NAVER API HUB(NCP) 이관 시 그 두 곳만 바꾸면 되고 여기는 손댈 필요가 없다.
    ``post_json`` 은 4xx를 즉시 예외로 올린다 — 잘못된 cid나 한도 초과를
    무한 재시도로 태우지 않기 위해서다(일 1,000회 한도).
    """
    return post_json(f"{NAVER_BASE}{path}", naver_headers(), body)


# ── CLI 공통 ──────────────────────────────────────────────
def date_args(desc: str) -> argparse.ArgumentParser:
    """``--start`` / ``--end``. 인자가 없으면 어제 하루(일일 배치 모드)."""
    y = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--start", default=y, help="YYYY-MM-DD (기본: 어제)")
    p.add_argument("--end", default=y, help="YYYY-MM-DD (기본: 어제)")
    p.add_argument("--partition", default=None,
                   help="저장 파일명. 기본은 start의 YYYY-MM")
    return p


def partition_of(args: argparse.Namespace) -> str:
    return args.partition or args.start[:7]


def check_period(start: str, end: str) -> None:
    """기간 분할 금지 규칙을 코드에서도 명시한다.

    데이터랩은 긴 기간을 **한 번에** 요청하는 편이 정규화 일관성에 유리하다.
    기간을 쪼개면 배치마다 분모(그 기간의 최댓값)가 달라져 앵커 재정규화만으로는
    이어 붙일 수 없다.

    TODO: 3년 백필도 한 번에 요청하는 것이 원칙이지만, 만약 응답이 잘리거나
      타임아웃이 나서 부득이 분할해야 한다면 **30일 이상 겹치는 구간**을 두고
      겹침 구간의 (앞배치 rescaled / 뒷배치 rescaled) 중앙값을 보정계수로 삼아
      체이닝해야 한다. 겹침 없이 이어 붙이면 경계에서 계단이 생긴다.
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s > e:
        raise ValueError(f"start({start}) > end({end})")
    if s < pd.Timestamp("2016-01-01"):
        raise ValueError("데이터랩 보유 시작 이전 (쇼핑인사이트 2017-08, 검색어트렌드 2016-01)")
