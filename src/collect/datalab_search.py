"""T4. 네이버 데이터랩 — 검색어트렌드 (보조 타깃 / 선행지표).

``POST https://openapi.naver.com/v1/datalab/search``
키워드 그룹 **최대 5개** → 앵커 1 + 대상 4. (그룹당 키워드는 최대 20개)

T1·T3(쇼핑 클릭)은 **구매 직전** 신호, 이쪽은 **관심 단계** 신호다.
둘의 시차를 보는 것 자체가 Q4(정량·정성 트렌드 선행 관계)의 재료가 된다.

T3와 달리 **카테고리 개념이 없다.** 쇼핑 밖 검색 전체가 모집단이라
전역 앵커(``anchor.keyword``) 하나로 모든 배치를 하나의 축에 올릴 수 있다.
카테고리를 가로지르는 비교가 필요할 때 이 데이터셋을 쓰는 이유다.

사용법::

    python -m src.collect.datalab_search --start 2026-07-01 --end 2026-07-31
    python -m src.collect.datalab_search --groups rain,outerwear_winter
    python -m src.collect.datalab_search --dry-run

저장: ``data/raw/datalab_search/{YYYY-MM}.csv``
"""
from __future__ import annotations

import datetime as dt

from ._common import write_csv
from ._datalab import (SCHEMA_COLS, call, chunk_with_anchor, check_period,
                       date_args, finalize, iter_results, load_categories,
                       partition_of)

PATH = "/v1/datalab/search"
SOURCE = "naver_datalab_search"      # shopping_trend_daily.source 컬럼 값
DATASET = "datalab_search"          # data/raw/ 아래 디렉토리명 (docs/datasets.md §5)
MAX_GROUPS = 5               # API 상한. 1슬롯은 앵커 고정 → 배치당 신규 키워드 4개


def build_batches(cfg: dict, groups: list[str] | None) -> list[list[str]]:
    """전역 앵커 1개 + 키워드 4개씩.

    그룹당 키워드를 1개만 넣는 1:1 매핑을 쓴다(``groupName == keywords[0]``).
    유의어를 묶으면(예: 패딩+롱패딩) 지수가 합산돼 개별 키워드의 날씨 반응을
    분리할 수 없게 된다. 묶는 판단은 EDA 후에 해도 늦지 않다.
    """
    ws = cfg["weather_sensitive"]
    names = groups or list(ws)
    unknown = [g for g in names if g not in ws]
    if unknown:
        raise SystemExit(f"알 수 없는 그룹: {unknown}\n선택 가능: {list(ws)}")

    kws: list[str] = []
    for g in names:
        for kw in ws[g]["items"]:
            if kw not in kws:
                kws.append(kw)

    return chunk_with_anchor(kws, cfg["anchor"]["keyword"], MAX_GROUPS)


def collect(start: str, end: str, *, groups: list[str] | None = None,
            time_unit: str = "date",
            dry_run: bool = False) -> tuple[list[dict], list[dict]]:
    check_period(start, end)
    cfg = load_categories()
    anchor = cfg["anchor"]["keyword"]
    batches = build_batches(cfg, groups)

    rows: list[dict] = []
    log: list[dict] = []

    for i, chunk in enumerate(batches, 1):
        bid = f"srch_b{i:02d}"
        log.append({"batch_id": bid, "anchor_keyword": anchor, "groups": chunk})
        print(f"[{bid}] anchor={anchor} {chunk}")
        if dry_run:
            continue

        payload = call(PATH, {
            "startDate": start, "endDate": end, "timeUnit": time_unit,
            "keywordGroups": [{"groupName": k, "keywords": [k]} for k in chunk],
        })

        for title, data in iter_results(payload):
            if not data:
                print(f"  ! '{title}' 데이터 0건")
            for d in data:
                rows.append({
                    "date": d["period"],
                    "category_l1": None,
                    "category_l2": None,
                    "keyword": title,
                    "ratio": float(d["ratio"]),
                    "batch_id": bid,
                    "cid": None,
                    "is_anchor": title == anchor,
                    "gender": None,
                    "age_group": None,
                })
    return rows, log


def main() -> None:
    p = date_args(__doc__.splitlines()[0])
    p.add_argument("--groups", default=None,
                   help="수집할 weather_sensitive 그룹 (쉼표 구분). 기본: 전체")
    p.add_argument("--time-unit", default="date", choices=["date", "week", "month"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    groups = [g.strip() for g in args.groups.split(",")] if args.groups else None
    rows, log = collect(args.start, args.end, groups=groups,
                        time_unit=args.time_unit, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n총 {len(log)}회 호출 예정")
        return

    df = finalize(rows, SOURCE)
    path = write_csv(df, DATASET, partition_of(args))
    print(f"\n{path}  rows={len(df)}  calls={len(log)}")


if __name__ == "__main__":
    main()
