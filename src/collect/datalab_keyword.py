"""T3. 네이버 데이터랩 쇼핑인사이트 — 카테고리 내 키워드별 클릭지수 (세분 타깃).

``POST https://openapi.naver.com/v1/datalab/shopping/category/keywords``
요청당 키워드 **최대 5개** → 앵커 1 + 대상 4.

중분류(T1)는 뭉뚱그려져 날씨 효과가 희석된다. **패딩·우산·선크림 같은 실제 신호는 여기 있다.**

T1과 다른 점 하나: ratio가 **"그 카테고리 안에서의"** 상대값이다.
따라서 요청은 (카테고리 1개 × 키워드 5개) 단위이고, 앵커 키워드도 그 카테고리에서
클릭이 잡히는 것이어야 한다. 화장품 카테고리에서 '반팔티'를 앵커로 쓰면 ratio가 0이라
재정규화가 통째로 NaN이 된다. → ``configs/categories.yaml`` 의 ``keyword_anchor_by_cid``.

그 결과 **서로 다른 cid의 배치는 rescaled끼리도 직접 비교할 수 없다.**
앵커가 다르기 때문이다. 카테고리를 가로지르는 비교가 필요하면 T4(검색어트렌드)를 쓴다.

사용법::

    python -m src.collect.datalab_keyword --start 2026-07-01 --end 2026-07-31
    python -m src.collect.datalab_keyword --groups rain,top_summer
    python -m src.collect.datalab_keyword --dry-run

저장: ``data/raw/datalab_keyword/{YYYY-MM}.csv``
"""
from __future__ import annotations

import datetime as dt

from ._common import write_csv
from ._datalab import (SCHEMA_COLS, call, chunk_with_anchor, check_period,
                       date_args, finalize, iter_results, keyword_anchor,
                       load_categories, partition_of)

PATH = "/v1/datalab/shopping/category/keywords"
SOURCE = "naver_datalab_keyword"      # shopping_trend_daily.source 컬럼 값
DATASET = "datalab_keyword"          # data/raw/ 아래 디렉토리명 (docs/datasets.md §5)
MAX_PER_REQUEST = 5          # API 상한. 1슬롯은 앵커 고정 → 배치당 신규 키워드 4개

# 대분류 cid → 이름 (category_l1 컬럼 채우기용)
L1_NAME = {
    "50000000": "패션의류",
    "50000001": "패션잡화",
    "50000002": "화장품/미용",
    "50000007": "스포츠/레저",
}


def build_batches(cfg: dict, groups: list[str] | None) -> list[dict]:
    """cid별로 키워드를 모아 [앵커, k1..k4] 배치를 만든다.

    같은 cid의 여러 그룹은 **합쳐서** 배치를 만든다. 그래야 슬롯이 낭비되지 않고
    호출 수가 준다(일 1,000회 한도). 예: top_summer(3) + bottom_summer(2) = 1회.
    """
    ws = cfg["weather_sensitive"]
    names = groups or list(ws)
    unknown = [g for g in names if g not in ws]
    if unknown:
        raise SystemExit(f"알 수 없는 그룹: {unknown}\n선택 가능: {list(ws)}")

    # cid → 키워드 (순서 유지 + 중복 제거)
    by_cid: dict[str, list[str]] = {}
    origin: dict[str, list[str]] = {}
    for g in names:
        cid = str(ws[g]["cid"])
        bucket = by_cid.setdefault(cid, [])
        for kw in ws[g]["items"]:
            if kw not in bucket:
                bucket.append(kw)
        origin.setdefault(cid, []).append(g)

    batches: list[dict] = []
    for cid, kws in by_cid.items():
        anchor = keyword_anchor(cfg, cid)
        for chunk in chunk_with_anchor(kws, anchor, MAX_PER_REQUEST):
            batches.append({
                "cid": cid,
                "anchor": anchor,
                "keywords": chunk,
                "groups": origin[cid],
            })
    return batches


def collect(start: str, end: str, *, groups: list[str] | None = None,
            time_unit: str = "date",
            dry_run: bool = False) -> tuple[list[dict], list[dict]]:
    check_period(start, end)
    cfg = load_categories()
    batches = build_batches(cfg, groups)

    rows: list[dict] = []
    log: list[dict] = []

    for i, b in enumerate(batches, 1):
        bid = f"kw_b{i:02d}"
        log.append({"batch_id": bid, "cid": b["cid"],
                    "category_l1": L1_NAME.get(b["cid"], b["cid"]),
                    "anchor_keyword": b["anchor"],
                    "keywords": b["keywords"],
                    "from_groups": b["groups"]})
        print(f"[{bid}] cid={b['cid']} anchor={b['anchor']} {b['keywords']}")
        if dry_run:
            continue

        payload = call(PATH, {
            "startDate": start, "endDate": end, "timeUnit": time_unit,
            "category": b["cid"],
            "keyword": [{"name": k, "param": [k]} for k in b["keywords"]],
        })

        for title, data in iter_results(payload):
            if not data:
                print(f"  ! '{title}' 데이터 0건 (cid={b['cid']} 안에서 클릭 없음)")
            for d in data:
                rows.append({
                    "date": d["period"],
                    "category_l1": L1_NAME.get(b["cid"], b["cid"]),
                    "category_l2": None,
                    "keyword": title,
                    "ratio": float(d["ratio"]),
                    "batch_id": bid,
                    "cid": b["cid"],
                    "is_anchor": title == b["anchor"],
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
