"""T1. 네이버 데이터랩 쇼핑인사이트 — 카테고리별 클릭지수 (메인 타깃).

``POST https://openapi.naver.com/v1/datalab/shopping/categories``
요청당 카테고리 **최대 3개** → 앵커 1 + 대상 2.

사용법::

    python -m src.collect.datalab_category --start 2026-07-01 --end 2026-07-31
    python -m src.collect.datalab_category                 # 어제 하루
    python -m src.collect.datalab_category --dry-run       # 호출 없이 배치 구성만 확인

저장: ``data/raw/datalab_category/{YYYY-MM}.parquet``

주의 — 기간을 쪼개지 마라. 3년 백필도 ``--start 2023-01-01 --end 2026-07-31`` 한 번이다.
이유는 ``_datalab.check_period`` 주석 참조.
"""
from __future__ import annotations

import datetime as dt

from ._common import write_parquet
from ._datalab import (SCHEMA_COLS, call, chunk_with_anchor, check_period,
                       date_args, finalize, iter_results, load_categories,
                       partition_of)

PATH = "/v1/datalab/shopping/categories"
SOURCE = "naver_datalab_category"      # shopping_trend_daily.source 컬럼 값
DATASET = "datalab_category"          # data/raw/ 아래 디렉토리명 (docs/datasets.md §5)
MAX_PER_REQUEST = 3          # API 상한. 1슬롯은 앵커 고정 → 배치당 신규 대상 2개


def build_batches(cfg: dict) -> list[list[dict]]:
    """[앵커, A, B] 형태의 배치 목록.

    cid가 아직 TBD인 항목은 **건너뛴다.** 잘못된 cid로 호출하면 에러가 아니라
    빈 결과가 돌아올 수 있어서, 틀린 데이터가 조용히 쌓이는 게 가장 위험하다.
    """
    a = cfg["anchor"]["category"]
    anchor = {"l1": a["name"], "l2": None, "cid": str(a["cid"])}

    targets = [
        {"l1": c["l1"], "l2": c["l2"], "cid": str(c["cid"])}
        for c in cfg["mid_categories"]
        if str(c.get("cid", "TBD")).upper() != "TBD"
    ]
    skipped = [c["l2"] for c in cfg["mid_categories"]
               if str(c.get("cid", "TBD")).upper() == "TBD"]
    if skipped:
        print(f"[skip] cid 미확정으로 제외: {', '.join(skipped)}")

    return chunk_with_anchor(targets, anchor, MAX_PER_REQUEST, key=lambda c: c["cid"])


def collect(start: str, end: str, *, time_unit: str = "date",
            dry_run: bool = False) -> tuple[list[dict], list[dict]]:
    """배치를 순회하며 수집. (행 목록, 배치 기록) 반환."""
    check_period(start, end)
    cfg = load_categories()
    batches = build_batches(cfg)
    anchor_cid = str(cfg["anchor"]["category"]["cid"])

    rows: list[dict] = []
    log: list[dict] = []

    for i, batch in enumerate(batches, 1):
        bid = f"cat_b{i:02d}"
        names = [c["l2"] or c["l1"] for c in batch]
        log.append({"batch_id": bid, "members": names,
                    "cids": [c["cid"] for c in batch]})
        print(f"[{bid}] {names}")
        if dry_run:
            continue

        payload = call(PATH, {
            "startDate": start, "endDate": end, "timeUnit": time_unit,
            "category": [{"name": c["l2"] or c["l1"], "param": [c["cid"]]}
                         for c in batch],
        })

        # 응답 title 로 요청 대상을 되찾는다 (요청 시 name을 그대로 돌려준다)
        by_name = {c["l2"] or c["l1"]: c for c in batch}
        for title, data in iter_results(payload):
            c = by_name.get(title)
            if c is None:                     # 방어: 예상 밖의 title
                print(f"  ! 알 수 없는 title: {title}")
                continue
            if not data:
                print(f"  ! {title}(cid={c['cid']}) 데이터 0건 — cid 유효성 의심")
            for d in data:
                rows.append({
                    "date": d["period"],
                    "category_l1": c["l1"],
                    "category_l2": c["l2"],
                    "keyword": None,
                    "ratio": float(d["ratio"]),
                    "batch_id": bid,
                    "cid": c["cid"],
                    "is_anchor": c["cid"] == anchor_cid,
                    "gender": None,
                    "age_group": None,
                })
    return rows, log


def main() -> None:
    p = date_args(__doc__.splitlines()[0])
    p.add_argument("--time-unit", default="date", choices=["date", "week", "month"])
    p.add_argument("--dry-run", action="store_true", help="API 호출 없이 배치만 출력")
    args = p.parse_args()

    rows, log = collect(args.start, args.end,
                        time_unit=args.time_unit, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n총 {len(log)}회 호출 예정")
        return

    df = finalize(rows, SOURCE)
    part = partition_of(args)
    path = write_parquet(df, DATASET, part)
    print(f"\n{path}  rows={len(df)}  calls={len(log)}")


if __name__ == "__main__":
    main()
