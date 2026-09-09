import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. API 접속 환경 설정
# ==========================================
CLIENT_ID = os.getenv("NAVER_API_HUB_ACCESS_KEY_ID") or os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_API_HUB_SECRET_KEY") or os.getenv("NAVER_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(".env 파일에서 네이버 인증키를 확인해주세요.")

ENDPOINT = "https://naverapihub.apigw.ntruss.com/shopping/v1/category/keywords"
HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": CLIENT_ID.strip(),
    "X-NCP-APIGW-API-KEY": CLIENT_SECRET.strip(),
    "Content-Type": "application/json"
}

CATEGORY_ID = "50000000"  # 패션의류 대분류
SOURCE_NAME = "naver_datalab_keyword"
START_DATE = "2023-01-01"
END_DATE = "2026-07-31"

# ==========================================
# 2. 앵커 키워드 및 날씨 민감 타깃 키워드 정의 (param은 1개만 허용)
# ==========================================
ANCHOR_KEYWORD = {"name": "패션_앵커", "param": ["원피스"]}  # 사계절 스테디셀러 기준 키워드

TARGET_KEYWORDS = [
    # 겨울/한파 민감 아이템
    {"name": "패딩", "param": ["패딩"]},
    {"name": "코트", "param": ["코트"]},
    {"name": "목도리", "param": ["목도리"]},
    {"name": "히트텍", "param": ["히트텍"]}, 
    # 봄/가을 환절기 아이템
    {"name": "가디건", "param": ["가디건"]},
    {"name": "자켓", "param": ["자켓"]},
    # 여름/폭염/장마 아이템
    {"name": "반팔", "param": ["반팔"]},
    {"name": "레인부츠", "param": ["레인부츠"]},
]

def call_keyword_api(kw_batch: list[dict], batch_id: str) -> list[dict]:
    """[앵커 1개 + 타깃 키워드 최대 4개] 조합으로 3개년 전체를 호출"""
    payload = {
        "startDate": START_DATE,
        "endDate": END_DATE,
        "timeUnit": "date",
        "category": CATEGORY_ID,
        "keyword": [ANCHOR_KEYWORD] + kw_batch,
        "device": "",
        "gender": "",
        "ages": []
    }
    
    batch_names = [k["name"] for k in kw_batch]
    print(f"[{batch_id}] 호출 진행: [앵커] + {batch_names}")
    
    res = requests.post(ENDPOINT, headers=HEADERS, data=json.dumps(payload))
    if res.status_code != 200:
        print(f"  └ [에러 {res.status_code}]: {res.text}")
        return []
        
    res_json = res.json()
    batch_rows = []
    
    for item in res_json.get("results", []):
        kw_title = item["title"]
        for d in item.get("data", []):
            batch_rows.append({
                "date": d["period"],
                "source": SOURCE_NAME,
                "category_l1": "패션의류",
                "category_l2": None,
                "keyword": kw_title,
                "ratio": float(d["ratio"]),
                "batch_id": batch_id,
                "gender": None,
                "age_group": None,
                "cid": CATEGORY_ID,
                "is_anchor": (kw_title == ANCHOR_KEYWORD["name"])
            })
            
    return batch_rows

def main():
    print(f"=== T3. 쇼핑인사이트 키워드별 클릭 트렌드 수집 ({START_DATE} ~ {END_DATE}) ===")
    
    # 1회 최대 5개이므로 [앵커 1 + 타깃 4] 단위로 묶음 처리
    batch_size = 4
    all_raw_rows = []
    
    for i in range(0, len(TARGET_KEYWORDS), batch_size):
        b_idx = (i // batch_size) + 1
        batch_id = f"kw_b{b_idx:02d}"
        batch = TARGET_KEYWORDS[i:i + batch_size]
        
        rows = call_keyword_api(batch, batch_id)
        all_raw_rows.extend(rows)
        time.sleep(0.5)  # 호출 간격 유지
        
    if not all_raw_rows:
        print("[실패] 수집된 데이터가 없습니다.")
        return

    # ==========================================
    # 3. 앵커 정규화 (Rescaling) 및 표준 스키마 정렬
    # ==========================================
    raw_df = pd.DataFrame(all_raw_rows)
    
    # 각 배치별 앵커 추출 (배치마다 스케일 기준이 다르므로 batch_id 단위로 조인)
    anchors = raw_df[raw_df["is_anchor"]][["date", "batch_id", "ratio"]].rename(columns={"ratio": "anchor_ratio"})
    
    # 원본 데이터에 해당 배치의 앵커 ratio 결합
    merged = pd.merge(raw_df, anchors, on=["date", "batch_id"], how="left")
    merged["rescaled"] = merged["ratio"] / merged["anchor_ratio"]
    
    # 멘토님 12개 컬럼 표준 순서 그대로 정렬
    schema_cols = [
        "date", "source", "category_l1", "category_l2", "keyword",
        "ratio", "rescaled", "batch_id", "gender", "age_group", "cid", "is_anchor"
    ]
    final_df = merged[schema_cols].sort_values(by=["date", "batch_id", "keyword"]).reset_index(drop=True)
    
    # 4. 저장
    os.makedirs("data/raw/datalab_keyword", exist_ok=True)
    out_path = "data/raw/datalab_keyword/all_years.csv"
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    print(f"\n[성공] T3 데이터셋 수집 및 정규화 완료 -> {out_path}")
    print(f"총 행 수: {len(final_df)}행")
    print(final_df.head(10))

if __name__ == "__main__":
    main()