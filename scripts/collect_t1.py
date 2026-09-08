import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. API 접속 환경 설정 (API HUB 기준)
# ==========================================
CLIENT_ID = os.getenv("NAVER_API_HUB_ACCESS_KEY_ID") or os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_API_HUB_SECRET_KEY") or os.getenv("NAVER_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(".env 파일에서 네이버 인증키를 확인해주세요.")

# 정의서 상의 엔드포인트 분리 원칙 적용
BASE_URL = "https://naverapihub.apigw.ntruss.com"
ENDPOINT = f"{BASE_URL}/shopping/v1/categories"

HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": CLIENT_ID.strip(),
    "X-NCP-APIGW-API-KEY": CLIENT_SECRET.strip(),
    "Content-Type": "application/json"
}

# ==========================================
# 2. 카테고리 정의 및 앵커 전략
# ==========================================
# 앵커: 모든 비교의 기준선이 되는 대분류 (슬롯 1 고정)
ANCHOR = {"name": "패션의류_앵커", "param": ["50000000"]}

# 비교 대상 하위/대분류 카테고리 (슬롯 2, 3으로 2개씩 묶어서 호출)
TARGET_CATEGORIES = [
    {"name": "여성의류", "param": ["50000167"]},
    {"name": "남성의류", "param": ["50000169"]},
    {"name": "패션잡화", "param": ["50000001"]},
    {"name": "스포츠_레저", "param": ["50000007"]},
]

START_DATE = "2023-01-01"
END_DATE = "2026-07-31"

def call_datalab(cat_batch: list[dict]) -> pd.DataFrame:
    """[앵커 + 대상 카테고리 2개] 묶음으로 3년 전체를 한 번에 조회"""
    payload = {
        "startDate": START_DATE,
        "endDate": END_DATE,
        "timeUnit": "date",
        "category": [ANCHOR] + cat_batch,
        "device": "",
        "gender": "",
        "ages": []
    }
    
    cat_names = [c["name"] for c in cat_batch]
    print(f"호출 진행: 앵커 + {cat_names}")
    
    res = requests.post(ENDPOINT, headers=HEADERS, data=json.dumps(payload))
    if res.status_code != 200:
        print(f"  └ [에러 {res.status_code}]: {res.text}")
        return pd.DataFrame()
        
    res_json = res.json()
    rows = []
    for group in res_json.get("results", []):
        name = group["title"]
        for item in group.get("data", []):
            rows.append({
                "date": item["period"],
                "category": name,
                "ratio": float(item["ratio"])
            })
            
    return pd.DataFrame(rows)

def main():
    print(f"=== T1. 쇼핑인사이트 카테고리 트렌드 백필 ({START_DATE} ~ {END_DATE}) ===")
    
    # 2개씩 슬라이싱하여 앵커와 함께 호출
    batch_size = 2
    collected_dfs = []
    
    for i in range(0, len(TARGET_CATEGORIES), batch_size):
        batch = TARGET_CATEGORIES[i:i + batch_size]
        sub_df = call_datalab(batch)
        if not sub_df.empty:
            collected_dfs.append(sub_df)
        time.sleep(0.5)  # RPS 속도 제어
        
    if not collected_dfs:
        print("[실패] 수집된 데이터가 없습니다.")
        return

    # ==========================================
    # 3. 앵커 정규화 (Rescaling) 처리
    # ==========================================
    full_df = pd.concat(collected_dfs, ignore_index=True)
    
    # 앵커 데이터와 타깃 데이터 분리
    anchor_df = full_df[full_df["category"] == ANCHOR["name"]][["date", "ratio"]].drop_duplicates(subset=["date"])
    anchor_df = anchor_df.rename(columns={"ratio": "anchor_ratio"})
    
    targets_df = full_df[full_df["category"] != ANCHOR["name"]].copy()
    
    # 같은 날짜 기준으로 앵커 ratio와 병합 후 재스케일링
    merged = pd.merge(targets_df, anchor_df, on="date", how="left")
    merged["rescaled"] = merged["ratio"] / merged["anchor_ratio"]
    
    # 앵커 자체도 동일 기준(rescaled = 1.0)으로 데이터셋에 포함
    anchor_self = anchor_df.copy()
    anchor_self["category"] = ANCHOR["name"]
    anchor_self["rescaled"] = 1.0
    anchor_self = anchor_self.rename(columns={"anchor_ratio": "ratio"})
    
    final_df = pd.concat([anchor_self[["date", "category", "ratio", "rescaled"]],
                          merged[["date", "category", "ratio", "rescaled"]]], ignore_index=True)
    
    final_df = final_df.sort_values(by=["date", "category"]).reset_index(drop=True)
    
    # 4. 결과 저장
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/t1_shopping_category.csv"
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n[성공] T1 데이터셋 수집 및 정규화 완료 -> {output_path}")
    print(f"총 행 수: {len(final_df)}행")
    print(final_df.head(10))

if __name__ == "__main__":
    main()