import os
import io
import time
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# 1. 환경 설정 및 API 키 로드
load_dotenv()
AUTH_KEY = os.getenv("KMA_APIHUB_AUTH_KEY")

if not AUTH_KEY:
    raise ValueError(".env 파일에 KMA_APIHUB_AUTH_KEY가 없습니다.")

def fetch_w1_asos_monthly(start_date: str, end_date: str) -> pd.DataFrame:
    """한 달(최대 31일) 단위로 W1(지상관측) 데이터를 수집하는 함수"""
    
    # 1. 인증키(.env) 끝에 숨어있을 수 있는 공백이나 줄바꿈(엔터) 완벽 제거
    auth = AUTH_KEY.strip()
    
    # 2. requests.get(params=...) 의 자동 인코딩이 기상청 서버와 충돌하는 것을 방지!
    # 실무 기상청 국룰인 'URL 직접 쌩으로 조립' 방식으로 변경 (스펙에서 사라진 disp=1 도 제거)
    req_url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php?tm1={start_date}&tm2={end_date}&stn=108&help=0&authKey={auth}"
    
    try:
        # params=params 옵션을 빼버리고 조립된 URL을 그대로 꽂아넣습니다.
        res = requests.get(req_url, timeout=20)
        
        if res.status_code != 200:
            print(f"❌ HTTP 에러: {res.status_code}")
            return pd.DataFrame()
            
        if "#START7777" not in res.text:
            print(f"\n⚠️ 기상청 거절 메시지: {res.text.strip()[:100]}...") # 길면 짤라서 출력
            return pd.DataFrame()
            
        # 정상 처리: 주석(#) 떼어내고 순수 데이터만 추출
        lines = [line.strip() for line in res.text.split('\n') if not line.startswith('#') and line.strip()]
        if not lines:
            return pd.DataFrame()
            
        # 공백 기준으로 컬럼 분리
        # 공백 기준으로 컬럼 분리
        df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=r'\s+', header=None, on_bad_lines='skip')
        return df
        
    except Exception as e:
        print(f"❌ W1 수집 오류 ({start_date}~{end_date}): {e}")
        return pd.DataFrame()

def run_w1_backfill(start_ymd="20230101", end_ymd="20260731"):
    print("[W1 지상관측] 3개년 백필 수집을 시작합니다...")
    
    os.makedirs("data/01_raw", exist_ok=True)
    
    start_dt = datetime.strptime(start_ymd, "%Y%m%d")
    end_dt = datetime.strptime(end_ymd, "%Y%m%d")
    
    current_dt = start_dt
    all_data = []
    
    # 한 달씩 쪼개서 호출 (최대 31일 제한 회피)
    while current_dt <= end_dt:
        # 이번 달의 마지막 날 계산
        next_month_first = (current_dt + relativedelta(months=1)).replace(day=1)
        chunk_end_dt = next_month_first - relativedelta(days=1)
        
        if chunk_end_dt > end_dt:
            chunk_end_dt = end_dt
            
        s_str = current_dt.strftime("%Y%m%d")
        e_str = chunk_end_dt.strftime("%Y%m%d")
        
        print(f"수집 중: {s_str} ~ {e_str}")
        df_chunk = fetch_w1_asos_monthly(s_str, e_str)
        
        if not df_chunk.empty:
            all_data.append(df_chunk)
            
        current_dt = next_month_first
        time.sleep(0.3)  # 서버 부하 방지
        
    if not all_data:
        print("수집된 W1 데이터가 없습니다.")
        return
        
    df_final = pd.concat(all_data, ignore_index=True)
    

    out_path = "data/01_raw/raw_w1_asos.csv"
    df_final.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    print(f"\n[W1 수집 완료!] 저장 위치: {out_path}")
    print(f"총 수집된 일수: {len(df_final)}행")

if __name__ == "__main__":
    run_w1_backfill()