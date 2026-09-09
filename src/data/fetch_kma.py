import io
import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ==============================================================================
# 1. 환경 설정 및 기상청 API 인증키 로드
# ==============================================================================
# .env 파일에 숨겨둔 KMA_AUTH_KEY 값을 가져옵니다.
load_dotenv()
AUTH_KEY = os.getenv("KMA_AUTH_KEY")

# 만약 .env 파일에 키가 없거나 이름이 틀렸다면 에러를 발생시켜 조기에 중단합니다.
if not AUTH_KEY:
    raise ValueError("⚠️ .env 파일에 KMA_AUTH_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# 기상청 API에서 날씨를 'WB01', 'WB03' 같은 코드로 주기 때문에, 이를 사람이 읽기 편한 한글로 변환하는 사전(Dictionary)입니다.
WEATHER_CODE_MAP = {
    'WB01': '맑음',
    'WB02': '구름조금',
    'WB03': '구름많음',
    'WB04': '흐림',
    'WB09': '비',
    'WB11': '비/눈',
    'WB12': '눈/비',
    'WB13': '눈',
    'WB14': '소나기'
}


# ==============================================================================
# 2. [기상청 API 1] 중기 기온예보 수집 함수 (서울 지점)
# ==============================================================================
def fetch_temperature_chunk(tmfc1: str, tmfc2: str) -> pd.DataFrame:
    """
    [역할] 지정된 기간 동안 서울(11B10101)에 발표된 예상 최저/최고 기온을 기상청에서 가져옵니다.
    - tmfc1: 수집 시작 일시 (예: '2023010106' -> 2023년 1월 1일 06시 발표)
    - tmfc2: 수집 종료 일시 (예: '2023063018' -> 2023년 6월 30일 18시 발표)
    """
    url = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wc.php"
    params = {
        "reg": "11B10101",     # 기상청 서울 지점 고유 코드
        "tmfc1": tmfc1,        # 조회 시작 시각
        "tmfc2": tmfc2,        # 조회 종료 시각
        "disp": "1",           # 1: 쉼표(,)로 구분된 CSV 텍스트 형태로 수신
        "help": "0",           # 0: 불필요한 도움말 텍스트 생략
        "authKey": AUTH_KEY    # .env에서 불러온 내 인증키
    }
    
    try:
        # 기상청 서버로 데이터 요청을 보냅니다 (20초 동안 응답이 없으면 타임아웃)
        res = requests.get(url, params=params, timeout=20)
        
        # 정상 응답(200)이 아니거나 기상청 정상 시작 태그(#START7777)가 없으면 빈 표 반환
        if res.status_code != 200 or "#START7777" not in res.text:
            return pd.DataFrame()
            
        # '#'으로 시작하는 주석/태그 줄을 걸러내고 순수 데이터 줄만 리스트로 만듭니다.
        lines = [line.strip() for line in res.text.split('\n') if not line.startswith('#') and line.strip()]
        if not lines:
            return pd.DataFrame()
            
        # 텍스트 줄들을 표(Pandas DataFrame) 형태로 변환합니다.
        df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=',', header=None, on_bad_lines='skip')
        
        # 기상청 기온 데이터 규격에서 필요한 열만 골라냅니다:
        # [열 1] tm_fc: 예보를 발표한 시각 (예: 202401010600)
        # [열 2] tm_ef: 예보가 적용되는 대상 날짜시각 (예: 202401040000)
        # [열 6] ta_min: 예상 최저기온 (℃)
        # [열 7] ta_max: 예상 최고기온 (℃)
        df = df.iloc[:, [1, 2, 6, 7]]
        df.columns = ['tm_fc', 'tm_ef', 'ta_min', 'ta_max']
        return df
        
    except Exception as e:
        print(f"❌ 기온 데이터 수집 중 오류 ({tmfc1}~{tmfc2}): {e}")
        return pd.DataFrame()


# ==============================================================================
# 3. [기상청 API 2] 중기 육상예보 수집 함수 (서울/경기 광역)
# ==============================================================================
def fetch_land_chunk(tmfc1: str, tmfc2: str) -> pd.DataFrame:
    """
    [역할] 서울/경기(11B00000) 지역의 날씨 상태(맑음/흐림 등)와 강수확률(0~100%)을 가져옵니다.
    """
    url = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php"
    params = {
        "reg": "11B00000",     # 서울·인천·경기도 광역 구역 코드
        "tmfc1": tmfc1,
        "tmfc2": tmfc2,
        "disp": "1",
        "help": "0",
        "authKey": AUTH_KEY
    }
    
    try:
        res = requests.get(url, params=params, timeout=20)
        if res.status_code != 200 or "#START7777" not in res.text:
            return pd.DataFrame()
            
        lines = [line.strip() for line in res.text.split('\n') if not line.startswith('#') and line.strip()]
        if not lines:
            return pd.DataFrame()
            
        df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=',', header=None, on_bad_lines='skip')
        
        # 육상예보 규격에 맞게 열 추출:
        # [열 1] tm_fc: 발표시각
        # [열 2] tm_ef: 발효시각
        # [열 6] weather_desc: 날씨 코드 (WB01, WB03 등)
        # [열 7] rn_st: 강수확률 (%)
        if df.shape[1] >= 8:
            df = df.iloc[:, [1, 2, 6, 7]]
        else:
            df = df.iloc[:, [1, 2, 6, 6]]
            
        df.columns = ['tm_fc', 'tm_ef', 'weather_desc', 'rn_st']
        
        # 'WB01' -> '맑음', 'WB03' -> '구름많음' 등 한글로 자동 변환
        df['weather_desc'] = df['weather_desc'].astype(str).str.strip().map(lambda x: WEATHER_CODE_MAP.get(x, x))
        return df
        
    except Exception as e:
        print(f"❌ 육상예보 데이터 수집 중 오류 ({tmfc1}~{tmfc2}): {e}")
        return pd.DataFrame()


# ==============================================================================
# 4. [기상청 API 3] 기상특보 수집 함수 (서울 지역)
# ==============================================================================
def fetch_warnings(start_ymd: str, end_ymd: str) -> pd.DataFrame:
    """
    [역할] 한파(C), 폭염(H), 호우(R), 대설(S) 등 기상청 특보 발효 이력을 가져옵니다.
    - start_ymd: 시작 연월일 (예: '20230101')
    - end_ymd: 종료 연월일 (예: '20261231')
    """
    url = "https://apihub.kma.go.kr/api/typ01/url/wrn_met_data.php"
    params = {
        "reg": "11B00000",     # 서울·경기 구역
        "wrn": "A",            # A: 모든 종류의 특보를 전부 조회
        "tmfc1": f"{start_ymd}0000",
        "tmfc2": f"{end_ymd}2359",
        "disp": "1",
        "help": "0",
        "authKey": AUTH_KEY
    }
    
    try:
        res = requests.get(url, params=params, timeout=20)
        lines = [line.strip() for line in res.text.split('\n') if not line.startswith('#') and line.strip()]
        if not lines:
            return pd.DataFrame()
            
        df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=',', header=None, on_bad_lines='skip')
        
        # [열 2] tm_ef: 특보 시작 일시
        # [열 3] tm_ed: 특보 종료 일시
        # [열 10] wrn_type: 특보 종류 (C: 한파, H: 폭염, R: 호우 등)
        # [열 11] wrn_level: 특보 수준 (주의보, 경보 등)
        df = df.iloc[:, [2, 3, 10, 11]]
        df.columns = ['tm_ef', 'tm_ed', 'wrn_type', 'wrn_level']
        return df
        
    except Exception as e:
        print(f"❌ 특보 데이터 수집 중 오류: {e}")
        return pd.DataFrame()


# ==============================================================================
# 5. [메인 파이프라인] 3개년치 백필 수집 및 일자별 정제 테이블 생성
# ==============================================================================
def run_backfill(start_year=2023, end_year=2026):
    """
    [전체 실행 순서]
    1. 2023~2026년 데이터를 6개월 단위로 쪼개어 기상청 API를 차례대로 호출
    2. 수집된 원본을 data/01_raw 폴더에 안전하게 백업
    3. 일자(Daily)별로 하루 최저/최고기온, 날씨, 강수확률을 하나로 묶음(Groupby & Merge)
    4. 패션 분석에 필요한 5대 파생변수(일교차, 전일대비기온변화, 비예보, 특보여부 등) 계산
    5. data/02_intermediate/daily_weather.parquet 파일로 깔끔하게 저장
    """
    
    # 데이터를 저장할 폴더가 없으면 자동으로 생성합니다.
    os.makedirs("data/01_raw", exist_ok=True)
    os.makedirs("data/02_intermediate", exist_ok=True)
    
    print("🚀 [Step 1] 기상 데이터 3개년 백필(Backfill) 수집을 시작합니다...")
    
    # 3년 치를 한 번에 부르면 기상청 서버가 멈추므로, 6개월(반기) 단위 구간 리스트를 만듭니다.
    periods = []
    for y in range(start_year, end_year + 1):
        periods.append((f"{y}0101", f"{y}0630"))  # 상반기: 1월 1일 ~ 6월 30일
        periods.append((f"{y}0701", f"{y}1231"))  # 하반기: 7월 1일 ~ 12월 31일
    
    temp_list = []  # 기온 데이터들을 모아둘 리스트
    land_list = []  # 육상예보(날씨/강수) 데이터들을 모아둘 리스트
    
    # 구간별로 순회하며 API 호출
    for s_date, e_date in periods:
        print(f"  📥 기상청 데이터 수집 중: {s_date} ~ {e_date}...")
        df_t = fetch_temperature_chunk(f"{s_date}06", f"{e_date}18")
        df_l = fetch_land_chunk(f"{s_date}06", f"{e_date}18")
        
        if not df_t.empty:
            temp_list.append(df_t)
        if not df_l.empty:
            land_list.append(df_l)
            
        time.sleep(0.3)  # 기상청 서버에 부담을 주지 않도록 0.3초 대기
        
    print("  📥 기상특보(한파/폭염 등) 데이터 수집 중...")
    df_wrn = fetch_warnings(f"{start_year}0101", f"{end_year}1231")
    
    if not temp_list:
        print("❌ 수집된 기온 데이터가 없습니다. API 키나 인터넷 연결을 확인하세요.")
        return
        
    # 조각조각 수집된 데이터들을 하나의 큰 표로 이어 붙입니다.
    df_temp_raw = pd.concat(temp_list, ignore_index=True)
    df_land_raw = pd.concat(land_list, ignore_index=True) if land_list else pd.DataFrame()
    
    # [01_raw] 원본 기온 데이터 백업 저장
    df_temp_raw.to_csv("data/01_raw/raw_temp.csv", index=False)
    print("  ✅ [data/01_raw/raw_temp.csv] 원본 백업 저장 완료")

    # --------------------------------------------------------------------------
    # [Step 2] 일자(Daily) 기준 표준 테이블 정제
    # --------------------------------------------------------------------------
    print("⚙️ [Step 2] 일자별 표준 테이블로 그룹화 및 결합 중...")
    
    # 14자리 시각 문자열('202401010600')에서 앞 8자리('20240101')만 잘라 날짜(YYYY-MM-DD) 형식으로 변환
    df_temp_raw['date'] = pd.to_datetime(df_temp_raw['tm_ef'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    df_temp_raw['ta_min'] = pd.to_numeric(df_temp_raw['ta_min'], errors='coerce')
    df_temp_raw['ta_max'] = pd.to_numeric(df_temp_raw['ta_max'], errors='coerce')
    
    # 하루에도 여러 번 예보가 발표되므로, 같은 날짜(date) 기준으로 가장 낮은 최저기온과 가장 높은 최고기온을 뽑습니다.
    daily = df_temp_raw.groupby('date').agg({'ta_min': 'min', 'ta_max': 'max'}).reset_index()
    
    # 육상예보(강수확률, 날씨 상태) 결합
    if not df_land_raw.empty:
        df_land_raw['date'] = pd.to_datetime(df_land_raw['tm_ef'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        df_land_raw['rn_st'] = pd.to_numeric(df_land_raw['rn_st'], errors='coerce').fillna(0)
        
        # 같은 날짜 중 가장 높은 강수확률과 대표 날씨 텍스트를 추출
        daily_land = df_land_raw.groupby('date').agg({'rn_st': 'max', 'weather_desc': 'first'}).reset_index()
        # 기온 테이블과 날씨 테이블을 날짜(date) 기준으로 좌측 조인(Left Join)
        daily = pd.merge(daily, daily_land, on='date', how='left')
    else:
        daily['rn_st'] = 0.0
        daily['weather_desc'] = '맑음'
        
    # 빈 값(결측치) 안전 처리
    daily['rn_st'] = daily['rn_st'].fillna(0.0)
    daily['weather_desc'] = daily['weather_desc'].fillna('맑음')
    
    # --------------------------------------------------------------------------
    # [Step 3] 패션 수요 분석용 핵심 파생변수(Feature Engineering) 생성
    # --------------------------------------------------------------------------
    print("🧠 [Step 3] 패션 수요 예측용 파생변수 계산 중...")
    
    # 1. 일교차: 최고기온 - 최저기온 (10도 이상 벌어지면 자켓/가디건 등 간절기 옷 수요 발생)
    daily['temp_diff'] = daily['ta_max'] - daily['ta_min']
    
    # 2. 전일 대비 기온 변화: 오늘 최고기온 - 어제 최고기온 (갑자기 추워질 때 소비자가 옷을 삼)
    daily['temp_delta_prev'] = daily['ta_max'] - daily['ta_max'].shift(1)
    
    # 3. 비 예보 여부 (0 또는 1): 강수확률이 60% 이상이거나 날씨 설명에 비/눈/소나기가 있으면 1
    is_rain_desc = daily['weather_desc'].str.contains('비|소나기|눈', na=False)
    is_rain_prob = daily['rn_st'] >= 60
    daily['has_rain'] = (is_rain_desc | is_rain_prob).astype(int)
    
    # 4. 기상특보 매핑: 특보 발효 기간(시작일~종료일) 사이에 해당하는 날짜에 라벨링
    daily['is_warning'] = 0
    daily['warning_type'] = 'NONE'
    
    if not df_wrn.empty:
        for _, row in df_wrn.iterrows():
            st = pd.to_datetime(str(row['tm_ef'])[:8], format='%Y%m%d', errors='coerce')
            ed = pd.to_datetime(str(row['tm_ed'])[:8], format='%Y%m%d', errors='coerce') if pd.notnull(row['tm_ed']) else st
            if pd.notnull(st) and pd.notnull(ed):
                # 특보 기간에 해당하는 날짜 행을 찾아서 마킹
                mask = (daily['date'] >= st) & (daily['date'] <= ed)
                daily.loc[mask, 'is_warning'] = 1
                daily.loc[mask, 'warning_type'] = str(row['wrn_type'])
                
    # --------------------------------------------------------------------------
    # [Step 4] 최종 정제 데이터 저장 (Parquet 포맷)
    # --------------------------------------------------------------------------
    out_path = "data/02_intermediate/daily_weather.parquet"
    # 날짜 순서대로 정렬하여 가볍고 빠른 Parquet 형식으로 저장합니다.
    daily.sort_values('date').reset_index(drop=True).to_parquet(out_path, index=False)
    
    print(f"\n🎉 [수집 및 정제 완료!] 저장 위치: {out_path}")
    print(f"📊 총 수집된 일수: {len(daily)}일")
    print("\n[상위 5건 데이터 미리보기]")
    print(daily.head())


# 이 파이썬 파일이 직접 실행될 때 run_backfill 함수를 작동시킵니다.
if __name__ == "__main__":
    run_backfill(start_year=2023, end_year=2026)