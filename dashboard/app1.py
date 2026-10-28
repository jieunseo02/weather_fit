# dashboard/app.py
# [목적]"사이드바 큰 틀 → 그 안에 노트북 코드를 옮겨 read_csv(원본)부터 실행"으로 구성한 Streamlit 리포트.
#   - 1 · EDA: 원본을 읽어 기상 관측·특보·예보·쇼핑 구조를 단계별로 확인 (기상 관측은 함수로 분리)
#   - 1-2 · 분석 과정: 노트북 순서 그대로 예보 병합·오차, 특보 구간 조립, 한파·폭염 채점을 앱에서 직접 실행
#   - 2 · 시각화: 노트북에서 계산해 저장한 결과(data/dashboard/*.csv)를 읽어 차트로 보여 줌
#   - 모델링은 이번 범위에서 제외
# [경로] Path(__file__) 기준으로 찾고, 후보 경로를 차례로 시도한다. 파일이 없으면 에러 대신 안내를 보여 준다.
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title='WEATHER_FIT Report', layout='centered')

ROOT = Path(__file__).resolve().parents[1]            # 프로젝트 맨 위 폴더
DASH = ROOT / 'data' / 'dashboard'                    # 시각화 메뉴가 읽는 결과 파일 폴더

# ---------- 원본 파일 후보 경로 (실제 위치에 맞게 수정) ----------
RAW = {
    'asos': [ROOT / 'data' / '01_raw' / 'raw_w1_asos.csv', ROOT / 'data' / 'raw' / 'kma_asos' / 'raw_w1_asos.csv'],
    'warn': [ROOT / 'data' / 'raw' / 'kma_warning' / 'kma_warning_all.csv', ROOT / 'data' / '01_raw' / 'kma_warning_all.csv'],
    'fcst': [ROOT / 'data' / 'raw' / 'kma_forecast' / 'kma_forecast_all.csv', ROOT / 'data' / '01_raw' / 'kma_forecast_all.csv'],
    't3':   [ROOT / 'data' / 'raw' / 'datalab_keyword' / 'all_years.csv', ROOT / 'data' / '01_raw' / 'datalab_keyword_all_years.csv'],
}

# ---------- 색·테마 ----------
BG, INK, BURGUNDY, GRID = '#FAF7F2', '#2B2B2B', '#8E2C2C', '#E6DED0'
COLD, HEAT, GRAY = '#2F5D8A', '#C0502D', '#8C8C8C'
ANCHORS = ['반팔티', '양말', '립스틱', '등산화']       # 묶음 크기를 맞추려고 넣은 기준 키워드 (날씨 민감 품목이 아님)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;600;700;900&display=swap');
div.block-container {max-width: 980px !important; padding-top: 2rem;}
h1, h2, h3 {font-family: 'Noto Serif KR', serif; letter-spacing: -0.01em;}
.tag {color:#8E2C2C; font-size:0.8rem; letter-spacing:0.25em; font-weight:700; margin-bottom:-0.4rem;}
.lead {font-family:'Noto Serif KR', serif; font-size:1.2rem; line-height:1.6; margin:0.2rem 0 1rem 0;}
.pull {font-family:'Noto Serif KR', serif; font-size:1.1rem; line-height:1.65; border-left:4px solid #8E2C2C; padding:0.4rem 1rem; margin:1rem 0;}
[data-testid="stHeaderActionElements"], a.header-anchor {display:none !important;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# 공통 도우미
# ======================================================================
def find_file(key):
    # 후보 경로 중 실제로 있는 첫 파일. 없으면 None (앱이 멈추지 않게)
    for p in RAW[key]:
        if p.exists():
            return p
    return None

def need(key, label):
    # 원본이 없으면 안내를 보여 주고 None을 돌려준다
    p = find_file(key)
    if p is None:
        st.warning(f'{label} 원본을 찾지 못했습니다. app.py 위쪽 RAW 목록의 경로를 실제 위치로 고쳐 주세요.')
    return p

@st.cache_data
def read_dash(name, **kw):
    return pd.read_csv(DASH / name, **kw)

def style(fig, height=420):
    # 모든 차트의 배경·글꼴·격자를 같은 톤으로 맞춘다. 축 이름은 각 차트에서 직접 지정한다.
    # 제목(t)과 x축 이름·범례(b)가 잘리지 않도록 여백을 넉넉히 둔다.
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(family='Noto Serif KR, serif', color=INK, size=13),
                      margin=dict(l=10, r=10, t=70, b=80),
                      legend=dict(orientation='h', y=-0.3))
    fig.update_xaxes(showline=True, linecolor=INK, gridcolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, automargin=True)
    return fig

def header(tag, title, lead):
    st.markdown(f"<div class='tag'>{tag}</div>", unsafe_allow_html=True)
    st.markdown(f'# {title}')
    st.markdown(f"<div class='lead'>{lead}</div>", unsafe_allow_html=True)

def pull(text):
    st.markdown(f"<div class='pull'>{text}</div>", unsafe_allow_html=True)

def show(df):
    st.dataframe(df, hide_index=True, use_container_width=True)

def short_term(s, w=31):
    # 앞뒤 w일 평균(= 그 시기 평소 수준)을 뺀다 → 계절과 장기 하락을 단순하게 제거 (노트북과 같은 방식)
    return s - s.rolling(w, center=True, min_periods=15).mean()

def log_level(s):
    # 0이 있어도 log가 되도록 양수 값의 하위 1%를 더한다 (노트북과 같음)
    eps = max(s[s > 0].quantile(0.01), 1e-3)
    return np.log(s + eps)

def rule_first_days(cond, min_len=2):
    # 노트북과 같음: cond가 min_len일 이상 연속되는 구간의 첫날 목록
    run_end = cond.astype(int).rolling(min_len).sum().eq(min_len)
    in_run = run_end.copy()
    for s in range(1, min_len):
        in_run |= run_end.shift(-s, fill_value=False)
    return in_run.index[in_run & ~in_run.shift(1, fill_value=False)]

# ======================================================================
# W1 기상 관측 함수 (노트북 코드를 함수로 분리 → 단계마다 호출)
# ======================================================================
W1_COLUMNS = [
    'TM', 'STN', 'WS_AVG', 'WR_DAY', 'WD_MAX', 'WS_MAX', 'WS_MAX_TM', 'WD_INS', 'WS_INS', 'WS_INS_TM', 'TA_AVG', 'TA_MAX', 'TA_MAX_TM',
    'TA_MIN', 'TA_MIN_TM', 'TD_AVG', 'TS_AVG', 'TG_MIN', 'HM_AVG', 'HM_MIN', 'HM_MIN_TM', 'PV_AVG', 'EV_S', 'EV_L', 'FG_DUR', 'PA_AVG',
    'PS_AVG', 'PS_MAX', 'PS_MAX_TM', 'PS_MIN', 'PS_MIN_TM', 'CA_TOT', 'SS_DAY', 'SS_DUR', 'SS_CMB', 'SI_DAY', 'SI_60M_MAX', 'SI_60M_MAX_TM',
    'RN_DAY', 'RN_D99', 'RN_DUR', 'RN_60M_MAX', 'RN_60M_MAX_TM', 'RN_10M_MAX', 'RN_10M_MAX_TM', 'RN_POW_MAX', 'RN_POW_MAX_TM',
    'SD_NEW', 'SD_NEW_TM', 'SD_MAX', 'SD_MAX_TM', 'TE_05', 'TE_10', 'TE_15', 'TE_30', 'TE_50']
KEEP = {'TA_MAX': 't_max', 'TA_MIN': 't_min', 'TA_AVG': 't_avg', 'RN_DAY': 'precip_mm', 'SD_NEW': 'snow_cm',
        'HM_AVG': 'humidity', 'WS_AVG': 'wind', 'CA_TOT': 'cloud'}

@st.cache_data
def load_asos(path):
    # ① 헤더 없는 원본을 읽고 컬럼 이름을 붙인다. ② 날짜가 0 이하인 이상 행을 지우고 날짜형으로 바꾼다.
    w1 = pd.read_csv(path, header=None)
    w1.columns = W1_COLUMNS[:len(w1.columns)]
    n_raw = len(w1)
    w1 = w1[w1['TM'] > 0].copy()
    w1['date'] = pd.to_datetime(w1['TM'].astype(str), format='%Y%m%d')
    return w1.sort_values('date').reset_index(drop=True), n_raw

def select_asos(w1):
    # 분석에 쓸 컬럼만 고르고 이름을 알기 쉽게 바꾼다 (선정 이유는 EDA STEP 2 설명 참고)
    return w1[['date'] + list(KEEP)].rename(columns=KEEP).set_index('date')

def clean_asos(w):
    # 컬럼의 물리적 의미에 따라 결측 표기(-9, -99)를 다르게 처리한다.
    wd = w.copy()
    stats = {'rain_m9': int((wd['precip_mm'] == -9).sum()), 'rain_zero': int((wd['precip_mm'] == 0).sum())}
    for c in ['t_max', 't_min', 't_avg']:                       # 기온: -99만 결측 (서울 겨울엔 -9.0℃가 실제 기온)
        wd.loc[wd[c] == -99, c] = np.nan
    for c in ['humidity', 'wind', 'cloud']:                     # 음수가 불가능한 값: -9, -99 모두 결측
        wd.loc[wd[c].isin([-9, -99]), c] = np.nan
    for c in ['precip_mm', 'snow_cm']:                          # 강수·적설: -9는 "없음"(0), -99는 결측
        wd.loc[wd[c] == -9, c] = 0.0
        wd.loc[wd[c] == -99, c] = np.nan
    stats['wind_na_before'] = int(wd['wind'].isna().sum())
    wd['wind'] = wd['wind'].interpolate(limit=3)                # 연속 3일까지만 보간 (그 이상은 값을 지어내는 셈)
    stats['wind_na_after'] = int(wd['wind'].isna().sum())
    return wd, stats

def add_features(wd):
    # 가설 검증용 파생 변수: 일교차, 전일 대비 변화, 날씨 유형
    wd = wd.copy()
    wd['dtr'] = wd['t_max'] - wd['t_min']
    wd['dt_min'] = wd['t_min'].diff()
    wd['dt_max'] = wd['t_max'].diff()
    wd['weather'] = np.select([wd['snow_cm'] > 0, wd['precip_mm'] >= 0.1, wd['cloud'] >= 9, wd['cloud'] >= 6],
                              ['눈', '비', '흐림', '구름많음'], default='맑음')     # 눈 > 비 > 운량 순서로 분류
    return wd

@st.cache_data
def load_csv(path, **kw):
    return pd.read_csv(path, **kw)

# ======================================================================
# 메뉴 0. 개요
# ======================================================================
def page_overview():
    header('REPORT · SEOUL · 2023.01 – 2026.07', 'WEATHER_FIT', '서울의 날씨는 옷차림 수요를 얼마나 설명하는가')
    st.caption('기상청 관측·예보·특보 × 네이버 쇼핑 클릭·검색 · 1,308일')
    st.markdown('날씨에 따른 옷차림 수요를 이해하고, 날씨 기반 추천이 가능한지 살펴보기 위한 프로젝트입니다. '
                '월·계절 단위의 추천은 급격한 일교차나 첫 한파, 폭우 같은 날의 수요를 반영하지 못한다는 문제의식에서 출발했습니다.')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('분석 기간', '1,308일')
    c2.metric('쇼핑 키워드', '30개')
    c3.metric('한파·폭염 특보', '각 20건')
    c4.metric('가장 민감한 키워드', '롱패딩 -0.85')
    st.divider()

    st.markdown('### 핵심 결과')
    st.markdown('1. **계절을 빼지 않으면 상관이 부풀려진다.** 장갑×최저기온 -0.71 → 평소 수준을 뺀 뒤 -0.46\n'
                '2. **방한 키워드는 평소보다 추운 날 클릭이 늘어난다.** 롱패딩 -0.85, 패딩 -0.74, 어그 -0.70, 장갑 -0.68\n'
                '3. **초기 가설 중 전일 대비 변화와 체감온도는 지지되지 않았다.** 평균 |r| 0.03, 체감온도는 평균기온보다 약간 낮음\n'
                '4. **이벤트 기준을 특보로 검증했다.** 한파 ≤ -10℃ 2일(9건, 정밀도 1.00), 폭염 ≥ 33℃ 2일(14건, 정밀도 1.00)\n'
                '5. **"내일 예보" 기반 추천은 성립하지 않았다.** 가장 가까운 예보가 3~4일 앞')
    st.divider()

    st.markdown('### 이 앱의 구성')
    show(pd.DataFrame({
        '메뉴': ['1 · EDA (원본부터)', '1-2 · 분석 과정 (노트북 순서)', '2 · 시각화 (분석 결과)', '3 · 인사이트와 한계', '(제외) 모델링'],
        '내용': ['원본 파일을 read_csv부터 읽어 결측 확인 → 처리 → 파생 변수 → 시각화를 단계별로 실행 (기상 관측은 함수로 분리)',
               '노트북 순서 그대로 예보 병합·오차, 특보 발효 구간 조립, 한파·폭염 규칙 채점을 앱에서 직접 실행',
               '노트북에서 계산해 저장한 결과(계절 제거, 키워드 순위, 이벤트 기준, 예보 오차)를 차트로 확인',
               '알게 된 것, 가설과 달랐던 결과, 제외한 항목, 한계',
               '"내일 예보"가 없고 예보 오차가 효과 크기와 비슷해 이번 범위에서 제외']}))
    st.caption('분석 코드는 노트북(`notebooks/`)에서 먼저 완성했고, 이 앱은 그 코드를 단계별로 옮겼습니다.')

# ======================================================================
# 메뉴 1. EDA (원본부터)
# ======================================================================
def page_eda():
    header('EDA', '원본부터 단계별', '원본 파일을 read_csv로 읽는 것부터, 노트북에서 확인한 순서 그대로 실행합니다.')
    t_wx, t_warn, t_fc, t_shop = st.tabs(['기상 관측', '특보', '예보', '쇼핑 키워드'])

    # ------------------------------------------------------------------ 기상 관측
    with t_wx:
        p = need('asos', '기상 관측(raw_w1_asos.csv)')
        if p is None:
            return
        with st.expander('STEP 1 · 원본 불러오기 (read_csv)', expanded=True):
            w1, n_raw = load_asos(p)
            st.markdown('기상청 ASOS 일자료 원본은 **헤더가 없는 CSV**라서 컬럼 이름을 직접 붙여 읽고, 날짜(`20230101` 형태)를 날짜형으로 바꿨습니다.')
            c1, c2, c3 = st.columns(3)
            c1.metric('원본 행 수', f'{n_raw:,}')
            c2.metric('이상 행 제거 후', f'{len(w1):,}일')
            c3.metric('중복 날짜', f'{int(w1["date"].duplicated().sum())}건')
            st.caption(f'기간 {w1["date"].min().date()} ~ {w1["date"].max().date()}')
            show(w1[['date', 'TA_AVG', 'TA_MAX', 'TA_MIN', 'HM_AVG', 'WS_AVG', 'RN_DAY']].head(5))
        st.divider()

        w = select_asos(w1)
        with st.expander('STEP 2 · 분석에 쓸 컬럼 선택', expanded=True):
            show(pd.DataFrame({
                '컬럼': ['TA_MAX / TA_MIN / TA_AVG', 'RN_DAY', 'SD_NEW', 'HM_AVG', 'WS_AVG', 'CA_TOT'],
                '이름': ['t_max / t_min / t_avg', 'precip_mm', 'snow_cm', 'humidity', 'wind', 'cloud'],
                '선택 이유': ['일교차, 한파·폭염 이벤트 정의', '우산 등 비 관련 상품, 호우 특보 대조', '눈 오는 날 구분, 대설 특보 대조',
                          '습도(여름 체감)', '풍속(겨울 체감온도 계산)', '맑음/흐림 구분']}))
            st.caption('원본에는 50개가 넘는 컬럼이 있고, 나머지(시각 컬럼, 지중온도, 기압, 일조 등)는 일 단위 패션 수요 분석과 직접 관련이 없어 제외했습니다.')
        st.divider()

        with st.expander('STEP 3 · 결측 표기 확인 (-9, -99)', expanded=True):
            st.markdown('기상청은 "측정하지 못함"을 -9나 -99로 적습니다. 컬럼마다 이 값이 몇 번 나오는지 먼저 셉니다.')
            bad = pd.DataFrame({'컬럼': w.columns, '-9인 날': (w == -9).sum().values, '-99인 날': (w == -99).sum().values})
            show(bad)
            st.caption('서울 겨울에는 -9.0℃가 실제 기온이라 기온 컬럼의 -9는 결측이 아닙니다.')
        st.divider()

        wd, stt = clean_asos(w)
        with st.expander('STEP 4 · 컬럼별 규칙으로 결측 처리', expanded=True):
            show(pd.DataFrame({'컬럼 종류': ['기온', '습도·풍속·운량', '강수·적설'],
                               '처리': ['-99만 결측', '-9, -99 모두 결측', '-9는 0(없음), -99는 결측'],
                               '이유': ['서울 겨울의 -9.0℃는 실제 기온', '음수가 될 수 없는 값', '-9는 "비·눈이 안 왔다"는 뜻']}))
            c1, c2, c3 = st.columns(3)
            c1.metric('무강수(0mm) 일수', f'{int((wd["precip_mm"] == 0).sum())}일', help=f'원본 -9 표기 {stt["rain_m9"]}일 + 원본 0.0 기록 {stt["rain_zero"]}일')
            c2.metric('최저기온 -9℃인 날', f'{int((wd["t_min"] == -9).sum())}일', help='일괄 삭제하지 않았다는 증거')
            c3.metric('풍속 결측', f'{stt["wind_na_before"]}건 → {stt["wind_na_after"]}건')
            st.markdown(f'**검증**: 무강수 {int((wd["precip_mm"] == 0).sum())}일 = 원본 `-9` 표기 {stt["rain_m9"]}일 + 원본 `0.0` 기록 {stt["rain_zero"]}일. '
                        '풍속 결측은 연속 3일까지만 선형 보간했습니다(그 이상은 값을 지어내는 셈).')
            show(wd.isna().sum().rename('남은 결측').reset_index().rename(columns={'index': '컬럼'}))
        st.divider()

        wd = add_features(wd)
        with st.expander('STEP 5 · 파생 변수 만들기', expanded=True):
            st.markdown('- **일교차** `dtr` = 최고 − 최저 → 가설 "일교차에 반응한다" 검증\n'
                        '- **전일 대비 변화** `dt_min`, `dt_max` = 오늘 − 어제 → 가설 "전날 대비 기온 변화에 반응한다" 검증 (첫날은 이전 값이 없어 결측)\n'
                        '- **날씨 유형** = 눈 > 비 > 운량(0~5 맑음, 6~8 구름많음, 9~10 흐림) 순서로 분류 (원본에 유형 컬럼이 비어 있어 직접 파생)')
            show(wd[['t_min', 't_max', 'dtr', 'dt_min', 'precip_mm']].describe().round(1).reset_index().rename(columns={'index': '통계'}))
            c1, c2 = st.columns(2)
            c1.metric('일교차 최솟값', f'{wd["dtr"].min():.1f}℃', help='음수면 최저 > 최고라서 데이터 오류')
            c2.metric('빠진 날', f'{len(pd.date_range(wd.index.min(), wd.index.max())) - len(wd)}일')
        st.divider()

        with st.expander('STEP 6 · 서울 기온·일교차 추이', expanded=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wd.index, y=wd['t_max'], name='최고기온', line=dict(color=HEAT, width=1)))
            fig.add_trace(go.Scatter(x=wd.index, y=wd['t_min'], name='최저기온', line=dict(color=COLD, width=1), fill='tonexty', fillcolor='rgba(150,150,150,0.15)'))
            fig.update_layout(title='서울 일별 최고·최저기온')
            fig.update_xaxes(title_text='날짜')
            fig.update_yaxes(title_text='기온 (℃)')
            st.plotly_chart(style(fig, 380), use_container_width=True)

            m = wd.groupby(wd.index.month)['dtr'].agg(['mean', 'std']).round(1).reset_index().rename(columns={'date': '월'})
            fig2 = px.bar(m, x='월', y='mean', error_y='std', color_discrete_sequence=[BURGUNDY], labels={'mean': '평균 일교차 (℃, 선은 표준편차)'})
            fig2.update_layout(title='월별 평균 일교차')
            fig2.update_xaxes(tickmode='linear', dtick=1)
            st.plotly_chart(style(fig2, 330), use_container_width=True)
            st.markdown('일교차는 봄(3~5월)에 가장 크고 장마철(7~8월)에 가장 작습니다. 일교차도 자체 계절성이 있어서, 수요와 비교하려면 계절 성분을 먼저 빼야 합니다.')
        st.divider()

        with st.expander('STEP 7 · 날씨 유형 분포 (맑음·구름 많음·흐림·비·눈)', expanded=True):
            cnt = wd['weather'].value_counts().reindex(['맑음', '구름많음', '흐림', '비', '눈']).fillna(0).astype(int).reset_index()
            cnt.columns = ['날씨 유형', '일수']
            fig3 = px.bar(cnt, x='날씨 유형', y='일수', color='날씨 유형',
                          color_discrete_map={'맑음': '#D9A441', '구름많음': '#B7B7B7', '흐림': '#7F7F7F', '비': COLD, '눈': '#9CC3E6'})
            fig3.update_layout(title='날씨 유형별 일수 (운량·강수·적설로 직접 분류)', showlegend=False)
            st.plotly_chart(style(fig3, 330), use_container_width=True)
            st.caption('분류 기준(운량 구간, 비 0.1mm 이상)은 직접 정한 것입니다.')

    # ------------------------------------------------------------------ 특보
    with t_warn:
        p = need('warn', '특보(kma_warning_all.csv)')
        if p is None:
            return
        with st.expander('STEP 1 · 원본 불러오기와 구조 확인', expanded=True):
            wn = load_csv(p)
            st.markdown('특보 원본은 **한 행이 사건 하나**(발표·해제·변경)입니다. 해제 시각 컬럼(`tm_ed`)이 있지만 비어 있어서, 해제는 `cmd=3`인 별도 행으로 따로 짝지어야 합니다.')
            c1, c2, c3 = st.columns(3)
            c1.metric('행 수', f'{len(wn):,}')
            c2.metric('컬럼 수', f'{wn.shape[1]}')
            c3.metric('권역 수', f'{wn["reg_id"].nunique()}')
            show(wn['cmd_name'].value_counts().rename('건수').reset_index().rename(columns={'cmd_name': '사건'}))
        st.divider()

        issued = wn[wn['cmd'] == 1]
        with st.expander('STEP 2 · 특보 종류별 발표 건수', expanded=True):
            by_type = issued['wrn_name'].value_counts().reset_index()
            by_type.columns = ['특보 종류', '발표 건수']
            fig4 = px.bar(by_type, x='특보 종류', y='발표 건수', color_discrete_sequence=[BURGUNDY])
            fig4.update_layout(title='특보 종류별 발표 건수 (발표 행 기준, 등급 구분 없음)')
            st.plotly_chart(style(fig4, 340), use_container_width=True)
            st.caption('문서의 코드표(C=대설, D=한파)는 실측과 달랐습니다. 이름 컬럼과 계절 분포로 확인해 C=한파, H=폭염으로 정했습니다.')
        st.divider()

        with st.expander('STEP 3 · 권역별 비율 (한파·건조 등)', expanded=True):
            st.markdown('서울은 2020년부터 **4개 권역**(동남·동북·서남·서북)으로만 특보가 발표됩니다.')
            pick = st.multiselect('비율을 볼 특보', list(by_type['특보 종류']), default=[t for t in ['한파', '건조'] if t in set(by_type['특보 종류'])], key='eda_warn_pick')
            if pick:
                ct = pd.crosstab(issued['region'], issued['wrn_name'])
                share = ct.div(ct.sum(axis=1), axis=0)                      # 그 권역 전체 발표 중 해당 특보가 차지하는 비율
                long = share[pick].reset_index().melt(id_vars='region', var_name='특보', value_name='비율')
                fig5 = px.bar(long, x='region', y='비율', color='특보', barmode='group', labels={'region': '권역', '비율': '그 권역 전체 발표 중 비율'})
                fig5.update_layout(title='권역별 특보 비율')
                st.plotly_chart(style(fig5, 350), use_container_width=True)
                show(ct[pick].reset_index().rename(columns={'region': '권역'}))
        st.divider()

        with st.expander('STEP 4 · 한파 발표·해제 짝이 맞는가', expanded=True):
            cold = wn[wn['wrn_name'] == '한파']
            ct2 = pd.crosstab(cold['reg_id'], cold['cmd_name'])
            show(ct2.reset_index().rename(columns={'reg_id': '권역 ID'}))
            st.markdown('한파는 권역별로 **발표와 해제가 정확히 같은 수**라서 짝지어 발효 구간을 만들 수 있었습니다. '
                        '변경은 경보 → 주의보 하향이었습니다. 서울 전체 특보일은 4개 권역 중 하나라도 발효 중이면 서울 특보일(any)로 집계했습니다.')

    # ------------------------------------------------------------------ 예보
    with t_fc:
        p = need('fcst', '예보(kma_forecast_all.csv)')
        if p is None:
            return
        with st.expander('STEP 1 · 원본 불러오기', expanded=True):
            fc = load_csv(p, parse_dates=['tmfc', 'target_date'])
            st.markdown('중기예보 **발표분 아카이브**입니다. 발표 시각(`tmfc`)과 예보 대상일(`target_date`)이 있고, 기온(`temp`)과 육상(`land`)이 한 파일에 섞여 있습니다.')
            c1, c2 = st.columns(2)
            c1.metric('행 수', f'{len(fc):,}')
            c2.metric('발표 시각대', ', '.join(f'{h}시' for h in sorted(fc['tmfc'].dt.hour.unique())))
            show(fc['src'].value_counts().rename('행 수').reset_index().rename(columns={'src': '구분'}))
        st.divider()

        with st.expander('STEP 2 · 리드타임: "내일 예보"가 있는가', expanded=True):
            tp = fc[fc['src'] == 'temp'].copy()
            tp['lead_days'] = (tp['target_date'] - tp['tmfc'].dt.normalize()).dt.days     # 발표일 → 대상일 일수
            c1, c2 = st.columns(2)
            c1.metric('최소 리드타임', f'{int(tp["lead_days"].min())}일')
            c2.metric('최대 리드타임', f'{int(tp["lead_days"].max())}일')
            g = tp.groupby([tp['tmfc'].dt.year.rename('발표연도'), tp['tmfc'].dt.hour.rename('발표시')])['lead_days'].agg(['count', 'min', 'max']).reset_index()
            g.columns = ['발표연도', '발표시', '발표 횟수', '최소 리드', '최대 리드']
            show(g)
            st.markdown('가장 가까운 예보가 **3일 앞**이고, 2025년 발표분부터는 06시 4일 / 18시 5일로 늘었습니다(2024-11-29 06시 발표분부터). **"내일 예보"는 이 데이터에 없습니다.**')

    # ------------------------------------------------------------------ 쇼핑 키워드
    with t_shop:
        p = need('t3', '쇼핑 키워드(datalab_keyword/all_years.csv)')
        if p is None:
            return
        with st.expander('STEP 1 · 원본 불러오기', expanded=True):
            t3 = load_csv(p, parse_dates=['date'], dtype={'cid': str})
            st.markdown('네이버 쇼핑인사이트 키워드 클릭지수입니다. 요청 묶음(`batch_id`)마다 기준(앵커) 키워드가 함께 들어 있습니다.')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('행 수', f'{len(t3):,}')
            c2.metric('배치 수', f'{t3["batch_id"].nunique()}')
            c3.metric('키워드 수', f'{t3["keyword"].nunique()}')
            c4.metric('날짜 수', f'{t3["date"].nunique():,}')
            show(t3.head(6)[['date', 'keyword', 'ratio', 'rescaled', 'batch_id', 'cid', 'is_anchor']])
        st.divider()

        with st.expander('STEP 2 · 상대지수의 함정: 묶음 사이 배율', expanded=True):
            st.markdown('지수는 "같이 요청한 묶음에서 가장 큰 값 = 100"인 상대값입니다. 앵커 키워드의 묶음 간 배율이 날짜와 무관한 **상수**인지 변동계수(CV)로 확인합니다.')
            rows = []
            for cid, a in t3[t3['is_anchor']].groupby('cid'):
                pv = a.pivot_table(index='date', columns='batch_id', values='ratio')
                r = pv.div(pv[pv.columns[0]], axis=0)               # 첫 묶음을 기준으로 한 날짜별 배율
                for b in pv.columns:
                    rows.append({'cid': cid, '앵커': a['keyword'].iloc[0], '묶음': b, '기준 대비 배율': r[b].mean(), '배율의 CV': r[b].std() / r[b].mean()})
            show(pd.DataFrame(rows).round(4))
            st.markdown('CV가 모든 묶음에서 0 → 차이는 **상수 하나**로 설명됩니다. 묶음마다 상수를 곱해 같은 척도로 맞췄습니다. '
                        '저장돼 있던 `rescaled`(같은 날 앵커로 나눈 값)는 앵커의 계절성이 모든 키워드에 섞여 들어가서 쓰지 않았습니다.')
        st.divider()

        with st.expander('STEP 3 · 희소 키워드: 며칠이나 데이터가 있나', expanded=True):
            n_days = t3['date'].nunique()
            cov = (t3[~t3['is_anchor']].groupby('keyword')['date'].nunique() / n_days).rename('커버리지').reset_index().sort_values('커버리지')
            low = cov[cov['커버리지'] < 0.95]
            st.metric('커버리지 95% 미만 키워드', f'{len(low)}개 / {len(cov)}개')
            show(low.round(3))
            st.markdown('클릭이 적은 날은 행이 누락됩니다. 이 중 대부분은 같은 키워드의 **검색량이 전부 채워져** 있어서(정황 근거) 수요가 없는 게 아니라 조회한 카테고리가 실제 분류와 달라 클릭이 안 잡힌 것으로 판단하고, 검색량으로 대체했습니다.')

# ======================================================================
# 메뉴 1-2. 분석 과정 (노트북 순서 그대로 옮긴 코드, 방법 2)
# ======================================================================
def page_flow():
    header('ANALYSIS', '분석 과정', '노트북에서 실행한 순서 그대로, 원본을 read_csv로 읽어 이 앱에서 직접 실행합니다.')

    # ---------------- W1: 기상 관측 (앞에서 만든 함수 재사용) ----------------
    p_asos = need('asos', '기상 관측(raw_w1_asos.csv)')
    if p_asos is None:
        return
    w1, _ = load_asos(p_asos)
    wd, _ = clean_asos(select_asos(w1))
    wd = add_features(wd)
    st.success(f'W1 기상 관측 {len(wd):,}일 준비 완료 (1 · EDA 메뉴에서 단계별로 확인한 것과 같은 처리)')
    st.divider()

    # ---------------- W2·W3: 중기예보 ----------------
    st.markdown('## W2·W3 중기예보: 예보와 관측은 얼마나 다른가')
    p_fc = need('fcst', '예보(kma_forecast_all.csv)')
    if p_fc is not None:
        fc = pd.read_csv(p_fc, parse_dates=['tmfc', 'target_date'])

        with st.expander('① 기온 예보와 육상 예보를 나누고, 리드타임 계산', expanded=True):
            temp = fc[fc['src'] == 'temp'].copy()
            land = fc[fc['src'] == 'land'].copy()
            temp['lead_days'] = (temp['target_date'] - temp['tmfc'].dt.normalize()).dt.days     # 발표일 → 대상일 일수
            c1, c2, c3 = st.columns(3)
            c1.metric('기온 예보 행', f'{len(temp):,}')
            c2.metric('육상 예보 행', f'{len(land):,}')
            c3.metric('최소 리드타임', f'{int(temp["lead_days"].min())}일')
            st.markdown('가장 가까운 예보도 3일 앞입니다. **"내일 예보"는 없습니다.**')

        with st.expander('② 육상 예보를 하루 한 줄로 접기', expanded=True):
            land['target_day'] = land['target_date'].dt.normalize()
            land_daily = (land.sort_values(['tmfc', 'target_date']).groupby(['tmfc', 'target_day'])
                              .agg(rn_max=('rn_st', 'max'), rn_mean=('rn_st', 'mean'),
                                   sky_pm=('SKY', 'last'), wf_pm=('WF', 'last'), n_rows=('rn_st', 'size')).reset_index())
            c1, c2 = st.columns(2)
            c1.metric('접은 뒤 행 수', f'{len(land_daily):,}', help='기온 예보 행 수(19,096)와 같아야 병합해도 행이 늘지 않음')
            c2.metric('기온 예보 행 수', f'{len(temp):,}')
            show(land_daily['n_rows'].value_counts().rename('날 수').reset_index().rename(columns={'n_rows': '하루당 원래 행 수'}))
            st.caption('가까운 날짜는 오전·오후 2줄, 먼 날짜는 하루 1줄로 섞여 있어서 하루 한 줄로 접었습니다(강수확률은 최댓값).')

        with st.expander('③ 병합 후, 대상일마다 "그날 알 수 있었던 가장 최신 예보" 선택', expanded=True):
            temp2 = temp.assign(target_day=temp['target_date'].dt.normalize())
            fc_all = temp2.merge(land_daily, on=['tmfc', 'target_day'], how='inner')
            cand = fc_all[fc_all['tmfc'].dt.normalize() < fc_all['target_day']]                  # 발표일이 대상일보다 앞선 것만
            latest = cand.sort_values('tmfc').groupby('target_day').tail(1).set_index('target_day')
            latest['lead_days'] = (latest.index - latest['tmfc'].dt.normalize()).dt.days
            cmp = latest[['t_min', 't_max', 'lead_days']].join(wd[['t_min', 't_max']], rsuffix='_obs').dropna()
            c1, c2 = st.columns(2)
            c1.metric('병합 전 → 후 행 수', f'{len(temp2):,} → {len(fc_all):,}')
            c2.metric('관측과 비교한 날', f'{len(cmp):,}일')
            show(latest['lead_days'].value_counts().sort_index().rename('날 수').reset_index().rename(columns={'lead_days': '선택된 리드타임(일)'}))

        with st.expander('④ 예보 vs 관측 오차 (A: 3일 앞, B: 4일 이상 앞)', expanded=True):
            cmp['시기'] = np.where(cmp['lead_days'] == 3, 'A: 3일 앞', 'B: 4일 이상 앞')
            rows = [{'시기': pname, '일수': len(g),
                     '최저기온 MAE (℃)': (g['t_min'] - g['t_min_obs']).abs().mean(),
                     '최고기온 MAE (℃)': (g['t_max'] - g['t_max_obs']).abs().mean()} for pname, g in cmp.groupby('시기')]
            show(pd.DataFrame(rows).round(2))
            st.caption('노트북 결과와 같아야 합니다: A 1.13 / 1.32, B 1.42 / 1.78')
    st.divider()

    # ---------------- W4: 특보 ----------------
    st.markdown('## W4 특보: 한파·폭염 규칙을 특보로 채점')
    p_w = need('warn', '특보(kma_warning_all.csv)')
    if p_w is None:
        return
    w = pd.read_csv(p_w)
    DATA_END = pd.Timestamp('2026-07-31 23:59:59')
    full_idx = pd.date_range('2023-01-01', '2026-07-31')
    flags, first = {}, {}

    with st.expander('① 발표·해제를 권역별로 짝지어 발효 구간 만들기', expanded=True):
        st.markdown('해제 시각 컬럼이 비어 있어서, 발표(`cmd=1`) → 해제(`cmd=3`) 행을 권역별로 직접 짝지었습니다. 변경(`cmd=6`)은 구간을 끊지 않고 등급만 갱신했습니다.')
        for code, name in [('C', '한파'), ('H', '폭염')]:
            rows = []
            for reg, g in w[w['wrn'] == code].sort_values('tm_ef').groupby('reg_id'):
                g = g.copy()
                g['tm_ef'] = pd.to_datetime(g['tm_ef'])
                start, max_lvl = None, 0
                for _, r in g.iterrows():
                    if r['cmd'] == 1:
                        if start is not None:                                    # 해제 없이 새 발표가 오면 앞 구간을 데이터 끝으로 닫음
                            rows.append(dict(reg=reg, start=start, end=DATA_END, max_lvl=max_lvl, closed=False))
                        start, max_lvl = r['tm_ef'], r['lvl']
                    elif r['cmd'] == 6 and start is not None:
                        max_lvl = max(max_lvl, r['lvl'])                         # 변경: 등급만 갱신
                    elif r['cmd'] == 3 and start is not None:
                        rows.append(dict(reg=reg, start=start, end=r['tm_ef'], max_lvl=max_lvl, closed=True))
                        start = None
                if start is not None:                                            # 끝까지 해제 행이 없으면 분석 기간 끝으로 닫음
                    rows.append(dict(reg=reg, start=start, end=DATA_END, max_lvl=max_lvl, closed=False))
            ep = pd.DataFrame(rows)

            flag = pd.Series(False, index=full_idx)                              # 서울(any): 4개 권역 중 하나라도 발효 중이면 True
            for _, r in ep.iterrows():
                flag.loc[r['start'].normalize():r['end'].normalize()] = True
            flags[name] = flag
            first[name] = flag.index[flag & ~flag.shift(1, fill_value=False)]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f'{name} 구간', f'{len(ep)}개')
            c2.metric('해제 없이 끝난 구간', f'{int((~ep["closed"]).sum())}개')
            c3.metric('서울(any) 에피소드', f'{len(first[name])}건')
            c4.metric('특보 일수', f'{int(flag.sum())}일')
        st.caption('노트북 결과와 같아야 합니다: 한파 구간 69 / 에피소드 20 / 68일, 폭염 구간 77 / 에피소드 20 / 165일')

    with st.expander('② 규칙(임계값 + 2일 지속)이 실제 특보와 얼마나 맞는가', expanded=True):
        st.markdown('- **정밀도**: 규칙 이벤트 첫날 ±2일 안에 특보가 있었던 비율\n- **재현율**: 특보 에피소드 첫날 ±2일 안에 규칙 이벤트가 있었던 비율')
        TOL = 2                                                                  # 허용 오차(일), 임의값

        def score(rule_days, name):
            flag, eps = flags[name], first[name]
            hit_p = [flag.loc[d - pd.Timedelta(days=TOL): d + pd.Timedelta(days=TOL)].any() for d in rule_days]
            hit_r = [any(abs((d - e).days) <= TOL for d in rule_days) for e in eps]
            return len(rule_days), (float(np.mean(hit_p)) if len(hit_p) else np.nan), float(np.mean(hit_r))

        rows = []
        for th in [-12, -10, -8, -6]:
            n, pr, rc = score(rule_first_days(wd['t_min'] <= th), '한파')
            rows.append({'이벤트': '한파', '임계값': f'최저기온 ≤ {th}℃', '이벤트 수': n, '정밀도': pr, '재현율': rc})
        for th in [35, 34, 33, 32, 31]:
            n, pr, rc = score(rule_first_days(wd['t_max'] >= th), '폭염')
            rows.append({'이벤트': '폭염', '임계값': f'최고기온 ≥ {th}℃', '이벤트 수': n, '정밀도': pr, '재현율': rc})
        show(pd.DataFrame(rows).round(2))
        st.caption('노트북 결과와 같아야 합니다: 한파 -10℃ → 9건 / 1.00 / 0.45, 폭염 33℃ → 14건 / 1.00 / 0.25')

# ======================================================================
# 메뉴 2. 시각화 (분석 결과)
# ======================================================================
def page_viz():
    header('RESULTS', '분석 결과 시각화', '노트북에서 계산해 저장한 결과(data/dashboard)를 차트로 확인합니다.')
    t1, t2, t3, t4, t5 = st.tabs(['계절 제거', '시즌 지도', '날씨 민감도', '이벤트 기준', '예보'])

    with t1:
        st.markdown('### 계절을 빼지 않으면 상관이 부풀려진다')
        show(pd.DataFrame({'비교': ['장갑 × 최저기온', '수영 × 최고기온', '우산 × 최저기온', '우산 × 비'],
                           '그대로': [-0.71, 0.63, 0.66, 0.49], '평소 수준을 뺀 뒤': [-0.46, 0.15, 0.16, 0.62]}))
        st.caption('"평소 수준" = 앞뒤 31일 이동평균. 수요와 날씨 양쪽에 같은 방식으로 적용했고, 요일 평균도 한 번 더 뺐습니다.')
        fm = read_dash('fashion_monthly.csv', index_col='월')
        fig = go.Figure()
        for yr, col in zip(fm.columns, ['#2B2B2B', '#8E2C2C', '#C0502D', '#D9A441']):
            fig.add_trace(go.Scatter(x=fm.index, y=fm[yr], mode='lines+markers', name=f'{yr}년', line=dict(color=col)))
        fig.update_layout(title='패션 전체 클릭지수 (월평균, 2026년은 7월까지): 같은 달끼리 비교해도 해마다 낮아진다')
        fig.update_xaxes(title_text='월', tickmode='linear', dtick=1)
        fig.update_yaxes(title_text='클릭지수 (월평균)')
        st.plotly_chart(style(fig, 400), use_container_width=True)
        st.markdown('2026년 1~7월은 2023년의 약 0.22배입니다. 날씨와 상관없는 플랫폼 전체의 변화라서, 연도를 넘어 숫자를 직접 비교하지 않고 **평소 수준과의 차이**를 중심으로 봤습니다.')

    with t2:
        st.markdown('### 키워드별 월 계절성')
        sh = read_dash('seasonal_heatmap.csv', index_col='키워드')
        sh = sh.drop(index=[k for k in ANCHORS if k in sh.index])
        peak = sh.idxmax(axis=1).str.replace('월', '').astype(int)
        sh = sh.loc[peak.sort_values(kind='stable').index]
        fig = px.imshow(sh, zmin=0, zmax=1, aspect='auto', color_continuous_scale=[[0, '#FBF3E6'], [0.4, '#E8B88A'], [1, '#7A1F2B']],
                        labels=dict(x='월', y='키워드 (가장 많이 찾는 달 순)', color='최고 달 대비'))
        fig.update_layout(title='키워드별 월 계절성 (각 키워드의 최고 달 = 1.0)')
        st.plotly_chart(style(fig, 720), use_container_width=True)
        show(pd.DataFrame({'무리': ['겨울', '여름', '환절기', '가을 상의'],
                           '대표 키워드': ['롱패딩·패딩·장갑·목도리·어그', '반바지·민소매·샌들·래쉬가드', '블레이저·트렌치코트·가죽자켓·야상', '가디건·맨투맨·후드티'],
                           '피크': ['11~12월', '6~8월', '3월과 10월 (1년에 두 번)', '10월']}))
        st.caption('한 줄(키워드) 안에서만 비교합니다. 월마다 3~4개 관측의 평균입니다.')

    with t3:
        st.markdown('### 날씨에 민감한 키워드 순위')
        s3 = read_dash('screen_t3.csv', index_col=0)
        r3 = s3.drop(index=[a for a in ANCHORS if a in s3.index]).copy()
        r3['키워드'] = r3.index
        r3 = r3.sort_values('t_min')

        def kind(row):
            if row['최강 변수'] == 'log_precip':
                return '비에 반응 (기온은 해석 안 함)'
            if abs(row['t_min']) < 0.2:
                return '약한 상관 (|r| 0.2 미만)'
            return '추울수록 증가' if row['t_min'] < 0 else '더울수록 증가'
        r3['구분'] = r3.apply(kind, axis=1)
        fig = px.bar(r3, x='t_min', y='키워드', orientation='h', color='구분', category_orders={'키워드': r3['키워드'].tolist()},
                     color_discrete_map={'추울수록 증가': COLD, '더울수록 증가': HEAT, '약한 상관 (|r| 0.2 미만)': '#D8D2C8', '비에 반응 (기온은 해석 안 함)': GRAY},
                     labels={'t_min': '평소 대비 최저기온과의 상관 (성수기, 31일 평균 제거 + 요일 보정)'})
        fig.update_yaxes(autorange='reversed', title_text='키워드')
        fig.update_layout(title='키워드별 기온 반응')
        st.plotly_chart(style(fig, 760), use_container_width=True)
        st.caption('30개 × 5개 변수를 훑은 탐색이라 |r| 0.2 미만은 우연일 수 있습니다.')

        st.divider()
        st.markdown('### 직접 골라 보기')
        wd = read_dash('weather_daily.csv', index_col='date', parse_dates=True)
        dm = read_dash('demand_t3.csv', index_col='date', parse_dates=True)
        kws = [k for k in s3.index if k not in ANCHORS]
        c1, c2, c3 = st.columns(3)
        kw = c1.selectbox('키워드', kws, index=kws.index('롱패딩') if '롱패딩' in kws else 0, key='viz_kw')
        var = c2.radio('날씨 변수', ['최저기온', '최고기온', '강수량'], key='viz_var')
        only_peak = c3.checkbox('성수기만 보기', value=True, key='viz_peak')
        wvar = {'최저기온': wd['t_min'], '최고기온': wd['t_max'], '강수량': np.log1p(wd['precip_mm'])}[var]
        unit = 'log(1+mm)' if var == '강수량' else '℃'
        months = [int(m) for m in str(s3.loc[kw, '성수기 월']).split(',')]
        y = short_term(log_level(dm[kw]))
        y = y - y.groupby(y.index.dayofweek).transform('mean')          # 요일 보정
        x = short_term(wvar)
        d = pd.concat([x, y], axis=1, keys=['x', 'y']).dropna()
        if only_peak:
            d = d[d.index.month.isin(months)]
        slope, icpt = np.polyfit(d['x'], d['y'], 1)
        r = d['x'].corr(d['y'])
        sc = px.scatter(d, x='x', y='y', opacity=0.5, color_discrete_sequence=[BURGUNDY],
                        labels={'x': f'평소 대비 {var} 차이 ({unit})', 'y': '평소 대비 클릭 차이 (로그, 요일 보정)'})
        xs = np.linspace(d['x'].min(), d['x'].max(), 50)
        sc.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode='lines', name='추세선', line=dict(color=INK, width=2)))
        sc.update_layout(title=f'점 하나가 하루: 상관 {r:.2f} ({len(d)}일)')
        st.plotly_chart(style(sc, 420), use_container_width=True)
        if var != '강수량':
            pct = (np.exp(-slope) - 1) * 100
            st.markdown(f'**평소보다 1℃ 추우면 {kw} 클릭이 약 {abs(pct):.1f}% {"늘어나는" if pct > 0 else "줄어드는"} 경향** (31일 평균 제거 기준, 범위로 읽어야 함).')
        st.caption(f'성수기 월: {", ".join(map(str, months)) if only_peak else "전체"}. 같은 겨울 안의 날들은 서로 비슷하게 움직여 점이 독립이 아닙니다.')

        st.divider()
        st.markdown('### 어떤 날씨 변수가 가장 잘 설명하나')
        hv = read_dash('hypothesis_table.csv')
        long = hv.melt(id_vars='날씨 변수', var_name='데이터', value_name='점수')
        fig = px.bar(long, x='날씨 변수', y='점수', color='데이터', barmode='group', color_discrete_sequence=[GRAY, BURGUNDY],
                     labels={'점수': '평균 |r| (성수기, 평소 대비)'})
        fig.update_layout(title='날씨 변수별 수요와의 평균 상관')
        st.plotly_chart(style(fig, 400), use_container_width=True)
        st.markdown('설명력은 "평소보다 얼마나 추웠나"에 있었고, **전일 대비 변화는 평균 0.03으로 가장 약했습니다.** 3일 전과 비교하면 0.10까지 오르지만(장갑 -0.05 → -0.32) 여전히 기온 편차보다 약합니다.')

    with t4:
        st.markdown('### 몇 도부터 한파·폭염인가')
        st.markdown('기상청 특보를 **정답지**로 두고 규칙을 채점했습니다. **정밀도** = 규칙이 한파라고 한 날 중 실제 특보가 있던 비율, **재현율** = 실제 특보 중 규칙이 잡아낸 비율.')
        ev = read_dash('event_threshold.csv')
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=('한파 (최저기온 ≤ 임계값)', '폭염 (최고기온 ≥ 임계값)'))
        for col, name in [(1, '한파'), (2, '폭염')]:
            sub = ev[ev['이벤트'] == name]
            fig.add_trace(go.Scatter(x=sub['임계값'], y=sub['정밀도'], mode='lines+markers', name='정밀도', line=dict(color=COLD), showlegend=(col == 1)), row=1, col=col)
            fig.add_trace(go.Scatter(x=sub['임계값'], y=sub['재현율'], mode='lines+markers', name='재현율', line=dict(color=HEAT), showlegend=(col == 1)), row=1, col=col)
            ch = sub[sub['선택'].eq('✔')]
            fig.add_trace(go.Scatter(x=ch['임계값'], y=ch['정밀도'], mode='markers', name='선택한 임계값', marker=dict(symbol='star', size=16, color=BURGUNDY), showlegend=(col == 1)), row=1, col=col)
        fig.update_xaxes(title_text='임계값 (℃, 오른쪽일수록 완화)')
        fig.update_yaxes(title_text='특보와의 일치 비율 (±2일 허용)', col=1)
        st.plotly_chart(style(fig, 420), use_container_width=True)
        show(ev.drop(columns='선택'))
        pull('한파는 원안(-12℃)이 3년 7개월 동안 4번뿐이라 정밀도를 잃지 않고 9건이 잡히는 -10℃로, 폭염은 원안 33℃(14건, 정밀도 1.00)를 그대로 정했습니다.')
        st.caption('특보는 예보 기준이고 서울 4개 권역 중 한 곳만 해당돼도 발표돼서 규칙(서울 한 관측소)보다 자주 나옵니다. 어느 쪽이 틀린 게 아니라 기준이 다릅니다. 허용 오차 ±2일은 임의값입니다.')

        wn_ = read_dash('warning_daily.csv', index_col='date', parse_dates=True)
        rows = []
        for flag, name in [('cold_warn', '한파특보'), ('heat_warn', '폭염특보')]:
            f = wn_[flag].astype(bool)
            grp = (f != f.shift()).cumsum()
            for _, g in f.groupby(grp):
                if g.iloc[0]:
                    rows.append(dict(종류=name, 시작=g.index[0], 끝=g.index[-1] + pd.Timedelta(days=1)))
        tl = px.timeline(pd.DataFrame(rows), x_start='시작', x_end='끝', y='종류', color='종류', color_discrete_map={'한파특보': COLD, '폭염특보': HEAT})
        tl.update_xaxes(title_text='날짜')
        tl.update_yaxes(title_text='')
        tl.update_layout(showlegend=False, title='서울 한파·폭염 특보 기간 (4개 권역 중 하나라도 발효 중이면 표시)')
        st.plotly_chart(style(tl, 280), use_container_width=True)

    with t5:
        st.markdown('### 예보로 옷을 고를 수 있나')
        fc = read_dash('forecast_error.csv', index_col='date', parse_dates=True)
        fc['시기'] = np.where(fc['lead_days'] == 3, 'A: 3일 앞', 'B: 4일 이상 앞')
        rows = [{'시기': p, '일수': len(g), '최저기온 평균 오차 (℃)': (g['t_min'] - g['t_min_obs']).abs().mean(),
                 '최고기온 평균 오차 (℃)': (g['t_max'] - g['t_max_obs']).abs().mean()} for p, g in fc.groupby('시기')]
        show(pd.DataFrame(rows).round(2))
        which = st.radio('기온', ['최저기온', '최고기온'], horizontal=True, key='viz_fc')
        obs, pred = ('t_min_obs', 't_min') if which == '최저기온' else ('t_max_obs', 't_max')
        sp = px.scatter(fc, x=obs, y=pred, color='시기', opacity=0.45, color_discrete_map={'A: 3일 앞': COLD, 'B: 4일 이상 앞': HEAT},
                        labels={obs: f'관측 {which} (℃)', pred: f'예보 {which} (℃)'})
        lo, hi = float(fc[[obs, pred]].min().min()), float(fc[[obs, pred]].max().max())
        sp.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='예보 = 관측', line=dict(color=INK)))
        sp.update_layout(title=f'{which}: 예보 vs 관측 (검은 선에 가까울수록 정확)')
        st.plotly_chart(style(sp, 440), use_container_width=True)
        pull('예보 오차(1.1~1.8℃)는 방한 용품이 1℃ 추울 때 늘어나는 정도(약 3~9%)와 비슷한 크기입니다. 그래서 추천 모델은 이번 범위에서 제외했습니다.')
        st.caption('A·B가 서로 다른 연도라 오차 차이가 예보 시점 때문인지 연도 때문인지 알 수 없습니다.')

# ======================================================================
# 메뉴 3. 인사이트와 한계
# ======================================================================
def page_insights():
    header('INSIGHTS', '인사이트와 한계', '알게 된 것, 가설과 달랐던 것, 남은 것')
    st.markdown('### 알게 된 것')
    show(pd.DataFrame({
        '인사이트': ['방한 상품은 평소보다 추운 날 수요가 늘어난다', '여름 상품은 더운 날, 비 관련 상품은 비 온 당일에 반응한다',
                  '모든 겨울옷이 추위에 반응하지는 않는다', '추위 반응은 당일이 가장 크고 다음 날까지 일부 이어진다',
                  '카테고리보다 키워드 단위에서 신호가 선명하다', '검색과 쇼핑 클릭은 같은 날 같은 방향으로 움직인다'],
        '근거': ['롱패딩 -0.85, 패딩 -0.74, 어그 -0.70, 장갑 -0.68 (1℃ 추우면 롱패딩 약 +9%, 장갑 약 +8%, 범위로 해석)',
               '반바지 +0.48, 샌들 +0.42 / 우산 비와 +0.63 ~ 0.68', '코트 -0.09, 니트 -0.08, 가디건 -0.09, 맨투맨 -0.06 (성수기·바닥값을 바꿔도 약함)',
               '장갑 차분 시차 lag 0 -0.57, lag 1 -0.32 / 우산은 당일만', '카테고리는 장갑·우산 정도만 뚜렷, 키워드는 방한 7개가 -0.49 ~ -0.85',
               '같은 키워드 25개 중 22개(88%) 방향 일치, 차분 lag 0 상관 0.42 ~ 0.89']}))
    st.markdown('### 가설과 달랐던 결과')
    show(pd.DataFrame({
        '가설': ['전일 대비 기온 변화에 반응한다', '체감온도가 기온보다 더 잘 설명한다', '일교차에 반응한다', '내일 예보로 추천할 수 있다'],
        '결과': ['❌ 지지 안 됨', '❌ 지지 안 됨', '△ 대부분 약함', '❌ 성립 안 함'],
        '근거': ['평균 |r| 0.03 (3일 전과 비교하면 0.10)', '방한 키워드 5개 모두 평균기온이 약간 우세 (차이 -0.02 ~ -0.03)',
               '우산류에서만 강하고 비의 대리 변수로 보임(미확인)', '가장 가까운 예보가 3~4일 앞, 오차 1.1~1.8℃']}))
    st.divider()
    st.markdown('### 내가 정한 값과 점검')
    show(pd.DataFrame({
        '값': ['평소 수준 창', '성수기 기준', '로그 바닥값', '특보 대조 허용 오차', '"며칠 전 대비" 길이'],
        '선택': ['31일', '최고 달의 40% 이상', '하위 1%', '±2일', '3일'],
        '점검': ['15·61일: 방향은 같고 크기는 달라짐', '30~60%: 롱패딩·목도리·우산은 안정, 장갑·반바지는 크기 변함, 코트는 계속 약함', '0.5~10%: 거의 영향 없음', '점검하지 않음', '점검하지 않음']}))
    st.markdown('### 제외·보류한 것')
    show(pd.DataFrame({
        '항목': ['선케어', '수영', '기모레깅스', '기준용 키워드 4종', '추천 모델'],
        '결정': ['제외 (다른 주제와 함께 따로 분석)', '판단 보류', '제외', '순위 해석에서 제외', '제외'],
        '이유': ['옷차림과 직접 관련이 없음. 클릭은 4~7월 내내 꾸준히 높고 날씨와 연결이 약했는데, 검색은 더운 날과 강하게 연결됨',
               '성수기 기준에 따라 상관이 0.03~0.27로 흔들림', '쇼핑 클릭 커버리지 0.146, 검색도 0.875', '묶음 크기를 맞추려고 넣은 기준 키워드', '"내일 예보"가 없고 예보 오차가 효과 크기와 비슷, 정답(라벨) 정의가 남아 있음']}))
    st.divider()
    st.markdown('### 한계')
    st.markdown('- **상관은 원인이 아닙니다.** 같이 움직인다는 것까지만 확인했습니다.\n'
                '- **완전한 겨울은 3번**(2023-24, 2024-25, 2025-26)이고 2023년은 1~3월만 있습니다. 같은 겨울 안의 날들은 서로 비슷하게 움직여서 성수기 307일은 사실상 겨울 3~4번의 정보량입니다. 정확한 크기보다 **방향과 순서**를 근거로 삼아야 합니다.\n'
                '- 평소 수준을 **단순한 31일 이동평균**으로 구했습니다. 더 정교한 분해(MSTL)와 비교하면 겨울 용품의 효과 크기가 달라질 수 있어, 1℃당 +9.0%는 범위로 읽어야 합니다. 이 비교는 직접 하지 못했습니다.\n'
                '- 30개 키워드 × 5개 날씨 변수를 훑은 탐색이라 |r| 0.2 미만은 우연일 수 있습니다.\n'
                '- 한파·폭염 시작일 전후 수요 변화(D-7 ~ D+14)는 계산하지 못했습니다.')
    st.markdown('### 다음 단계')
    st.markdown('1. 코로나 이전(2017~2019)을 더해 겨울·여름 횟수 늘리기\n2. 겨울 용품에 MSTL을 적용해 효과 크기를 기준선별로 비교하기\n'
                '3. 한파·폭염 D0 기준 이벤트 전후 수요 곡선\n4. 패션 전체 트래픽을 통제한 날씨 효과\n5. 단기예보를 더해 "내일 추천" 다시 시도하기\n6. 선케어를 다른 주제와 함께 별도 분석하기')

# ======================================================================
# 사이드바 (큰 틀)
# ======================================================================
PAGES = {
    '0 · 개요': page_overview,
    '1 · EDA (원본부터)': page_eda,
    '1-2 · 분석 과정 (노트북 순서)': page_flow,
    '2 · 시각화 (분석 결과)': page_viz,
    '3 · 인사이트와 한계': page_insights,
}

st.sidebar.markdown('### WEATHER_FIT')
st.sidebar.caption('Seoul · Weather & Wardrobe')
st.sidebar.divider()
menu = st.sidebar.radio('메뉴', list(PAGES), label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('모델링: 이번 범위에서 제외')
PAGES[menu]()