# dashboard/app.py
# [목적] 노트북에서 계산해 저장한 결과(data/dashboard/*.csv)를 "패션 리포트"처럼 읽히게 보여 주는 포트폴리오용 앱.
# [구조] 사이드바에서 장을 고르면 해당 함수가 실행된다. 장 하나 = 함수 하나.
# [색 약속] 파랑 = 추위/한파, 주황 = 더위/폭염, 회색 = 기온 해석 안 함, 옅은 색 = 약한 상관
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title='WEATHER_FIT · Report', layout='centered')
DATA = Path(__file__).resolve().parents[1] / 'data' / 'dashboard'   # 어디서 실행해도 같은 경로

BG, INK, BURGUNDY, GRID = '#FAF7F2', '#2B2B2B', '#8E2C2C', '#E6DED0'
COLD, HEAT, GRAY = '#2F5D8A', '#C0502D', '#8C8C8C'
ANCHORS = ['반팔티', '양말', '립스틱', '등산화']      # 묶음 크기를 맞추려고 넣은 기준 키워드라 날씨 민감 품목이 아님

# ---------- 디자인 ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;600;700;900&display=swap');
div.block-container {max-width: 980px !important; padding-top: 2.5rem;}
h1, h2, h3 {font-family: 'Noto Serif KR', serif; letter-spacing: -0.01em;}
.chap-no {color:#8E2C2C; font-size:0.8rem; letter-spacing:0.25em; font-weight:700; margin-bottom:-0.4rem;}
.mast {font-family:'Noto Serif KR', serif; font-weight:900; font-size:3.2rem; letter-spacing:0.08em; line-height:1.1; margin:0;}
.sub {font-family:'Noto Serif KR', serif; font-size:1.15rem; color:#555; margin:0.2rem 0 0.4rem 0;}
.question {font-family:'Noto Serif KR', serif; font-size:1.3rem; line-height:1.6; margin:0.2rem 0 1rem 0;}
.pull {font-family:'Noto Serif KR', serif; font-size:1.2rem; line-height:1.65; border-left:4px solid #8E2C2C; padding:0.4rem 1rem; margin:1.2rem 0;}
.nextup {color:#7A7A7A; font-style:italic; margin-top:2rem;}
[data-testid="stHeaderActionElements"], a.header-anchor {display:none !important;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def read(name, **kw):
    return pd.read_csv(DATA / name, **kw)

def style(fig, height=420):
    # 모든 차트의 배경·글꼴·격자를 같은 톤으로 맞춘다. 축 이름은 각 차트에서 직접 지정한다.
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(family='Noto Serif KR, serif', color=INK, size=13),
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation='h', y=-0.22))
    fig.update_xaxes(showline=True, linecolor=INK, gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig

def chapter(no, title, subtitle):
    st.markdown(f"<div class='chap-no'>{no}</div>", unsafe_allow_html=True)
    st.markdown(f'# {title}')
    st.markdown(f"<div class='question'>{subtitle}</div>", unsafe_allow_html=True)

def pull(text):
    st.markdown(f"<div class='pull'>{text}</div>", unsafe_allow_html=True)

def nextup(text):
    st.markdown(f"<div class='nextup'>다음 장 → {text}</div>", unsafe_allow_html=True)

def table(df):
    st.dataframe(df, hide_index=True, use_container_width=True)

def short_term(s, w=31):
    # 앞뒤 w일 평균(= 그 시기 평소 수준)을 뺀다 → 계절과 장기 하락을 단순하게 제거 (노트북과 같은 방식)
    return s - s.rolling(w, center=True, min_periods=15).mean()

def log_level(s):
    # 0이 있어도 log가 되도록 양수 값의 하위 1%를 더한다 (노트북과 같음)
    eps = max(s[s > 0].quantile(0.01), 1e-3)
    return np.log(s + eps)

# ======================================================================
# COVER
# ======================================================================
def page_cover():
    st.markdown("<div class='chap-no'>REPORT · SEOUL · 2023.01 – 2026.07</div>", unsafe_allow_html=True)
    st.markdown("<div class='mast'>WEATHER_FIT</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>서울의 날씨는 옷장을 얼마나 움직이는가</div>", unsafe_allow_html=True)
    st.caption('1,308일 · 기상청 관측·예보·특보 × 네이버 쇼핑 클릭·검색')

    st.markdown('### 11월 15일에 패딩을 팔면 되는 걸까?')
    st.markdown('패션 추천은 대개 달력을 따릅니다. 11월이니 패딩, 6월이니 반바지. 그런데 11월 중순에 갑자기 영하로 떨어진 날과 포근했던 날의 옷장은 같을까요? '
                '이 리포트는 이 질문에서 시작했습니다.')
    pull('날씨가 달라지면 사람들이 찾는 옷도 달라질까? 달라진다면 얼마나, 언제, 어떤 옷에서?')
    st.markdown('서울의 하루하루 날씨 기록 1,308일과 같은 기간의 쇼핑 클릭·검색 기록을 날짜별로 맞춰 확인했습니다. 결과는 두 가지로 요약됩니다.\n\n'
                '- **롱패딩과 우산은 날씨에 솔직했습니다.** 반면 코트와 니트는 거의 무심했습니다.\n'
                '- **처음에 믿었던 가설 두 개는 틀렸습니다.** 이 리포트의 절반은 그 이야기입니다.')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('분석 기간', '1,308일')
    c2.metric('쇼핑 키워드', '30개')
    c3.metric('한파·폭염 특보', '각 20건')
    c4.metric('가장 솔직한 상품', '롱패딩 -0.85')

    st.markdown('## CONTENTS')
    table(pd.DataFrame({
        '': ['01', '02', '03', '04', '05', 'DATA DESK', 'NOTES'],
        '제목': ['계절이라는 착시', '옷장의 달력', '날씨에 솔직한 옷들', '추위를 따르지 않는 옷들', '틀린 가설과 한계',
                '전처리: 함정, 파생 변수, 계절 제거', '한계와 다음 단계']}))
    st.markdown('## 두 가지 원칙')
    st.markdown('1. 계절에 의한 가짜 상관을 피하기 위해, 모든 값을 **"평소 수준 대비"** 로 본다.\n'
                '2. 분석 기준은 **실제 날씨(관측)**, 모델 입력 후보는 **예보**로 구분해 "그날 실제로 알 수 있었던 정보"의 오차를 방어한다.')
    nextup('01 · 계절이라는 착시')

# ======================================================================
# 01
# ======================================================================
def page_ch01():
    chapter('CHAPTER 01', '계절이라는 착시', '"추워서 샀다"와 "겨울이라서 샀다"는 다른 이야기다')
    st.markdown('장갑 클릭과 최저기온을 그대로 비교하면 상관이 -0.71입니다. 하지만 겨울에는 원래 춥고, 원래 장갑을 많이 찾습니다. '
                '아이스크림 판매량과 물놀이 사고가 둘 다 여름에 늘어 같이 움직이는 것과 같은 착시입니다.')
    st.markdown('그래서 모든 값에서 **그 시기의 평소 수준(앞뒤 31일 평균)** 을 빼고, "평소보다 얼마나 추웠고, 평소보다 얼마나 더 찾았나"만 남겼습니다.')
    table(pd.DataFrame({
        '비교': ['장갑 × 최저기온', '수영 × 최고기온', '우산 × 최저기온', '우산 × 비'],
        '그대로': [-0.71, 0.63, 0.66, 0.49],
        '평소 수준을 뺀 뒤': [-0.46, 0.15, 0.16, 0.62]}))
    pull('수영과 우산의 "기온 반응"은 대부분 여름이라서였습니다. 반대로 비는 계절이 아니라 그날의 사건이라서, 계절을 빼자 오히려 선명해졌습니다. 이후 모든 숫자는 "평소 대비"입니다.')

    st.markdown('### 쇼핑 클릭은 해마다 줄었다')
    fm = read('fashion_monthly.csv', index_col='월')
    fig = go.Figure()
    for yr, col in zip(fm.columns, ['#2B2B2B', '#8E2C2C', '#C0502D', '#D9A441']):
        fig.add_trace(go.Scatter(x=fm.index, y=fm[yr], mode='lines+markers', name=f'{yr}년', line=dict(color=col)))
    fig.update_layout(title='패션 전체 클릭지수 (월평균, 2026년은 7월까지)')
    fig.update_xaxes(title_text='월', tickmode='linear', dtick=1)
    fig.update_yaxes(title_text='클릭지수 (월평균)')
    st.plotly_chart(style(fig, 400), use_container_width=True)
    st.markdown('같은 달끼리 비교해도 해마다 내려갑니다(2026년 1~7월은 2023년의 약 0.22배). 날씨와 상관없는 플랫폼 전체의 변화라서, '
                '연도를 넘어 숫자를 직접 비교하지 않고 **평소 수준과의 차이**를 중심으로 봤습니다.')
    nextup('02 · 옷장의 달력')

# ======================================================================
# 02
# ======================================================================
def page_ch02():
    chapter('CHAPTER 02', '옷장의 달력', '4개의 시즌으로 나뉜 30개의 키워드')
    sh = read('seasonal_heatmap.csv', index_col='키워드')
    sh = sh.drop(index=[k for k in ANCHORS if k in sh.index])
    peak = sh.idxmax(axis=1).str.replace('월', '').astype(int)       # 키워드별 가장 많이 찾는 달
    sh = sh.loc[peak.sort_values(kind='stable').index]                # 같은 계절 상품이 이웃하게 정렬
    fig = px.imshow(sh, zmin=0, zmax=1, aspect='auto',
                    color_continuous_scale=[[0, '#FBF3E6'], [0.4, '#E8B88A'], [1, '#7A1F2B']],
                    labels=dict(x='월', y='키워드 (가장 많이 찾는 달 순)', color='최고 달 대비'))
    fig.update_layout(title='키워드별 월 계절성 (각 키워드의 최고 달 = 1.0)')
    st.plotly_chart(style(fig, 720), use_container_width=True)
    st.caption('한 줄(키워드) 안에서만 비교합니다. 월마다 3~4개 관측의 평균이며 기준용 키워드 4종은 제외했습니다.')
    table(pd.DataFrame({
        '무리': ['겨울', '여름', '환절기', '가을 상의'],
        '대표 키워드': ['롱패딩·패딩·장갑·목도리·어그', '반바지·민소매·샌들·래쉬가드', '블레이저·트렌치코트·가죽자켓·야상', '가디건·맨투맨·후드티'],
        '피크': ['11~12월 (1월도 높음)', '6~8월', '3월과 10월, 쌍봉', '10월']}))
    pull('환절기 무리의 쌍봉은 이 리포트의 복선입니다. 이 키워드들은 다음 장에서 날씨와 거의 상관이 없는 쪽에 서게 됩니다.')
    nextup('03 · 날씨에 솔직한 옷들')

# ======================================================================
# 03
# ======================================================================
def page_ch03():
    chapter('CHAPTER 03', '날씨에 솔직한 옷들', '롱패딩이 선두, 그 뒤를 어그와 장갑이')
    st.markdown('14개 큰 카테고리로 봤을 때는 장갑과 우산 정도만 뚜렷했습니다. 카테고리 안에 여러 품목이 섞여 신호가 희석됐기 때문입니다. '
                '30개 세부 키워드로 내려가자 방한 용품이 줄지어 선명해졌습니다.')
    s3 = read('screen_t3.csv', index_col=0)
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
    fig = px.bar(r3, x='t_min', y='키워드', orientation='h', color='구분',
                 category_orders={'키워드': r3['키워드'].tolist()},
                 color_discrete_map={'추울수록 증가': COLD, '더울수록 증가': HEAT, '약한 상관 (|r| 0.2 미만)': '#D8D2C8', '비에 반응 (기온은 해석 안 함)': GRAY},
                 labels={'t_min': '평소 대비 최저기온과의 상관 (성수기, 31일 평균 제거 + 요일 보정)'})
    fig.update_yaxes(autorange='reversed', title_text='키워드')
    fig.update_layout(title='키워드별 기온 반응')
    st.plotly_chart(style(fig, 760), use_container_width=True)
    st.caption('30개 키워드 × 5개 날씨 변수를 훑은 탐색이라 |r| 0.2 미만은 우연일 수 있습니다. 우산·레인부츠는 강수가 최강 변수라 기온 막대를 해석하지 않습니다.')

    st.markdown('### 얼마나 늘어날까')
    st.markdown('평소보다 기온이 1℃ 낮을 때 클릭이 늘어난 정도입니다(앞뒤 31일 평균 기준).')
    table(pd.DataFrame({'키워드': ['롱패딩', '장갑', '목도리', '패딩', '어그', '부츠'],
                        '1℃ 추울 때': ['약 +9.0%', '약 +7.7%', '약 +4.8%', '약 +4.6%', '약 +4.0%', '약 +3.4%']}))
    pull('평소보다 5℃ 추운 날, 롱패딩 클릭은 평소의 1.5배 안팎까지 올라갑니다. 다만 이 퍼센트는 "평소 수준"을 계산하는 방식에 따라 달라질 수 있어 범위로 읽어야 합니다.')

    st.markdown('### 직접 골라 보기')
    kws = [k for k in s3.index if k not in ANCHORS]
    c1, c2, c3 = st.columns(3)
    kw = c1.selectbox('키워드', kws, index=kws.index('롱패딩') if '롱패딩' in kws else 0, key='exp_kw')
    var = c2.radio('날씨 변수', ['최저기온', '최고기온', '강수량'], key='exp_var')
    only_peak = c3.checkbox('성수기만 보기', value=True, key='exp_peak')

    wd = read('weather_daily.csv', index_col='date', parse_dates=True)
    t3 = read('demand_t3.csv', index_col='date', parse_dates=True)
    wvar = {'최저기온': wd['t_min'], '최고기온': wd['t_max'], '강수량': np.log1p(wd['precip_mm'])}[var]
    unit = 'log(1+mm)' if var == '강수량' else '℃'
    months = [int(m) for m in str(s3.loc[kw, '성수기 월']).split(',')]

    y = short_term(log_level(t3[kw]))
    y = y - y.groupby(y.index.dayofweek).transform('mean')       # 요일 보정
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
        st.markdown(f'**평소보다 1℃ 추우면 {kw} 클릭이 약 {abs(pct):.1f}% {"늘어나는" if pct > 0 else "줄어드는"} 경향**이 있습니다 (31일 평균 제거 기준).')
    st.caption(f'성수기 월: {", ".join(map(str, months)) if only_peak else "전체"}. 같은 겨울 안의 날들은 서로 비슷하게 움직여 점이 독립이 아닙니다(성수기 307일은 사실상 겨울 3~4번의 정보량).')

    st.markdown('### 얼마나 빨리 반응할까')
    st.markdown('- 장갑은 추워진 **당일에 가장 크게**(-0.57) 반응했고 **다음 날까지 일부 이어졌습니다**(-0.32).\n'
                '- 우산은 **비 온 당일에만** 반응했습니다(+0.63). 어제 내린 비가 오늘 우산을 끌지 않습니다.\n'
                '- 날씨를 미리 보고 사 두는 선행 반응은 이 방법으로는 보이지 않았습니다.\n'
                '- 같은 키워드 25개 중 22개(88%)에서 검색과 쇼핑 클릭이 같은 방향으로 움직였고, 둘은 **같은 날** 가장 강하게 함께 움직였습니다. 검색이 클릭을 하루 앞선다는 근거는 찾지 못했습니다.')
    nextup('04 · 추위를 따르지 않는 옷들')

# ======================================================================
# 04
# ======================================================================
def page_ch04():
    chapter('CHAPTER 04', '추위를 따르지 않는 옷들', '모든 겨울옷이 추위에 반응하지는 않는다')
    table(pd.DataFrame({
        '키워드': ['롱패딩', '패딩', '어그', '장갑', '목도리', '코트', '니트', '가디건', '맨투맨'],
        '최저기온과의 상관': [-0.85, -0.74, -0.70, -0.68, -0.65, -0.09, -0.08, -0.09, -0.06],
        '반응': ['솔직'] * 5 + ['무심'] * 4}))
    st.markdown('같은 겨울옷인데 패딩류와는 완전히 다른 모양입니다. 성수기 기준을 30~60%로, 로그 보정값을 0.5~10%로 바꿔도 코트는 계속 0 근처(-0.09 ~ 0.01)였습니다. '
                '결과가 분석 선택의 부산물이 아니라는 뜻입니다.')
    pull('패딩은 "오늘 추워서" 사고, 코트는 "겨울이 왔으니" 산다는 가설이 가능합니다. 달력이 정하는 옷과 날씨가 정하는 옷이 다르다는 것입니다. (데이터로 확인한 건 아니고 모양에서 나온 추측입니다.)')

    st.markdown('### 비 오는 날의 우산')
    table(pd.DataFrame({'키워드': ['우산', '레인부츠', '레인코트 (검색)', '방수자켓 (검색)'], '비와의 상관': ['+0.63 ~ +0.68', '+0.62', '+0.67', '+0.59']}))
    st.markdown('비 관련 상품은 비 온 당일에 솟고 다음 날에는 사라집니다. 우산은 일교차와도 강한 음의 상관(-0.55)을 보였는데, 비 오는 날은 구름 때문에 일교차가 작아지는 것으로 보입니다. '
                '우산이 일교차에 반응한 게 아니라 **비의 그림자가 일교차에 비친 것**일 가능성이 큽니다(확인하지는 못했습니다).')
    nextup('05 · 틀린 가설과 한계')

# ======================================================================
# 05
# ======================================================================
def page_ch05():
    chapter('CHAPTER 05', '틀린 가설과 한계', '이 프로젝트에서 가장 많이 배운 장')
    st.markdown('처음 세운 가설은 이랬습니다. *소비자는 절대 기온보다 체감온도, 일교차, 전일 대비 변화에 더 반응한다.* 결과는 이렇습니다.')
    table(pd.DataFrame({
        '가설': ['계절을 빼야 진짜 효과가 보인다', '방한·여름 상품은 기온에 반응한다', '전일 대비 기온 변화에 반응한다',
               '체감온도가 기온보다 더 잘 설명한다', '일교차에 반응한다', '내일 예보로 추천할 수 있다'],
        '결과': ['✅', '✅', '❌', '❌', '△', '❌'],
        '근거': ['장갑 -0.71 → -0.46', '롱패딩 -0.85, 반바지 +0.48', '평균 점수 0.03 (3일 전과 비교하면 0.10)',
               '5개 키워드 모두 기온이 약간 우세 (차이 -0.02 ~ -0.03)', '우산류에서만 강하고 비의 대리 변수로 보임', '가장 가까운 예보가 3~4일 앞']}))
    pull('사람들은 변화의 크기가 아니라 "평소보다 춥다는 사실"에 반응한 것으로 보입니다. (추측) 가설이 틀렸다는 결과를 숨기지 않고 쓰는 것이 이 프로젝트에서 가장 중시한 습관입니다.')

    st.markdown('### 몇 도부터 한파인가')
    st.markdown('기상청 특보를 **정답지**로 두고, 우리가 정한 기준이 얼마나 맞는지 확인했습니다. **정밀도**는 규칙이 "한파"라고 한 날 중 실제 특보가 있던 비율, '
                '**재현율**은 실제 특보 중 규칙이 잡아낸 비율입니다.')
    ev = read('event_threshold.csv')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=('한파 (최저기온 ≤ 임계값)', '폭염 (최고기온 ≥ 임계값)'))
    for col, name in [(1, '한파'), (2, '폭염')]:
        sub = ev[ev['이벤트'] == name]
        fig.add_trace(go.Scatter(x=sub['임계값'], y=sub['정밀도'], mode='lines+markers', name='정밀도', line=dict(color=COLD), showlegend=(col == 1)), row=1, col=col)
        fig.add_trace(go.Scatter(x=sub['임계값'], y=sub['재현율'], mode='lines+markers', name='재현율', line=dict(color=HEAT), showlegend=(col == 1)), row=1, col=col)
        ch = sub[sub['선택'].eq('✔')]
        fig.add_trace(go.Scatter(x=ch['임계값'], y=ch['정밀도'], mode='markers', name='선택한 임계값',
                                 marker=dict(symbol='star', size=16, color=BURGUNDY), showlegend=(col == 1)), row=1, col=col)
    fig.update_xaxes(title_text='임계값 (℃, 오른쪽일수록 완화)')
    fig.update_yaxes(title_text='특보와의 일치 비율 (±2일 허용)', col=1)
    st.plotly_chart(style(fig, 420), use_container_width=True)
    table(ev.drop(columns='선택'))
    st.markdown('한파는 원안(-12℃)이 3년 7개월 동안 4번뿐이고 2024-25 겨울에는 한 번도 없어서, 정밀도를 잃지 않고 9건이 잡히는 **-10℃** 로 정했습니다. '
                '폭염은 원안인 **33℃**(14건, 정밀도 1.00)를 그대로 썼습니다. 특보는 서울 4개 구역 중 한 곳만 해당돼도 발표되기 때문에 규칙 이벤트보다 자주 나옵니다(한파 20건, 폭염 20건). '
                '어느 쪽이 틀린 게 아니라 기준이 다릅니다.')

    st.markdown('### 내일은 없다')
    fc = read('forecast_error.csv', index_col='date', parse_dates=True)
    fc['시기'] = np.where(fc['lead_days'] == 3, 'A: 3일 앞', 'B: 4일 이상 앞')
    rows = [{'시기': p, '일수': len(g), '최저기온 평균 오차 (℃)': (g['t_min'] - g['t_min_obs']).abs().mean(),
             '최고기온 평균 오차 (℃)': (g['t_max'] - g['t_max_obs']).abs().mean()} for p, g in fc.groupby('시기')]
    st.markdown('기상청 중기예보는 **3일 뒤부터** 발표합니다(2024-11-29 발표분부터 4일 뒤). 이 데이터에 "내일 예보"는 없습니다.')
    table(pd.DataFrame(rows).round(2))
    which = st.radio('기온', ['최저기온', '최고기온'], horizontal=True, key='fc_which')
    obs, pred = ('t_min_obs', 't_min') if which == '최저기온' else ('t_max_obs', 't_max')
    sp = px.scatter(fc, x=obs, y=pred, color='시기', opacity=0.45,
                    color_discrete_map={'A: 3일 앞': COLD, 'B: 4일 이상 앞': HEAT},
                    labels={obs: f'관측 {which} (℃)', pred: f'예보 {which} (℃)'})
    lo, hi = float(fc[[obs, pred]].min().min()), float(fc[[obs, pred]].max().max())
    sp.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='예보 = 관측', line=dict(color=INK)))
    sp.update_layout(title=f'{which}: 예보 vs 관측 (검은 선에 가까울수록 정확)')
    st.plotly_chart(style(sp, 440), use_container_width=True)
    pull('예보 오차(1.1~1.8℃)는 방한 용품이 1℃ 추울 때 늘어나는 정도(약 3~9%)와 비슷한 크기입니다. 그래서 추천 모델은 이 리포트에서 다루지 않았습니다.')
    nextup('DATA DESK · 전처리')

# ======================================================================
# DATA DESK
# ======================================================================
def page_datadesk():
    chapter('DATA DESK', '전처리', '이 리포트의 숫자들이 통과한 작업: 무엇을 했나 → 왜 그랬나 → 어떻게 검증했나')
    wd = read('weather_daily.csv', index_col='date', parse_dates=True)

    st.markdown('## 함정 1 · 결측 표기 (-9, -99)')
    st.markdown('기상청은 결측을 -9, -99로 적지만, 서울 겨울에는 **-9.0℃가 진짜 기온**입니다.')
    table(pd.DataFrame({'컬럼 종류': ['기온', '습도·풍속·운량', '강수·적설'],
                        '처리': ['-99만 결측', '-9, -99 모두 결측', '-9는 0, -99는 결측'],
                        '이유': ['-9℃는 실제로 존재하는 값', '음수가 될 수 없는 값', '-9는 결측이 아니라 "없음"']}))
    c1, c2, c3 = st.columns(3)
    c1.metric('관측 일수', f'{len(wd):,}일')
    c2.metric('무강수(0mm)', f'{int((wd["precip_mm"] == 0).sum()):,}일')
    c3.metric('최저기온 -9℃인 날', f'{int((wd["t_min"] == -9).sum())}일')
    st.markdown('**검증**: 무강수 915일 = 원본 `-9` 표기 754일 + 원본 `0.0` 기록 161일. 빠진 날 0, 최저>최고 0. 풍속 결측 5건(연속 최대 2일)은 선형 보간했고, '
                '연속 3일을 넘으면 값을 지어내는 셈이라 보간하지 않는 규칙(`limit=3`)입니다.')

    st.markdown('## 함정 2 · 상대 숫자')
    st.markdown('네이버 지수는 "같이 요청한 묶음에서 가장 큰 값 = 100"인 상대값이라 묶음마다 크기가 달라 비교가 무효가 됩니다.\n\n'
                '- 모든 묶음에 **앵커(기준) 키워드**를 넣어 수집했고, 앵커 값의 묶음 간 배율 변동계수(CV)가 **모든 묶음에서 0** → 차이는 날짜와 무관한 상수 하나였습니다. 상수를 곱해 정렬했습니다(정렬 후 편차 ≤ 6.3e-5). 정렬 전 배율은 최대 약 10배 차이였습니다.\n'
                '- 저장돼 있던 `rescaled`(= 같은 날 앵커로 나눈 값)는 쓰지 않았습니다. 앵커의 계절성·노이즈가 모든 키워드에 거꾸로 주입되기 때문입니다.')

    st.markdown('## 함정 3 · 해마다 줄어드는 클릭')
    table(pd.DataFrame({'연도': ['2023', '2024', '2025', '2026 (7월까지)'], '연평균': [50.3, 35.6, 21.9, 11.8],
                        '같은 달(1~7월), 2023 = 1.0': [1.00, 0.68, 0.47, 0.22]}))
    st.markdown('개별 카테고리도 0.16~0.20으로 비슷하게 내려가 플랫폼 전체의 공통 요인입니다. 그대로 두면 "시간이 갈수록 수요가 줄었다"는 흐름이 날씨 효과와 섞입니다. '
                '그래서 **점유율**(카테고리 ÷ 패션 전체)로 바꿔 공통 하락을 상쇄하고(점유율은 패션 전체와 같이 움직이는 요인도 지우므로 원 지수도 병행), 31일 평균 제거가 느린 추세도 함께 흡수하게 했습니다.')

    st.markdown('## 함정 4 · 희소 키워드')
    st.markdown('- 커버리지 95% 미만 키워드 7개 중 6개(버킷햇·선글라스·방수자켓·히트텍·레인코트·아쿠아슈즈)는 같은 키워드의 **검색량이 1,308일 전부 채워져** 있었습니다. '
                '조회한 카테고리가 실제 분류와 달라 클릭이 안 잡힌 것으로 판단하고(정황 근거) 검색량으로 대체했습니다. 기모레깅스는 클릭 0.146, 검색 0.875라 제외했습니다.\n'
                '- 나머지 키워드의 빠진 날은 비수기에 몰려 있어서(롱패딩 4~7월 등) 0으로 채웠고, 로그를 위해 양수 값의 하위 1%를 더했습니다. 0.5~10%로 바꿔도 결과는 거의 같았습니다.')

    st.markdown('## 예보와 특보')
    st.markdown('- **예보**: 대상일마다 "발표일이 대상일보다 앞선 발표분 중 가장 최신"을 골랐습니다("그날 실제로 알 수 있었던 정보"). 육상 예보는 가까운 날짜가 오전·오후 두 줄이라 하루 한 줄로 접고, 행 수가 기온 예보(19,096)와 일치함을 확인한 뒤 병합했습니다.\n'
                '- **특보**: 해제 시각 컬럼이 전부 결측이라 발표·해제 행을 권역별로 직접 짝지어 발효 구간을 조립했습니다. 한파는 발표 69 = 해제 69, 폭염은 발표 77 대비 해제 73이었는데 4개 권역이 모두 2026-07-23~24에 발표된 뒤 데이터 끝까지 진행 중이어서 종료일을 분석 기간 끝으로 닫았습니다. 4개 권역은 하나라도 발효 중이면 서울 특보일(any)로 집계했습니다.')

    st.markdown('## 파생 변수')
    table(pd.DataFrame({
        '변수': ['dtr 일교차', 'dt_min, dt_max', 'dt_min_3d', 'feels_like', '평소 대비 편차', 'log_precip', '특보 일별 컬럼', 'lead_days'],
        '정의': ['최고 − 최저', '전일 대비 변화', '3일 전 대비 변화', '겨울 체감온도 (평균기온 ≤ 10℃, 풍속 ≥ 4.8km/h일 때 공식, 그 외 평균기온)',
               '값 − 앞뒤 31일 평균', 'log(1 + 강수량)', '발효 여부, 그날 최고 등급, 연속 며칠째', '발표일 → 대상일 일수'],
        '왜 만들었나': ['"일교차에 반응" 가설 검증. 일교차는 자체 계절성이 있어 비교 전에 계절을 빼야 함', '"전일 변화에 반응" 가설의 실체',
                    '전일 변화가 약할 때 "하루 단위라서인가"를 가리는 대조 변수', '"체감온도가 더 잘 설명" 가설 검증 (풍속 m/s를 km/h로 환산)',
                    '계절과 장기 하락 제거 (수요·날씨 양쪽에 같은 방식)', '0mm인 날이 많아 log(0)이 불가능하고, 극단값(최대 128.8mm)이 상관을 지배하지 않게 함',
                    '이벤트 기준을 채점할 정답지', '"내일 예보"의 존재 여부와 예보 방식 변경 시점(2024-11-29) 확인']}))

    st.markdown('## 계절 제거와 내가 정한 값들')
    st.markdown('계절 제거는 **로그 → 31일 중심 이동평균 빼기 → 요일 평균 빼기** 순서입니다. 요일 효과는 우산·장갑 등 계절 상품에서 로그 단위로 약 10%였지만, '
                '보정해도 핵심 결과(장갑 -0.79 → -0.81, 우산 0.64 → 0.67)는 거의 달라지지 않아 이후 기본 보정으로 썼습니다.')
    table(pd.DataFrame({
        '값': ['평소 수준 창', '성수기 기준', '로그 바닥값', '특보 대조 허용 오차', '"며칠 전 대비" 길이'],
        '선택': ['31일', '최고 달의 40% 이상', '하위 1%', '±2일', '3일'],
        '점검': ['15·31·61일: 방향은 같고 크기는 달라짐 (장갑 -0.38 / -0.46 / -0.45)',
               '30~60%: 롱패딩·목도리·우산은 안정, 장갑(-0.68 → -0.86)·반바지는 크기가 변함, 코트는 어떤 기준에서도 약함',
               '0.5~10%: 거의 영향 없음', '점검하지 않음', '점검하지 않음']}))
    nextup('NOTES · 한계와 다음 단계')

# ======================================================================
# NOTES
# ======================================================================
def page_notes():
    chapter('NOTES', '한계와 다음 단계', '이 리포트가 말할 수 없는 것')
    st.markdown('## 한계')
    st.markdown('- **상관은 원인이 아닙니다.** 같이 움직인다는 것까지만 확인했습니다.\n'
                '- **완전한 겨울은 3번**(2023-24, 2024-25, 2025-26)이고 2023년은 1~3월만 있습니다. 같은 겨울 안의 날들은 서로 비슷하게 움직여서 성수기 307일은 사실상 겨울 3~4번의 정보량입니다. '
                '겨울별로 구한 장갑×최저기온이 모두 음수(-0.53, -0.88, -0.71, -0.82)였지만, 정확한 크기보다 **방향과 순서**를 근거로 삼아 주세요.\n'
                '- 평소 수준을 **단순한 31일 이동평균**으로 구했습니다. 더 정교한 분해와 비교하면 겨울 용품의 효과 크기가 달라질 수 있어서, 1℃당 +9.0%는 범위로 읽어야 합니다. 이 비교는 직접 하지 못했습니다.\n'
                '- 30개 키워드 × 5개 날씨 변수를 훑은 탐색이라 점수 0.2 미만은 우연일 수 있습니다.\n'
                '- 한파·폭염 시작일 전후로 수요가 며칠간 어떻게 변하는지는 계산하지 못했습니다.\n'
                '- 체감온도는 일 평균 풍속·기온으로 계산한 겨울 공식뿐이고 여름 체감은 만들지 않았습니다.')
    st.markdown('**확인하지 못한 해석들**: 일교차가 강수의 그림자라는 것, 환절기 옷이 달력 중심이라는 것, 폭염 규칙의 재현율이 낮은 이유, 선케어의 클릭과 검색이 다르게 움직인 이유.')

    st.markdown('## 이번 리포트에서 제외한 것')
    table(pd.DataFrame({
        '항목': ['선케어', '수영', '기모레깅스', '기준용 키워드 4종', '추천 모델'],
        '결정': ['제외 (다른 주제와 함께 따로 분석)', '판단 보류', '제외', '순위 해석에서 제외', '제외'],
        '이유': ['옷차림과 직접 관련이 없음. 클릭은 4~7월 내내 꾸준히 높고 날씨와 연결이 약했는데(0.30~0.40), 검색은 더운 날과 강하게 연결됨(+0.50)',
               '성수기 기준에 따라 점수가 0.03~0.27로 흔들림', '쇼핑 클릭 커버리지 0.146, 검색도 0.875',
               '묶음 크기를 맞추려고 넣은 기준 키워드', '"내일 예보"가 없고, 예보 오차가 날씨 효과와 비슷하며, 정답(라벨) 정의가 남아 있음']}))

    st.markdown('## 활용 아이디어 (제안, 분석에서 곧바로 도출되는 결론은 아님)')
    st.markdown('한파·비·폭염 예보 시점에 해당 상품군의 노출·재고를 조정하고, 환절기 품목은 날씨보다 시즌 전환 일정에 맞춰 운영한다.')

    st.markdown('## 다음 단계')
    st.markdown('1. 코로나 이전(2017~2019)을 더해 겨울·여름 횟수 늘리기\n'
                '2. 겨울 용품에 MSTL을 적용해 기준선별 효과 크기 비교하기\n'
                '3. 한파·폭염 시작일 기준 D-7 ~ D+14 수요 곡선\n'
                '4. 패션 전체 트래픽을 통제한 날씨 효과\n'
                '5. 단기예보를 더해 "내일 추천" 다시 시도하기\n'
                '6. 선케어를 다른 주제와 함께 별도 리포트로 분석하기')

PAGES = {
    'COVER': page_cover,
    '01 · 계절이라는 착시': page_ch01,
    '02 · 옷장의 달력': page_ch02,
    '03 · 날씨에 솔직한 옷들': page_ch03,
    '04 · 추위를 따르지 않는 옷들': page_ch04,
    '05 · 틀린 가설과 한계': page_ch05,
    'DATA DESK · 전처리': page_datadesk,
    'NOTES · 한계와 다음 단계': page_notes,
}

st.sidebar.markdown('### WEATHER_FIT')
st.sidebar.caption('Seoul · Weather & Wardrobe')
menu = st.sidebar.radio('목차', list(PAGES), label_visibility='collapsed')
PAGES[menu]()