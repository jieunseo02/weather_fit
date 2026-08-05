# weather_fit

# 날씨 × 패션 트렌드 분석

> 서울의 기상 데이터와 패션 카테고리 검색·콘텐츠 트렌드를 결합해, **(1) 기상 이벤트별 의류 카테고리 수요 분석 리포트**와 **(2) "내일 날씨 → 추천 카테고리" 분류 모델**을 제작하는 데이터 분석 프로젝트.

[![Status](https://img.shields.io/badge/status-WIP-yellow)]() [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

---


## 데이터 소스

- **기상**: 기상청 동네예보·종관기상관측 (서울, 부산, 대전, 대구, 광주, 강원 영서/영동)
- **쇼핑 트렌드**: 네이버 데이터랩 쇼핑인사이트, 네이버 쇼핑 API, 카카오 쇼핑하우 API
- **정성 콘텐츠**: 무신사 매거진, 인스타 #ootd 해시태그, 블로그 패션 키워드

기간: **2023-01-01 ~ 현재** (3년 백필, 코로나 영향 제외)

---

## 기술 스택

- 데이터: Python 3.10, Pandas, Polars, Parquet, DuckDB (미정)
- 분석: statsmodels (STL), scikit-learn (추후 추가예정)
- ML: LightGBM, Logistic Regression (미정)
- 시각화: Matplotlib, Seaborn, Plotly, Streamlit (미정)
- 운영: 로컬 cron

---

## 산출물

- **분석 리포트** ([reports/](reports/)) — 기상 이벤트별 카테고리 인사이트
- **분류 모델** ([src/model/](src/model/)) — 내일 날씨 → 추천 카테고리
- **대시보드** ([dashboard/](dashboard/)) — Streamlit 기반 일일 추천 + 트렌드 모니터링
- **EDA 노트북** ([notebooks/](notebooks/)) — 시계열 분해, 이벤트 케이스 스터디

