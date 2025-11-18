import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re # 행정동 파싱용 정규식
# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="CSC - 평택시 대기질 리스크 & 노인복지시설 분석",
    layout="wide"
)

# 사용자 PC에 저장된 데이터 폴더
BASE_DIR = (Path(__file__).parent / "data").resolve()

# 대기오염 항목별 컬럼명 매핑
POLLUTANT_COLS = {
    "SO2": "이산화황측정값(ppm)",
    "NO2": "이산화질소측정값(ppm)",
    "CO": "일산화탄소측정값(ppm)",
    "O3": "오존측정값(ppm)",
    "PM10": "PM10측정값(㎍/㎥)",
    "PM2.5": "PM25측정값(㎍/㎥)",
}

POLLUTANT_LABELS = {
    "SO2": "SO₂(이산화황)",
    "NO2": "NO₂(이산화질소)",
    "CO": "CO(일산화탄소)",
    "O3": "O₃(오존)",
    "PM10": "PM10(미세먼지)",
    "PM2.5": "PM2.5(초미세먼지)",
}

GRADE_TO_SCORE = {"좋음": 1, "보통": 2, "나쁨": 3, "매우나쁨": 4}

# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------
@st.cache_data
def read_csv_safely(path: Path) -> pd.DataFrame:
    """인코딩을 자동으로 맞춰서 CSV 읽기."""
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="ignore")


@st.cache_data
def load_data():
    """프로젝트에 사용하는 모든 데이터 한 번에 로드."""
    air = read_csv_safely(BASE_DIR / "경기도대기환경정보월평균자료.csv")
    grade = read_csv_safely(BASE_DIR / "경기도_대기환경정보항목별지수등급.csv")
    region = read_csv_safely(BASE_DIR / "경기도_대기환경_진단평가시스템_지역정보.csv")
    elderly = read_csv_safely(BASE_DIR / "경기도_평택시_노인복지시설_20250129_(1).csv")
    chem = read_csv_safely(BASE_DIR / "경기도_평택시_유해화학물질_취급사업장_현황_20250207.csv")
    return {
        "air": air,
        "grade": grade,
        "region": region,
        "elderly": elderly,
        "chem": chem,
    }


def add_air_quality_grades(df_air: pd.DataFrame,
                           df_grade: pd.DataFrame) -> pd.DataFrame:
    """대기질 월평균 데이터에 항목별 등급/점수/종합위험점수 추가."""
    df = df_air.copy()

    # 측정일자(YYYYMM)를 실제 날짜(매월 1일)로 변환
    df["측정일"] = pd.to_datetime(df["측정일자"].astype(str), format="%Y%m")

    # 항목별 등급 기준 (항목명 기준으로 중복 제거)
    grade_info = df_grade.drop_duplicates("항목명", keep="first").set_index("항목명")

    def calc_grade(value: float, standard_row: pd.Series) -> str:
        if pd.isna(value):
            return np.nan
        if value <= standard_row["좋음기준"]:
            return "좋음"
        elif value <= standard_row["보통기준"]:
            return "보통"
        elif value <= standard_row["나쁨기준"]:
            return "나쁨"
        else:
            return "매우나쁨"

    # 오염물질별 등급/점수 계산
    for pollutant, col_name in POLLUTANT_COLS.items():
        thresholds = grade_info.loc[pollutant]
        grade_col = f"{pollutant}_등급"
        score_col = f"{pollutant}_점수"

        df[grade_col] = df[col_name].apply(lambda v: calc_grade(v, thresholds))
        df[score_col] = df[grade_col].map(GRADE_TO_SCORE)

    # 종합위험점수: 6개 항목 점수 중 최댓값(=가장 나쁜 등급)
    score_cols = [c for c in df.columns if c.endswith("_점수")]
    df["종합위험점수"] = df[score_cols].max(axis=1)

    return df


def make_city_summary(df_air_scored: pd.DataFrame) -> pd.DataFrame:
    """도시별 평균 농도 / 평균 종합위험점수 요약."""
    agg_cols = {
        "이산화황측정값(ppm)": "mean",
        "이산화질소측정값(ppm)": "mean",
        "일산화탄소측정값(ppm)": "mean",
        "오존측정값(ppm)": "mean",
        "PM10측정값(㎍/㎥)": "mean",
        "PM25측정값(㎍/㎥)": "mean",
        "종합위험점수": "mean",
    }
    city_summary = (
        df_air_scored
        .groupby("도시명")
        .agg(agg_cols)
        .rename_axis("도시명")
        .reset_index()
    )
    return city_summary


# ★ 수정: 도로명주소에서 행정 읍·면·동만 뽑는 함수 (건물동 필터링 포함)
def extract_eupmyeondong(addr: str) -> str:
    if pd.isna(addr):
        return np.nan

    text = str(addr)
    # 공백, 쉼표, 괄호 기준으로 토큰 분리
    tokens = re.split(r"[ ,()]", text)

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # 읍/면/동으로 끝나지 않으면 패스
        if not tok.endswith(("읍", "면", "동")):
            continue

        # 행정구역이 될 수 없는 것들 제거
        if tok in ("경기도", "평택시"):
            continue

        # ① 숫자 + 동 (1동, 103동, 101동 등) → 건물동
        if re.fullmatch(r"\d+동", tok):
            continue

        # ② 영문/숫자 코드 + 동 (A동, B동, S001동 등) → 건물동
        if re.fullmatch(r"[A-Za-z0-9]+동", tok):
            continue

        # ③ 제1동, 제2동 형태 → 건물동
        if re.fullmatch(r"제\d+동", tok):
            continue

        # ④ 상가동 관련 (상가동, 상가2동, 참이슬아파트상가동 등) → 건물동
        if "상가동" in tok or (tok.startswith("상가") and tok.endswith("동")):
            continue

        # ⑤ 한 글자 + 동 (가동, 나동 등) → 건물동으로 간주
        if len(tok) == 2 and tok.endswith("동"):
            continue

        # 위 조건을 다 통과한 첫 번째 토큰을 행정읍면동으로 사용
        return tok

    return np.nan


# ------------------------------------------------------------
# 메인 대시보드
# ------------------------------------------------------------
def main():
    st.title("CSC 프로젝트 - 공공 ESG 관점 평택시 대기질 리스크 & 노인복지시설 분석")
    st.caption("데이터 경로: C:/Users/yomy0/Documents/2025 하반기/CSC")

    # 데이터 로드
    data = load_data()
    df_air_raw = data["air"]
    df_grade = data["grade"]
    df_region = data["region"]
    df_elderly = data["elderly"]
    df_chem = data["chem"]

    # 전처리 (대기질 등급/위험점수 계산)
    df_air = add_air_quality_grades(df_air_raw, df_grade)
    city_summary = make_city_summary(df_air)

    # 평택시/경기도 평균 위험 점수
    pyeongtaek_row = city_summary[city_summary["도시명"] == "평택시"].iloc[0]
    gyeonggi_mean_risk = city_summary["종합위험점수"].mean()
    pyeongtaek_risk = pyeongtaek_row["종합위험점수"]

    # 평택시 기초 정보 (대기환경 진단평가시스템)
    region_row = df_region[df_region["시군구명"] == "평택시"].iloc[0]

    # ★ 추가: 평택시 내부 읍·면·동 단위 '위험지수' 계산
    # ★ 노인복지시설: 도로명주소만 사용
    df_elderly["행정동"] = df_elderly["도로명주소"].apply(extract_eupmyeondong)

    # ★ 유해화학물질 사업장: 도로명주소 → 안 나오면 지번주소로 보완
    df_chem["행정동"] = df_chem["소재지도로명주소"].apply(extract_eupmyeondong)
    mask_na = df_chem["행정동"].isna()
    if "소재지지번주소" in df_chem.columns:
        df_chem.loc[mask_na, "행정동"] = df_chem.loc[mask_na, "소재지지번주소"].apply(
            extract_eupmyeondong
        )

    elderly_cnt = (
        df_elderly.groupby("행정동")
        .size()
        .rename("노인복지시설_수")
    )
    chem_cnt = (
        df_chem.groupby("행정동")
        .size()
        .rename("유해화학사업장_수")
    )

    local_risk = pd.concat([elderly_cnt, chem_cnt], axis=1).fillna(0)
    local_risk["노인복지시설_수"] = local_risk["노인복지시설_수"].astype(int)
    local_risk["유해화학사업장_수"] = local_risk["유해화학사업장_수"].astype(int)
    # 단순 위험지수: 유해화학사업장 수 / (노인복지시설 수 + 1)
    local_risk["위험지수"] = local_risk["유해화학사업장_수"] / (
        local_risk["노인복지시설_수"] + 1
    )
    local_risk = local_risk.sort_values("위험지수", ascending=False)

    # 탭 구성
    tabs = st.tabs([
        "1. 데이터 개요",
        "2. 대기질 분석 (경기도 vs 평택시)",
        "3. 평택시 유해화학물질 취급 사업장",
        "4. 평택시 노인복지시설 분포",
        "5. 공공 ESG 관점 종합 진단"
    ])

    # --------------------------------------------------------
    # 1. 데이터 개요
    # --------------------------------------------------------
    with tabs[0]:
        st.subheader("데이터 개요")
        c1, c2, c3 = st.columns(3)
        c1.metric("대기질 월평균 데이터 (행)", f"{len(df_air):,}")
        c2.metric("노인복지시설 수", f"{len(df_elderly):,}")
        c3.metric("유해화학물질 취급 사업장 수", f"{len(df_chem):,}")

        st.markdown("#### (1) 대기질 월평균 데이터 예시")
        st.dataframe(
            df_air[
                [
                    "도시명", "측정장소명", "측정일",
                    "PM10측정값(㎍/㎥)", "PM25측정값(㎍/㎥)",
                    "오존측정값(ppm)", "종합위험점수"
                ]
            ].head(20),
            use_container_width=True,
        )

        st.markdown("#### (2) 노인복지시설 데이터 예시")
        st.dataframe(df_elderly.head(20), use_container_width=True)

        st.markdown("#### (3) 유해화학물질 취급 사업장 데이터 예시")
        st.dataframe(df_chem.head(20), use_container_width=True)

        st.caption("※ 종합위험점수: 각 월/측정소별 6개 오염물질 점수(1~4) 중 최댓값")

    # --------------------------------------------------------
    # 2. 대기질 분석
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("경기도 / 평택시 대기질 비교 및 추이 분석")

        city_list = sorted(df_air["도시명"].unique())
        default_city_idx = city_list.index("평택시") if "평택시" in city_list else 0

        left, right = st.columns([2, 3])

        with left:
            sel_city = st.selectbox("도시 선택", city_list, index=default_city_idx)

            df_city = df_air[df_air["도시명"] == sel_city].copy()
            site_list = sorted(df_city["측정장소명"].unique())
            sel_site = st.selectbox("측정소 선택", site_list)

            pollutant_options = list(POLLUTANT_COLS.keys())
            sel_pollutant = st.selectbox(
                "오염물질 선택",
                pollutant_options,
                format_func=lambda x: POLLUTANT_LABELS.get(x, x),
            )

            df_site = (
                df_city[df_city["측정장소명"] == sel_site]
                .sort_values("측정일")
            )

            value_col = POLLUTANT_COLS[sel_pollutant]
            grade_col = f"{sel_pollutant}_등급"

        with right:
            st.markdown(
                f"##### [{sel_city} - {sel_site}] {POLLUTANT_LABELS.get(sel_pollutant, sel_pollutant)} 월별 추이"
            )

            plot_df = df_site.set_index("측정일")[[value_col]]
            plot_df.columns = ["농도"]
            st.line_chart(plot_df)

            st.markdown("##### 같은 기간 등급 분포")
            grade_counts = (
                df_site[grade_col]
                .value_counts()
                .reindex(["좋음", "보통", "나쁨", "매우나쁨"])
                .fillna(0)
                .astype(int)
            )
            grade_df = grade_counts.rename_axis("등급").to_frame("월 수")
            st.bar_chart(grade_df)

        st.markdown("----")
        st.markdown("#### 도시별 평균 농도 및 종합위험점수 (경기도 전체)")

        st.dataframe(
            city_summary.sort_values("종합위험점수", ascending=False),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 3. 평택시 유해화학물질 취급 사업장
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("평택시 유해화학물질 취급 사업장 현황")

        st.metric("사업장 수", f"{len(df_chem):,}")

        industry_all = sorted(df_chem["업종명"].unique())
        selected_industries = st.multiselect(
            "업종 필터 (선택 안 하면 전체)",
            industry_all,
        )
        if selected_industries:
            df_chem_view = df_chem[df_chem["업종명"].isin(selected_industries)].copy()
        else:
            df_chem_view = df_chem.copy()

        st.markdown("#### (1) 사업장 위치 (위도/경도 기반)")
        if {"위도", "경도"}.issubset(df_chem_view.columns):
            map_df = df_chem_view.rename(columns={"위도": "lat", "경도": "lon"})
            st.map(map_df[["lat", "lon"]])
        else:
            st.info("위도/경도 정보가 없어 지도 시각화는 생략합니다.")

        st.markdown("#### (2) 업종별 사업장 수")
        chem_counts = (
            df_chem_view["업종명"]
            .value_counts()
            .rename_axis("업종명")
            .to_frame("사업장 수")
        )
        st.bar_chart(chem_counts)

        st.markdown("#### (3) 상세 테이블")
        st.dataframe(df_chem_view.reset_index(drop=True), use_container_width=True)

    # --------------------------------------------------------
    # 4. 평택시 노인복지시설 분포
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("평택시 노인복지시설 현황")

        st.metric("노인복지시설 수", f"{len(df_elderly):,}")

        st.markdown("#### (1) 시설종류별 개수")
        facility_counts = (
            df_elderly["시설종류"]
            .value_counts()
            .rename_axis("시설종류")
            .to_frame("개수")
        )
        st.bar_chart(facility_counts)

        st.markdown("#### (2) 도로명주소 검색")
        addr_query = st.text_input("도로명주소에 포함될 키워드 (예: 고덕, 안중, 청북 등)")
        df_elderly_view = df_elderly.copy()
        if addr_query:
            df_elderly_view = df_elderly_view[
                df_elderly_view["도로명주소"].str.contains(addr_query, na=False)
            ]

        st.dataframe(df_elderly_view.reset_index(drop=True), use_container_width=True)

        st.caption("※ 지도 시각화를 위해서는 도로명주소를 위/경도로 변환(지오코딩)하는 추가 작업이 필요합니다.")

    # --------------------------------------------------------
    # 5. 공공 ESG 관점 종합 진단  (★ 결론 + 시각화 강화)
    # --------------------------------------------------------
    with tabs[4]:
        st.subheader("공공 ESG 관점에서 본 평택시 대기질 리스크 진단")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "평택시 평균 종합위험점수 (1~4)",
            f"{pyeongtaek_risk:.2f}",
        )
        col2.metric(
            "경기도 평균 종합위험점수",
            f"{gyeonggi_mean_risk:.2f}",
            delta=f"{pyeongtaek_risk - gyeonggi_mean_risk:+.2f}",
        )
        col3.metric(
            "평택시 평균 PM2.5 (㎍/㎥)",
            f"{pyeongtaek_row['PM25측정값(㎍/㎥)']:.1f}",
        )

        st.markdown("#### (1) 평택시 대기환경 진단평가시스템 정보")
        st.write(f"- 지역 구분: **{region_row['지역']}**")
        st.write(f"- 시군구명: **{region_row['시군구명']}**")
        st.write(f"- 지형 코드: **{region_row['지형']}**")

        st.markdown("#### (2) 도시별 종합위험점수 비교 (상위 → 하위)")
        st.dataframe(
            city_summary.sort_values("종합위험점수", ascending=False),
            use_container_width=True,
        )

        # ★ 핵심 시각화: 평택시 읍·면·동별 '노인복지시설 vs 유해화학사업장' + 위험지수
        st.markdown("#### (3) 평택시 읍·면·동별 노인복지시설 · 유해화학사업장 · 위험지수")

        st.caption("위험지수 = 유해화학사업장 수 / (노인복지시설 수 + 1)")

        # 막대그래프: 행정동별 사업장 수 vs 노인복지시설 수
        st.bar_chart(local_risk[["유해화학사업장_수", "노인복지시설_수"]])

        # 테이블: 위험지수까지 포함
        st.dataframe(
            local_risk.reset_index().rename(columns={"행정동": "읍·면·동"}),
            use_container_width=True,
        )

        # 상위/하위 지역 자동 요약
        top_risky = local_risk.head(3).index.tolist()
        top_safe = local_risk.tail(3).index.tolist()

        st.markdown("#### (4) 시각자료 기반 결론 요약")
        st.markdown(
            f"""
            - **취약 지역(위험지수 상위 3)**: {", ".join(top_risky)}  
              → 유해화학물질 취급사업장에 비해 노인복지시설이 상대적으로 부족한 지역으로,  
                신규 노인복지시설 입지 검토가 필요한 **우선 관리 대상 권역**으로 해석할 수 있습니다.  

            - **상대적으로 양호한 지역(위험지수 하위 3)**: {", ".join(top_safe)}  
              → 노인복지시설이 상대적으로 충분하거나 유해화학사업장 밀집도가 낮은 지역으로,  
                신규 공급보다는 **기존 시설의 질적 개선과 서비스 고도화** 중심의 전략이 적합합니다.  
            """
        )

        st.caption(
            "※ '위험지수'는 단순화된 지표이므로, 실제 정책 제안 시에는 고령인구 비율, 대기질, 교통 접근성 등 "
            "추가 지표와 함께 종합 판단이 필요합니다."
        )


if __name__ == "__main__":
    main()
