import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re  # 행정동 파싱용 정규식
import pydeck as pdk

import requests  # 지오코딩용
import time  #지오코딩 호출 간 간격 조절용
# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="CSC - 평택시 대기질 리스크 & 노인복지시설 분석",
    layout="wide"
)

# 사용자 PC/클라우드 기준 데이터 폴더
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

# 평택시 법정 읍·면·동 25개만 사용
LEGAL_EMD = [
    "팽성읍", "안중읍", "포승읍", "청북읍",
    "진위면", "서탄면", "고덕면", "오성면", "현덕면",
    "중앙동",
    "서정동", "송탄동", "지산동", "송북동", "신장1동",
    "신장2동", "신평동", "원평동", "통복동", "비전1동",
    "비전2동", "세교동", "용이동", "동삭동", "고덕동",
]
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
    elderly_pop = read_csv_safely(
        BASE_DIR / "202504_202510_주민등록인구기타현황(고령 인구현황)_월간.csv"
    )
    return {
        "air": air,
        "grade": grade,
        "region": region,
        "elderly": elderly,
        "chem": chem,
        "elderly_pop": elderly_pop,
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

# 행정 읍·면·동 추출 : 평택시 법정 25개만 허용
def extract_eupmyeondong(addr: str) -> str:
    if pd.isna(addr):
        return np.nan

    text = str(addr)

    # 주소 문자열 안에 25개 읍·면·동 이름이 들어있는지만 확인
    for emd in LEGAL_EMD:
        if emd in text:
            return emd

    # 25개에 해당하지 않으면 제외
    return np.nan

# ------------------------------------------------------------
# 노인복지시설 도로명주소 지오코딩 (OpenStreetMap Nominatim 예시)
# ------------------------------------------------------------
def _geocode_single(addr: str):
    """단일 주소를 위도/경도로 변환 (실패하면 (None, None) 반환)."""
    if not addr or pd.isna(addr):
        return None, None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": addr,
        "format": "json",
        "limit": 1,
    }
    headers = {
        # OSM 정책상 user-agent 꼭 필요. 이메일은 편한 걸로 바꿔도 됨.
        "User-Agent": "pyeongtaek-esg-app (contact: your_email@example.com)"
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None, None


@st.cache_data
def geocode_elderly_addresses(addresses: tuple) -> pd.DataFrame:
    """
    노인복지시설 도로명주소 목록을 받아 지오코딩 결과(lat, lon)를 반환.
    같은 주소 세트에 대해서는 캐시되어 다시 호출되지 않음.
    """
    rows = []
    for addr in addresses:
        lat, lon = _geocode_single(addr)
        rows.append({"도로명주소": addr, "위도": lat, "경도": lon})
        # 너무 공격적으로 호출하면 막힐 수 있으니 약간의 간격
        time.sleep(0.3)
    return pd.DataFrame(rows)


def ensure_elderly_geocoded(df_elderly: pd.DataFrame) -> pd.DataFrame:
    """
    노인복지시설 데이터에 위도/경도 컬럼이 없거나 전부 결측이면
    도로명주소 기준으로 지오코딩을 수행해 lat/lon을 붙여줌.
    """
    df = df_elderly.copy()

    # 이미 위도/경도 있고 어느 정도 채워져 있으면 그대로 사용
    if {"위도", "경도"}.issubset(df.columns) and not df[["위도", "경도"]].isna().all().all():
        return df

    # 도로명주소에서 유니크 주소만 추출
    addr_series = df["도로명주소"].dropna().unique().tolist()
    if len(addr_series) == 0:
        return df

    addr_tuple = tuple(sorted(addr_series))

    with st.spinner("노인복지시설 도로명주소를 지오코딩하는 중입니다. (한 번만 수행됩니다)"):
        geo_df = geocode_elderly_addresses(addr_tuple)

    df = df.merge(geo_df, on="도로명주소", how="left")
    return df


# ------------------------------------------------------------
# 메인 대시보드
# ------------------------------------------------------------
def main():
    st.title("CSC 프로젝트 - 공공 ESG 관점 평택시 대기질 리스크 & 노인복지시설 분석")
    st.caption(
        "데이터 출처: 공공데이터포털(data.go.kr) - "
        "경기도 대기환경정보, 평택시 노인복지시설, 유해화학물질 취급사업장, "
        "경기도 대기환경 진단평가시스템 지역정보, 주민등록인구(고령 인구현황)"
    )

    # 데이터 로드
    data = load_data()
    df_air_raw = data["air"]
    df_grade = data["grade"]
    df_region = data["region"]
    df_elderly_raw = data["elderly"]
    df_chem = data["chem"]
    df_pop = data["elderly_pop"]

    # 노인복지시설: 도로명주소 지오코딩 적용
    df_elderly = ensure_elderly_geocoded(df_elderly_raw)

    # 전처리 (대기질 등급/위험점수 계산)
    df_air = add_air_quality_grades(df_air_raw, df_grade)
    city_summary = make_city_summary(df_air)

    # 평택시/경기도 평균 위험 점수
    pyeongtaek_row = city_summary[city_summary["도시명"] == "평택시"].iloc[0]
    gyeonggi_mean_risk = city_summary["종합위험점수"].mean()
    pyeongtaek_risk = pyeongtaek_row["종합위험점수"]

    # 평택시 기초 정보 (대기환경 진단평가시스템)
    region_row = df_region[df_region["시군구명"] == "평택시"].iloc[0]

    # 평택시 내부 읍·면·동 단위 '위험지수' 계산
    # 노인복지시설: 도로명주소 사용
    # 평택시 내부 읍·면·동 단위 '위험지수' 계산
    # 노인복지시설: 도로명주소 사용
    df_elderly["행정동"] = df_elderly["도로명주소"].apply(extract_eupmyeondong)

    # 유해화학물질 사업장: 도로명주소 → 안 나오면 지번주소로 보완
    df_chem["행정동"] = df_chem["소재지도로명주소"].apply(extract_eupmyeondong)
    mask_na = df_chem["행정동"].isna()
    if "소재지지번주소" in df_chem.columns:
        df_chem.loc[mask_na, "행정동"] = df_chem.loc[mask_na, "소재지지번주소"].apply(
            extract_eupmyeondong
        )

    # 👉 평택시 법정 25개 읍·면·동 기준으로만 재정렬
    emd_index = pd.Index(LEGAL_EMD, name="행정동")

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

    local_risk = (
        pd.concat([elderly_cnt, chem_cnt], axis=1)
        .reindex(emd_index)   # ← 25개 읍·면·동으로 강제 정렬
        .fillna(0)
    )

    local_risk["노인복지시설_수"] = local_risk["노인복지시설_수"].astype(int)
    local_risk["유해화학사업장_수"] = local_risk["유해화학사업장_수"].astype(int)

    # 단순 위험지수: 유해화학사업장 수 / (노인복지시설 수 + 1)
    local_risk["위험지수"] = local_risk["유해화학사업장_수"] / (
        local_risk["노인복지시설_수"] + 1
    )
    local_risk = local_risk.sort_values("위험지수", ascending=False)

    # 읍·면·동별 평균 좌표 (노인복지시설 + 유해화학사업장 모두 활용)
    coords_list = []
    if {"위도", "경도"}.issubset(df_elderly.columns):
        elder_coords_all = (
            df_elderly.dropna(subset=["위도", "경도"])
            .groupby("행정동")[["위도", "경도"]]
            .mean()
        )
        coords_list.append(elder_coords_all)

    if {"위도", "경도"}.issubset(df_chem.columns):
        chem_coords = (
            df_chem.dropna(subset=["위도", "경도"])
            .groupby("행정동")[["위도", "경도"]]
            .mean()
        )
        coords_list.append(chem_coords)

    if coords_list:
        coords_all = (
            pd.concat(coords_list)
            .groupby("행정동")[["위도", "경도"]]
            .mean()
        )
        local_risk_map = local_risk.join(coords_all, how="left")
    else:
        coords_all = pd.DataFrame()
        local_risk_map = local_risk.copy()

    # 주민등록 인구(고령 인구) - 평택시 읍·면·동별 65세 이상 인구
    df_pop_pt = df_pop[df_pop["행정구역"].str.contains("평택시", na=False)].copy()
    df_pop_pt["읍면동"] = df_pop_pt["행정구역"].apply(extract_eupmyeondong)
    df_pop_pt = df_pop_pt[~df_pop_pt["읍면동"].isna()].copy()

    aged_total_cols = [c for c in df_pop_pt.columns if "65세이상전체" in c]
    aged_total_cols = sorted(aged_total_cols)
    default_month_idx = len(aged_total_cols) - 1 if aged_total_cols else 0

    # --------------------------------------------------------
    # 탭 구성
    # --------------------------------------------------------
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

        st.markdown("#### (4) 주민등록 고령 인구 데이터 예시 (평택시)")
        st.dataframe(df_pop_pt.head(20), use_container_width=True)

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

        with right:
            st.markdown(
                f"##### [{sel_city} - {sel_site}] {POLLUTANT_LABELS.get(sel_pollutant, sel_pollutant)} 월별 추이"
            )

            plot_df = df_site.set_index("측정일")[[value_col]]
            plot_df.columns = ["농도"]
            st.line_chart(plot_df)

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

        st.markdown("#### (2) 상세 테이블")
        st.dataframe(df_chem_view.reset_index(drop=True), use_container_width=True)
    # --------------------------------------------------------
    # 4. 평택시 노인복지시설 분포 (지도 시각화 + 충족도 데이터)
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("평택시 노인복지시설 현황")

        st.metric("노인복지시설 수", f"{len(df_elderly):,}")

        # (1) 노인복지시설 위치 지도 ----------------------------------------
        st.markdown("#### (1) 노인복지시설 위치 지도")
        if {"위도", "경도"}.issubset(df_elderly.columns) and not df_elderly[["위도", "경도"]].isna().all().all():
            elder_map = (
                df_elderly
                .dropna(subset=["위도", "경도"])
                .rename(columns={"위도": "lat", "경도": "lon"})
            )
            st.map(elder_map[["lat", "lon"]])
        else:
            st.info(
                "노인복지시설 도로명주소 지오코딩 결과가 없어서 지도를 표시하지 못했습니다."
            )

        # (2) 읍·면·동별 65세 이상 인구 대비 노인복지시설 충족도 -------------
        st.markdown("#### (2) 읍·면·동별 65세 이상 인구 대비 노인복지시설 충족도")

        if aged_total_cols:
            # 기준 월 선택
            month_label_map = {col: col.replace("_65세이상전체", "") for col in aged_total_cols}
            month_labels = list(month_label_map.values())
            sel_label = st.selectbox(
                "기준 월 선택 (65세 이상 인구)",
                month_labels,
                index=default_month_idx,
            )
            inv_month_label_map = {v: k for k, v in month_label_map.items()}
            sel_col = inv_month_label_map[sel_label]

            # 평택시 읍·면·동별 65세 이상 인구
            pop_month = (
                df_pop_pt[["읍면동", sel_col]]
                .assign(
                    고령인구_수=lambda d: d[sel_col]
                    .replace(",", "", regex=True)
                    .astype("int64")
                )[["읍면동", "고령인구_수"]]
                .groupby("읍면동")["고령인구_수"]
                .sum()
                .rename_axis("행정동")
            )

            # 노인복지시설 수 (행정동 기준)
            elderly_cnt_for_cov = (
                df_elderly.groupby("행정동")
                .size()
                .rename("노인복지시설_수")
            )

            coverage = pd.concat([elderly_cnt_for_cov, pop_month], axis=1)
            coverage["노인복지시설_수"] = coverage["노인복지시설_수"].fillna(0).astype(int)
            coverage["고령인구_수"] = coverage["고령인구_수"].fillna(0).astype(int)

            # 65세 이상 1천 명당 시설 수
            coverage["시설_천명당"] = np.where(
                coverage["고령인구_수"] > 0,
                coverage["노인복지시설_수"] / (coverage["고령인구_수"] / 1000.0),
                np.nan,
            )

            # 지도용 좌표 (행정동별 평균 위도/경도)
            if {"위도", "경도"}.issubset(df_elderly.columns):
                coords_cov = (
                    df_elderly
                    .dropna(subset=["위도", "경도"])
                    .groupby("행정동")[["위도", "경도"]]
                    .mean()
                    .rename(columns={"위도": "lat", "경도": "lon"})
                )
                coverage_with_coords = (
                    coverage.join(coords_cov, how="left")
                    .reset_index()
                    .rename(columns={"행정동": "읍면동"})
                )
            else:
                coverage_with_coords = (
                    coverage.reset_index()
                    .rename(columns={"행정동": "읍면동"})
                )

            # 지도 표시 (좌표와 충족도 모두 있는 행만 사용)
            if {"lat", "lon"}.issubset(coverage_with_coords.columns):
                cov_for_map = coverage_with_coords.dropna(subset=["lat", "lon", "시설_천명당"])
                if not cov_for_map.empty:
                    max_cov = float(cov_for_map["시설_천명당"].max())
                    min_radius, max_radius = 300, 1400
                    cov_for_map["marker_radius"] = (
                        min_radius
                        + (cov_for_map["시설_천명당"] / max_cov) * (max_radius - min_radius)
                    )

                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=cov_for_map,
                        get_position="[lon, lat]",
                        get_radius="marker_radius",
                        get_fill_color="[0, 153, 255, 150]",
                        pickable=True,
                    )
                    view_state = pdk.ViewState(
                        latitude=float(cov_for_map["lat"].mean()),
                        longitude=float(cov_for_map["lon"].mean()),
                        zoom=10.5,
                        pitch=0,
                    )
                    st.pydeck_chart(
                        pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            tooltip={
                                "text": "읍·면·동: {읍면동}\\n"
                                        "65세 이상 인구: {고령인구_수}명\\n"
                                        "노인복지시설 수: {노인복지시설_수}개\\n"
                                        "시설 수 (천 명당): {시설_천명당:.2f}"
                            },
                        )
                    )
                    st.caption(
                        f"※ 기준 월: **{sel_label}**, 65세 이상 1천 명당 시설 수가 클수록 "
                        "노인복지 인프라가 상대적으로 잘 갖춰진 지역입니다."
                    )

            # ✅ 지오코딩 + 충족도 결과 데이터 테이블 항상 보여주기
            st.markdown("#### (3) 읍·면·동별 노인복지시설 충족도 데이터")
            cols_to_show = ["읍면동", "고령인구_수", "노인복지시설_수", "시설_천명당"]
            # 좌표까지 같이 보고 싶으면 lat/lon도 포함
            if "lat" in coverage_with_coords.columns and "lon" in coverage_with_coords.columns:
                cols_to_show += ["lat", "lon"]

            st.dataframe(
                coverage_with_coords[cols_to_show].sort_values(
                    "시설_천명당", ascending=False
                ),
                use_container_width=True,
            )
        else:
            st.info("주민등록 인구 통계 데이터에서 '65세이상전체' 컬럼을 찾을 수 없습니다.")

        # (4) 도로명주소 검색 --------------------------------------------
        st.markdown("#### (4) 도로명주소 검색")
        addr_query = st.text_input("도로명주소에 포함될 키워드 (예: 고덕, 안중, 청북 등)")
        df_elderly_view = df_elderly.copy()
        if addr_query:
            df_elderly_view = df_elderly_view[
                df_elderly_view["도로명주소"].str.contains(addr_query, na=False)
            ]

        st.dataframe(df_elderly_view.reset_index(drop=True), use_container_width=True)

    # --------------------------------------------------------
    # 5. 공공 ESG 관점 종합 진단  (지도 + 결론)
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

        st.markdown("#### (3) 평택시 읍·면·동별 위험지수 지도")

        if {"위도", "경도"}.issubset(local_risk_map.columns) and not local_risk_map["위도"].isna().all():
            risk_map_df = local_risk_map.dropna(subset=["위도", "경도"]).reset_index()
            risk_map_df = risk_map_df.rename(
                columns={"행정동": "읍면동", "위도": "lat", "경도": "lon"}
            )

            # 🔹 원 크기 축소: 위험지수를 0~1 사이로 정규화해서 300~1300m 정도로만 표시
            max_risk = float(risk_map_df["위험지수"].max())
            min_radius = 300   # 최소 반경(미터)
            max_radius = 1300  # 최대 반경(미터)

            risk_map_df["marker_radius"] = (
                min_radius
                + (risk_map_df["위험지수"] / max_risk) * (max_radius - min_radius)
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=risk_map_df,
                get_position="[lon, lat]",
                get_radius="marker_radius",
                get_fill_color="[255, 0, 0, 140]",  # 약간 투명한 빨간색
                pickable=True,
            )

            view_state = pdk.ViewState(
                latitude=float(risk_map_df["lat"].mean()),
                longitude=float(risk_map_df["lon"].mean()),
                zoom=10.5,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"text": "읍·면·동: {읍면동}\\n위험지수: {위험지수}"},
                )
            )

            # 🔹 두 번째 지오코딩 설명 추가
            st.caption(
                "※ 위험지수 지도는 노인복지시설·유해화학사업장 주소를 지오코딩하여 얻은 좌표(위도·경도)를 "
                "읍·면·동별로 평균낸 위치에 표시한 것입니다."
            )

        else:
            st.info(
                "위험지수 지도를 표시하려면 읍·면·동별 위도/경도 정보가 필요합니다. "
                "노인복지시설 및 유해화학물질 사업장 도로명주소를 지오코딩해 '위도', '경도' 열을 추가해 주세요."
            )

        st.markdown("#### (4) 평택시 읍·면·동별 노인복지시설 · 유해화학사업장 · 위험지수")

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

        st.markdown("#### (5) 시각자료 기반 결론 요약")
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
