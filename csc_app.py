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

# QR 코드 이미지 경로 (csc_app.py와 같은 폴더에 저장했다고 가정)
QR_PATH = Path(__file__).parent / "평택 ESG streamlit QR.png"

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

# 평택시 법정동 23개 (비전1·2동, 신장1·2동 통합)
LEGAL_EMD = [
    "팽성읍", "안중읍", "포승읍", "청북읍",
    "진위면", "서탄면", "고덕면", "오성면", "현덕면",
    "중앙동",
    "서정동", "송탄동", "지산동", "송북동",
    "신장동",   # 신장1·2동 포함
    "신평동", "원평동", "통복동",
    "비전동",   # 비전1·2동 포함
    "세교동", "용이동", "동삭동", "고덕동",
]

# 동 이름 매핑 (비전1·2동 → 비전동, 신장1·2동 → 신장동)
# 동 이름 매핑 (비전1·2동 → 비전동, 신장1·2동 → 신장동)
# 동 이름 매핑 (비전1·2동, 신장1·2동 외에 예전 동 이름까지 포함)
EMD_ALIAS_MAP = {
    # 비전동 계열
    "비전동": "비전동",
    "비전1동": "비전동",
    "비전 1동": "비전동",
    "비전2동": "비전동",
    "비전 2동": "비전동",
    # 신장동 계열
    "신장동": "신장동",
    "신장1동": "신장동",
    "신장 1동": "신장동",
    "신장2동": "신장동",
    "신장 2동": "신장동",
    # 예전 동 이름 → 현재 동 이름
    "이충동": "중앙동",
    "평택동": "원평동",
    "합정동": "신평동",
    "소사동": "비전동",
    "유천동": "신평동",
    "가재동": "송탄동",
    "장당동": "중앙동",
    "칠괴동": "송탄동",
    "신대동": "원평동",
}

# 도로명주소 안의 예전 동 이름을 새 동 이름으로 치환 + 괄호 내용 제거
OLD_DONG_IN_ADDR_MAP = {
    "이충동": "중앙동",
    "평택동": "원평동",
    "합정동": "신평동",
    "소사동": "비전동",
    "유천동": "신평동",
    "가재동": "송탄동",
    "장당동": "중앙동",
    "칠괴동": "송탄동",
    "신대동": "원평동",
}
# extract_eupmyeondong()에서 사용할 옛 동이름 → 현재 동이름 매핑
OLD_EMD_TO_NEW = OLD_DONG_IN_ADDR_MAP

def normalize_address_for_geocode(addr: str) -> str:
    """지오코딩을 위한 주소 정리 (옛 동 이름 → 현재 동 이름, 괄호 내용 제거 등)."""
    if not addr or pd.isna(addr):
        return ""

    addr = str(addr)

    # 1) 옛 동 이름이 주소 문자열에 직접 들어가 있으면 현재 동 이름으로 치환
    for old, new in OLD_DONG_IN_ADDR_MAP.items():
        if old in addr:
            addr = addr.replace(old, new)

    # 2) 괄호 안 내용(동 이름이나 건물명 등)은 지오코딩에 방해가 되므로 제거
    addr = re.sub(r"\(.*?\)", "", addr)

    # 3) 쉼표 뒤(층/호, 지층 등) 정보는 지오코딩에 거의 필요 없으므로 첫 번째 쉼표 앞까지만 사용
    addr = addr.split(",", 1)[0]

    # 4) 양쪽 공백 정리
    addr = addr.strip()

    return addr


# 도로명주소 문자열 안에서도 같은 치환을 사용하기 위한 dict
OLD_DONG_IN_ADDR = {
    old: new for old, new in EMD_ALIAS_MAP.items() if old != new
}
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
    elderly = read_csv_safely(BASE_DIR / "경기도_평택시_노인복지시설_20250129_(1)_geocoded.csv")
    chem = read_csv_safely(BASE_DIR / "경기도_평택시_유해화학물질_취급사업장_현황_20250207.csv")
    cai = read_csv_safely(BASE_DIR / "pyeongtaek_CAI_index.csv")
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
        "cai": cai,
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
# 행정 읍·면·동만 뽑는 함수 (건물동 필터링 + 비전/신장 통합)
def extract_eupmyeondong(addr: str) -> str:
    if pd.isna(addr):
        return np.nan

    # 문자열로 변환 후, 옛 동 이름을 현재 명칭으로 교체
    text = str(addr)
    for old, new in OLD_EMD_TO_NEW.items():
        if old in text:
            text = text.replace(old, new)

    # 0단계: 주소 전체에서 비전/신장 계열 먼저 처리
    #   예: "비전1동 123-4", "신장 2동 11-3" 등
    for key, canon in EMD_ALIAS_MAP.items():
        if key in text:
            return canon

    # 1단계: 나머지는 기존 로직대로 토큰 단위 필터링
    tokens = re.split(r"[ ,()]", text)

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # 읍/면/동으로 끝나지 않으면 패스
        if not tok.endswith(("읍", "면", "동")):
            continue

        # 광역/기초 지자체 이름 제외
        if tok in ("경기도", "평택시"):
            continue

        # 숫자 + 동 (1동, 103동 등) → 건물동
        if re.fullmatch(r"\d+동", tok):
            continue

        # 영문/숫자 코드 + 동 (A동, B동, S001동 등) → 건물동
        if re.fullmatch(r"[A-Za-z0-9]+동", tok):
            continue

        # 제1동, 제2동 형태 → 건물동
        if re.fullmatch(r"제\d+동", tok):
            continue

        # 상가동 관련 → 건물동
        if "상가동" in tok or (tok.startswith("상가") and tok.endswith("동")):
            continue

        # 한 글자 + 동 (가동, 나동 등) → 건물동
        if len(tok) == 2 and tok.endswith("동"):
            continue

        # 비전1동/신장1동 등이 토큰으로 들어온 경우를 정규화
        norm = EMD_ALIAS_MAP.get(tok, tok)

        # 23개 법정동 안에 들어가는 것만 인정
        if norm in LEGAL_EMD:
            return norm

    return np.nan


# ------------------------------------------------------------
# 노인복지시설 도로명주소 지오코딩 (OpenStreetMap Nominatim 예시)
# ------------------------------------------------------------
def _geocode_single(addr: str):
    """단일 주소를 위도/경도로 변환 (실패하면 (None, None) 반환)."""
    if not addr or pd.isna(addr):
        return None, None

    # ----- 1) 옛 동 이름을 현재 동 이름으로 치환 -----
    addr_norm = str(addr)
    for old, new in OLD_DONG_IN_ADDR.items():
        if old in addr_norm:
            addr_norm = addr_norm.replace(old, new)

    query = normalize_address_for_geocode(addr)

    # ----- 2) OSM Nominatim 호출 -----
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
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
    노인복지시설 데이터에 대해 도로명주소 기준으로 지오코딩을 수행하여
    위도/경도 컬럼을 채워준다.
    - CSV에 lat/lon 컬럼이 이미 있다면 그것을 위도/경도로 사용하고,
      부족한 행만 추가로 지오코딩한다.
    """
    df = df_elderly.copy()

    # 0) _geocoded.csv에 lat / lon 으로 저장된 경우를 먼저 처리
    if "lat" in df.columns and "위도" not in df.columns:
        df.rename(columns={"lat": "위도"}, inplace=True)
    if "lon" in df.columns and "경도" not in df.columns:
        df.rename(columns={"lon": "경도"}, inplace=True)

    # 1) 위도/경도 컬럼이 없으면 생성
    if "위도" not in df.columns:
        df["위도"] = np.nan
    if "경도" not in df.columns:
        df["경도"] = np.nan

    # 2) 지오코딩이 필요한 행(위도 또는 경도가 비어 있는 경우)만 대상으로
    need_geo_mask = df["위도"].isna() | df["경도"].isna()
    addrs_to_geo = (
        df.loc[need_geo_mask, "도로명주소"]
        .dropna()
        .unique()
        .tolist()
    )

    if not addrs_to_geo:
        # 이미 전부 채워져 있으면 그대로 반환
        return df

    addr_tuple = tuple(sorted(addrs_to_geo))

    with st.spinner("노인복지시설 도로명주소를 지오코딩하는 중입니다. (한 번만 수행됩니다)"):
        geo_df = geocode_elderly_addresses(addr_tuple)

    # 3) 기존 df와 지오코딩 결과를 병합
    df = df.merge(geo_df, on="도로명주소", how="left", suffixes=("", "_geo"))

    # 기존 위도/경도가 비어 있는 경우에만 _geo 값으로 채우기
    for col in ["위도", "경도"]:
        geo_col = f"{col}_geo"
        if geo_col in df.columns:
            df[col] = df[col].where(~df[col].isna(), df[geo_col])
            df.drop(columns=[geo_col], inplace=True)
    # 4) 아직도 좌표가 없는 행은 '행정동 중심점'으로 보정
    if "행정동" in df.columns:
        # 좌표가 있는 시설들만 이용해서 행정동별 평균 좌표 계산
        emd_centers = (
            df.dropna(subset=["위도", "경도"])
              .groupby("행정동")[["위도", "경도"]]
              .mean()
        )

        if not emd_centers.empty:
            def fill_with_center(row):
                if (pd.isna(row["위도"]) or pd.isna(row["경도"])):
                    emd = row.get("행정동")
                    if emd in emd_centers.index:
                        center = emd_centers.loc[emd]
                        row["위도"] = center["위도"]
                        row["경도"] = center["경도"]
                return row

            df = df.apply(fill_with_center, axis=1)

    return df



# ------------------------------------------------------------
# 메인 대시보드
# ------------------------------------------------------------
def main():
    st.title("CSC 프로젝트 - 공공 ESG 관점 평택시 대기질 리스크 & 노인복지시설 분석")

    # Streamlit 대시보드 QR 코드 (제목 아래, 데이터 출처 위)
    qr_path = Path(__file__).parent / "평택 ESG streamlit QR.png"
    if qr_path.exists():
        st.image(
            str(qr_path),
            width=180,
            caption="모바일에서 열어보기 (Streamlit 대시보드 QR)",
        )

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
    df_cai = data["cai"]

    df_elderly_raw = df_elderly_raw.copy()
    df_elderly_raw["행정동"] = df_elderly_raw["도로명주소"].apply(extract_eupmyeondong)


    # 노인복지시설: 도로명주소 지오코딩 적용
    df_elderly = ensure_elderly_geocoded(df_elderly_raw)

    # 전처리 (대기질 등급/위험점수 계산)
    df_air = add_air_quality_grades(df_air_raw, df_grade)
    city_summary = make_city_summary(df_air)

    # 👉 평택시 종합위험점수는 CAI 파일 값으로 대체
    # pyeongtaek_CAI_index.csv : 읍·면·동별 CAI_Index (이미 종합 위험지수로 계산된 값)
    if "CAI_Index" in df_cai.columns:
        # 평택시 23개 읍·면·동 CAI_Index의 평균을 '평택시 종합위험점수'로 사용
        pyeongtaek_cai_mean = df_cai["CAI_Index"].mean()
        city_summary.loc[
            city_summary["도시명"] == "평택시", "종합위험점수"
        ] = pyeongtaek_cai_mean

    # 평택시/경기도 평균 위험 점수 (위 코드에서 평택시 값 이미 교체됨)
    pyeongtaek_row = city_summary[city_summary["도시명"] == "평택시"].iloc[0]
    gyeonggi_mean_risk = city_summary["종합위험점수"].mean()
    pyeongtaek_risk = pyeongtaek_row["종합위험점수"]

    # 평택시 기초 정보 (대기환경 진단평가시스템)
    region_row = df_region[df_region["시군구명"] == "평택시"].iloc[0]

    # 평택시 내부 읍·면·동 단위 '위험지수' 계산
    # 노인복지시설: 도로명주소 사용
    # 노인복지시설: 도로명주소 사용
    df_elderly["행정동"] = df_elderly["도로명주소"].apply(extract_eupmyeondong)

    # 유해화학물질 사업장: 도로명주소 → 안 나오면 지번주소로 보완
    df_chem["행정동"] = df_chem["소재지도로명주소"].apply(extract_eupmyeondong)
    mask_na = df_chem["행정동"].isna()
    if "소재지지번주소" in df_chem.columns:
        df_chem.loc[mask_na, "행정동"] = df_chem.loc[mask_na, "소재지지번주소"].apply(
            extract_eupmyeondong
        )

    # 👉 평택시 법정동 23개 기준으로 강제 정렬/채움
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

    # 0단계: 기본 집계 (시설 수)
    local_risk = (
        pd.concat([elderly_cnt, chem_cnt], axis=1)
        .reindex(emd_index)   # ← 23개 동으로 재정렬 + 없는 동은 0으로
        .fillna(0)
    )

    local_risk["노인복지시설_수"] = local_risk["노인복지시설_수"].astype(int)
    local_risk["유해화학사업장_수"] = local_risk["유해화학사업장_수"].astype(int)

    # 1단계: CAI 파일에서 '대기질 위험 지수' 가져오기
    # pyeongtaek_CAI_index.csv : [읍면동, CAI_Index, CAI_등급]
    cai_index = df_cai.set_index("읍면동")
    local_risk = local_risk.join(cai_index[["CAI_Index"]], how="left")

    # CAI_Index를 '대기질위험지수'로 사용
    local_risk = local_risk.rename(columns={"CAI_Index": "대기질위험지수"})

    # 2단계: 최종 위험 지수 = 0.3 * 유해화학사업장 수 + 0.7 * 대기질 위험 지수
    local_risk["위험지수"] = (
        0.3 * local_risk["유해화학사업장_수"]
        + 0.7 * local_risk["대기질위험지수"]
    )

    # 최종 위험 지수 기준 정렬
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
        "5. 위험지수 분석",
        "6. 공공 ESG 관점 종합 진단",
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

            # 결론 지도에서 쓸 건 시설_천명당만 필요
            coverage_final = coverage[["시설_천명당"]]

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
    # 5. 위험지수 분석 (읍·면·동 단위)
    # --------------------------------------------------------
    with tabs[4]:
        st.subheader("평택시 읍·면·동별 환경 위험지수 분석")

        # (1) 평택시 대기환경 진단평가시스템 정보 ---------------------------
        st.markdown("#### (1) 평택시 대기환경 진단평가시스템 정보")
        st.write(f"- 지역 구분: **{region_row['지역']}**")
        st.write(f"- 시군구명: **{region_row['시군구명']}**")
        st.write(f"- 지형 코드: **{region_row['지형']}**")

         # (2) 평택시 읍·면·동별 노인복지시설 · 유해화학사업장 · 대기질 위험 지수 · 위험 지수
        st.markdown("#### (2) 평택시 읍·면·동별 노인복지시설 · 유해화학사업장 · 대기질 위험 지수 · 위험 지수")
        st.caption(
            "대기질 위험 지수 = pyeongtaek_CAI_index.csv의 CAI_Index 값, "
            "위험 지수 = 0.3 × 유해화학사업장 수 + 0.7 × 대기질 위험 지수"
        )

        risk_table = (
            local_risk[["노인복지시설_수", "유해화학사업장_수", "대기질위험지수", "위험지수"]]
            .reset_index()
            .rename(
                columns={
                    "행정동": "읍·면·동",
                    "노인복지시설_수": "노인복지시설 수",
                    "유해화학사업장_수": "유해화학사업장 수",
                    "대기질위험지수": "대기질 위험 지수",
                    "위험지수": "위험 지수",
                }
            )
        )

        st.dataframe(risk_table, use_container_width=True)

        # (3) 평택시 읍·면·동별 위험 지수 지도 ------------------------------
        st.markdown("#### (3) 평택시 읍·면·동별 위험 지수 지도")

        if {"위도", "경도"}.issubset(local_risk_map.columns) and not local_risk_map["위도"].isna().all():
            risk_map_df = local_risk_map.dropna(subset=["위도", "경도"]).reset_index()
            risk_map_df = risk_map_df.rename(
                columns={"행정동": "읍면동", "위도": "lat", "경도": "lon"}
            )

            max_risk = float(risk_map_df["위험지수"].max())
            min_radius = 300
            max_radius = 1300

            risk_map_df["marker_radius"] = (
                min_radius
                + (risk_map_df["위험지수"] / max_risk) * (max_radius - min_radius)
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=risk_map_df,
                get_position="[lon, lat]",
                get_radius="marker_radius",
                get_fill_color="[255, 0, 0, 140]",
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
                    tooltip={"text": "읍·면·동: {읍면동}\n위험지수: {위험지수}"},
                )
            )

            st.caption(
                "※ 위험지수 지도는 노인복지시설·유해화학사업장 주소를 지오코딩하여 얻은 좌표(위도·경도)를 "
                "읍·면·동별로 평균낸 위치에 표시한 것입니다."
            )
        else:
            st.info(
                "위험지수 지도를 표시하려면 읍·면·동별 위도/경도 정보가 필요합니다. "
                "노인복지시설 및 유해화학물질 사업장 도로명주소를 지오코딩해 '위도', '경도' 열을 추가해 주세요."
            )

        # (4) 시각자료 기반 결론 요약 ---------------------------------------
        st.markdown("#### (4) 시각자료 기반 결론 요약")

        top_risky = local_risk.sort_values("위험지수", ascending=False).head(3).index.tolist()
        top_safe = local_risk.sort_values("위험지수", ascending=True).head(3).index.tolist()

        st.markdown(
            f"""
            - **위험 지수 상위 3개 읍·면·동**: {", ".join(top_risky)}  
              → 유해화학물질 취급 사업장 수(가중치 0.3)와 대기질 위험 지수(가중치 0.7)가 모두 높은 **복합 환경 취약 지역**입니다.  
                · 신규 노인복지시설 입지 선정 시에는 이들 지역은 지양하고,  
                  기존 시설에 대해서는 공기질 개선, 유해물질 관리, 실내 환기·필터링 강화 등 **환경 리스크 관리 중심 전략**이 필요합니다.  

            - **위험 지수 하위 3개 읍·면·동**: {", ".join(top_safe)}  
              → 대기질이 상대적으로 양호하거나 유해화학사업장 밀집도가 낮은 **상대적 안전 지역**으로 해석됩니다.  
                · 추가 노인복지시설 공급이 필요할 경우, 우선적으로 검토할 수 있는 후보지이며,  
                  동시에 **생활 편의·접근성·서비스 품질** 측면에서의 세밀한 개선이 적합한 지역입니다.
            """
        )

        st.markdown(
            f"""
            - **취약 지역(위험지수 상위 3)**: {", ".join(top_risky)}  
              → 유해화학물질 취급사업장 밀집도와 대기질(CAI)이 상대적으로 나쁜 지역으로,  
                동시에 노인복지시설이 부족할 가능성이 높은 **우선 관리 대상 권역**으로 해석할 수 있습니다.  

            - **상대적으로 양호한 지역(위험지수 하위 3)**: {", ".join(top_safe)}  
              → 노인복지시설이 상대적으로 충분하거나 유해화학사업장 밀집도가 낮은 지역으로,  
                신규 공급보다는 **기존 시설의 질적 개선과 서비스 고도화** 중심의 전략이 적합합니다.  
            """
        )
        # --------------------------------------------------------
    # 6. 공공 ESG 관점 종합 진단  (지도 + 결론)
    # --------------------------------------------------------
    with tabs[5]:
        st.subheader("공공 ESG 관점에서 본 평택시 노인복지시설 입지 전략")

        # 👉 읍·면·동별 환경 위험(local_risk_map) + 노인복지시설 충족도(coverage_final) 결합
        #   - index: 행정동
        #   - 필요한 컬럼: 시설_천명당, 위험지수, 위도, 경도
        emd_final = local_risk_map.join(coverage_final, how="left")
        emd_final = emd_final.dropna(subset=["시설_천명당", "위험지수", "위도", "경도"])

        # (1) 현재 종합위험지수 / 경기도 평균 / 평택시 PM2.5
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

        # 공통 데이터(읍·면·동별 위험지수 + 좌표)
        has_coords = {"위도", "경도"}.issubset(local_risk_map.columns) and not local_risk_map["위도"].isna().all()
        if has_coords:
            base_geo = local_risk_map.dropna(subset=["위도", "경도"]).reset_index()
            base_geo = base_geo.rename(
                columns={"행정동": "읍면동", "위도": "lat", "경도": "lon"}
            )
        else:
            base_geo = None

        # (2) 환경 위험 지도 (빨간색 원)
        st.markdown("#### (2) 환경 위험 지도 (읍·면·동별 환경 위험지수)")
        if has_coords and not base_geo.empty:
            max_risk = float(base_geo["위험지수"].max())
            min_radius = 300
            max_radius = 1300

            base_geo["marker_radius"] = (
                min_radius
                + (base_geo["위험지수"] / max_risk) * (max_radius - min_radius)
            )

            env_layer = pdk.Layer(
                "ScatterplotLayer",
                data=base_geo,
                get_position="[lon, lat]",
                get_radius="marker_radius",
                get_fill_color="[255, 0, 0, 140]",  # 빨간색
                pickable=True,
            )

            env_view = pdk.ViewState(
                latitude=float(base_geo["lat"].mean()),
                longitude=float(base_geo["lon"].mean()),
                zoom=10.5,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[env_layer],
                    initial_view_state=env_view,
                    tooltip={"text": "읍·면·동: {읍면동}\n위험지수: {위험지수}"},
                )
            )
        else:
            st.info("환경 위험 지도를 그리기 위한 읍·면·동 좌표(위도/경도)가 없습니다.")

        # (3) 청정 구역 지도 (초록색 원)
        st.markdown("#### (3) 청정 구역 지도 (위험지수 하위 지역)")
        if has_coords and not base_geo.empty:
            clean_threshold = base_geo["위험지수"].quantile(0.30)
            clean_geo = base_geo[base_geo["위험지수"] <= clean_threshold]

            if not clean_geo.empty:
                clean_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=clean_geo,
                    get_position="[lon, lat]",
                    get_radius=900,
                    get_fill_color="[0, 200, 0, 180]",  # 초록색
                    pickable=True,
                )

                clean_view = pdk.ViewState(
                    latitude=float(clean_geo["lat"].mean()),
                    longitude=float(clean_geo["lon"].mean()),
                    zoom=10.5,
                    pitch=0,
                )

                st.pydeck_chart(
                    pdk.Deck(
                        layers=[clean_layer],
                        initial_view_state=clean_view,
                        tooltip={"text": "읍·면·동: {읍면동}\n위험지수: {위험지수}"},
                    )
                )
            else:
                st.info("위험지수가 낮은(청정) 구역이 통계적으로 충분히 나오지 않았습니다.")
        else:
            st.info("청정 구역 지도를 그리기 위한 읍·면·동 좌표(위도/경도)가 없습니다.")

        # (4) 노인복지시설 위치 지도
        st.markdown("#### (4) 노인복지시설 위치 지도")
        if {"위도", "경도"}.issubset(df_elderly.columns):
            elder_geo = df_elderly.dropna(subset=["위도", "경도"]).rename(
                columns={"위도": "lat", "경도": "lon"}
            )
            st.map(elder_geo[["lat", "lon"]])
        else:
            st.info(
                "노인복지시설 데이터에 위도/경도 열이 없습니다. "
                "도로명주소를 지오코딩해 '위도', '경도' 열을 추가하면 지도 시각화가 가능합니다."
            )

        # (5) 결론 지도: 관리 집중 / 시설 증설 대상 구역
        st.markdown("#### (5) 결론 지도: 노인복지시설과 환경 리스크를 함께 본 우선·증설 대상 구역")

        if not emd_final.empty:
            # emd_final: 행정동 index + [노인복지시설_수, 유해화학사업장_수, 대기질위험지수, 위험지수, 위도, 경도, 시설_천명당]
            emd_geo = (
                emd_final
                .reset_index()
                .rename(columns={"행정동": "읍면동", "위도": "lat", "경도": "lon"})
                .dropna(subset=["lat", "lon"])
            )

            # 1) 기준선 계산
            risk_low_thr  = emd_geo["위험지수"].median()        # ‘청정’ 기준 (위험 지수 낮은 쪽)
            risk_high_thr = emd_geo["위험지수"].quantile(0.75)  # ‘고위험’ 기준
            cov_low_thr   = emd_geo["시설_천명당"].quantile(0.25)  # 시설 부족 기준 (하위 25%)
            cov_high_thr  = emd_geo["시설_천명당"].quantile(0.75)  # 시설 충분 기준 (상위 25%)

            # 1-1) 시설 증설 후보(파란 원)
            #  - 조건: 위험지수 ≤ 중위수(상대적으로 청정) AND 시설_천명당 ≤ 하위 25% (시설 부족)
            expand_candidates = (
                emd_geo[
                    (emd_geo["위험지수"] <= risk_low_thr)
                    & (emd_geo["시설_천명당"] <= cov_low_thr)
                ]
                .sort_values("시설_천명당")      # 시설_천명당 낮은 순 → 1순위
                .head(3)
                .copy()
            )
            expand_candidates["expand_rank"] = np.arange(1, len(expand_candidates) + 1)

            # 1-2) 관리 집중 후보(빨간 원)
            #  - 조건: 위험지수 ≥ 상위 25% AND 시설_천명당 ≥ 상위 25% (시설은 많은데 환경이 나쁨)
            focus_candidates = (
                emd_geo[
                    (emd_geo["위험지수"] >= risk_high_thr)
                    & (emd_geo["시설_천명당"] >= cov_high_thr)
                ]
                .sort_values("위험지수", ascending=False)  # 위험지수 높은 순 → 1순위
                .head(3)
                .copy()
            )
            focus_candidates["focus_rank"] = np.arange(1, len(focus_candidates) + 1)

            # 2) 원 크기 설정
            #   - 파란 원: 증설 우선순위 1 → 가장 큰 원, 2·3위는 점점 작게
            #   - 빨간 원: 관리 우선순위 1 → 가장 큰 원, 2·3위는 점점 작게
            blue_radius = {1: 1300, 2: 1000, 3: 700}
            red_radius  = {1: 1500, 2: 1150, 3: 800}

            expand_candidates["marker_radius_blue"] = expand_candidates["expand_rank"].map(blue_radius)
            focus_candidates["marker_radius_red"]   = focus_candidates["focus_rank"].map(red_radius)

            # 3) 레이어 구성 (배경 + 파란 원 + 빨간 원)
            layers = [
                # 전체 읍·면·동 배경 (옅은 회색 점)
                pdk.Layer(
                    "ScatterplotLayer",
                    data=emd_geo,
                    get_position="[lon, lat]",
                    get_radius=250,
                    get_fill_color="[120, 120, 120, 60]",
                    pickable=False,
                )
            ]

            if not expand_candidates.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=expand_candidates,
                        get_position="[lon, lat]",
                        get_radius="marker_radius_blue",
                        get_fill_color="[0, 153, 255, 220]",  # 파랑
                        pickable=True,
                    )
                )

            if not focus_candidates.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=focus_candidates,
                        get_position="[lon, lat]",
                        get_radius="marker_radius_red",
                        get_fill_color="[255, 0, 0, 220]",  # 빨강
                        pickable=True,
                    )
                )

            view_state = pdk.ViewState(
                latitude=float(emd_geo["lat"].mean()),
                longitude=float(emd_geo["lon"].mean()),
                zoom=10.5,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        "text": "읍·면·동: {읍면동}\n"
                                "시설 수 (천 명당): {시설_천명당:.2f}\n"
                                "위험지수: {위험지수:.2f}"
                    },
                )
            )

            # (6) 결론 요약 (표)
            st.markdown("#### (6) 결론 요약 (표)")
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**관리 집중 대상 구역 (복지시설 많고 대기질이 나쁜 곳)**")
                if focus_candidates.empty:
                    st.write("선정된 관리 집중 대상 구역이 없습니다.")
                else:
                    focus_table = focus_candidates[
                        ["읍면동", "시설_천명당", "위험지수"]
                    ].rename(
                        columns={
                            "읍면동": "읍·면·동",
                            "시설_천명당": "시설 수 (천 명당)",
                            "위험지수": "환경 위험 지수",
                        }
                    )
                    st.dataframe(focus_table, use_container_width=True)

            with col_right:
                st.markdown("**시설 증설 대상 구역 (청정하지만 복지시설이 부족한 곳)**")
                if expand_candidates.empty:
                    st.write("선정된 시설 증설 대상 구역이 없습니다.")
                else:
                    expand_table = expand_candidates[
                        ["읍면동", "시설_천명당", "위험지수"]
                    ].rename(
                        columns={
                            "읍면동": "읍·면·동",
                            "시설_천명당": "시설 수 (천 명당)",
                            "위험지수": "환경 위험 지수",
                        }
                    )
                    st.dataframe(expand_table, use_container_width=True)

        else:
            st.info(
                "노인복지시설 충족도(시설_천명당)와 위험지수, 좌표가 모두 있는 읍·면·동이 없어 결론 지도를 그릴 수 없습니다."
            )


if __name__ == "__main__":
    main()
