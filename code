import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 페이지 설정
st.set_page_config(page_title="상권 분석 AI", layout="wide")

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_excel("Seoul_Data.xlsx")
    return df

df = load_data()

# 타겟 변수 생성
df['target'] = (df['closure_rate'] > df['closure_rate'].mean()).astype(int)

# 인코딩용 복사본 생성
df_encoded = df.copy()
le_district = LabelEncoder()
le_neighborhood = LabelEncoder()
le_category = LabelEncoder()
df_encoded['district'] = le_district.fit_transform(df['district'])
df_encoded['neighborhood'] = le_neighborhood.fit_transform(df['neighborhood'])
df_encoded['category'] = le_category.fit_transform(df['category'])

# 특성 선택
features = [
    'store_count', 'avg_operating_years', 'rent_price',
    'traffic', 'avg_annual_sales',
    'district', 'neighborhood', 'category'
]
X = df_encoded[features]
y = df_encoded['target']

# 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# UI 시작
st.title("📊 지역 상권 폐업 예측 AI")
st.markdown(f"### ✅ 모델 정확도: **{accuracy*100:.2f}%**")

with st.sidebar:
    st.header("📍 지역 선택")
    selected_district = st.selectbox("구", sorted(df['district'].unique()))
    neighborhoods_in_district = df[df['district'] == selected_district]['neighborhood'].unique()
    selected_neighborhood = st.selectbox("동", sorted(neighborhoods_in_district))
    selected_category = st.selectbox("업종", sorted(df['category'].unique()))

# 선택된 지역의 평균 데이터 추출
selected_row = df[
    (df['district'] == selected_district) &
    (df['neighborhood'] == selected_neighborhood) &
    (df['category'] == selected_category)
]

if selected_row.empty:
    st.warning("선택한 지역과 업종의 데이터가 없습니다. 다른 조합을 선택해 주세요.")
else:
    # 예측 입력 준비
    input_df = pd.DataFrame([[
        selected_row['store_count'].values[0],
        selected_row['avg_operating_years'].values[0],
        selected_row['rent_price'].values[0],
        selected_row['traffic'].values[0],
        selected_row['avg_annual_sales'].values[0],
        le_district.transform([selected_district])[0],
        le_neighborhood.transform([selected_neighborhood])[0],
        le_category.transform([selected_category])[0]
    ]], columns=features)

    # 예측 실행
    prediction = model.predict(input_df)[0]
    sales = selected_row['avg_annual_sales'].values[0]
    sales_mean = df['avg_annual_sales'].mean()

    # 등급 판별
    if prediction == 0:
        label = "✅ 폐업 위험 낮음"
        grade = "A (안정)"
    else:
        if sales >= sales_mean:
            label = "⚠️ 폐업 위험 높음"
            grade = "B (경고)"
        else:
            label = "⚠️ 폐업 위험 높음"
            grade = "C (위험)"

    # 결과 출력
    st.subheader("📈 예측 결과")
    st.markdown(f"### {label}")
    st.markdown(f"**📌 등급: `{grade}`**")

    # 선택된 동의 실제 수치도 보여줌
    with st.expander("🔎 선택한 지역 정보 보기"):
        st.write(selected_row[[
            'store_count', 'avg_operating_years', 'rent_price',
            'traffic', 'avg_annual_sales'
        ]].rename(columns={
            'store_count': '상점 수',
            'avg_operating_years': '평균 운영 연수',
            'rent_price': '임대료(만원)',
            'traffic': '유동인구',
            'avg_annual_sales': '평균 연매출(만원)'
        }))
