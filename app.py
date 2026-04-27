
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Sayfa ayarı
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

BASE_DIR = Path(__file__).parent

# Model ve scaler yükleme
model = joblib.load(BASE_DIR / "best_model.pkl")
scaler = joblib.load(BASE_DIR / "selected_scaler.pkl")

selected_features = [
    "cp",
    "thal",
    "ca",
    "sex",
    "exang",
    "slope",
    "oldpeak",
    "thalach",
    "trestbps"
]

st.title("❤️ Kalp Hastalığı Risk Tahmini")
st.write(
    "Bu uygulama, makine öğrenmesi modeli kullanarak girilen klinik değerlere göre "
    "kalp hastalığı riskini tahmin eder."
)

st.warning(
    "Bu uygulama yalnızca eğitim amaçlıdır. Tıbbi teşhis veya tedavi amacıyla kullanılmamalıdır."
)

st.subheader("Hasta Bilgilerini Giriniz")

col1, col2 = st.columns(2)

with col1:
    cp = st.selectbox(
        "Göğüs Ağrısı Tipi (cp)",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Tipik anjina",
            2: "2 - Atipik anjina",
            3: "3 - Anjinal olmayan ağrı",
            4: "4 - Belirti yok / asemptomatik"
        }[x]
    )

    thal = st.selectbox(
        "Thal Değeri",
        options=[3, 6, 7],
        format_func=lambda x: {
            3: "3 - Normal",
            6: "6 - Sabit defekt",
            7: "7 - Geri dönüşümlü defekt"
        }[x]
    )

    ca = st.selectbox(
        "Floroskopi ile Görülen Damar Sayısı (ca)",
        options=[0, 1, 2, 3]
    )

    sex = st.selectbox(
        "Cinsiyet",
        options=[0, 1],
        format_func=lambda x: "Kadın" if x == 0 else "Erkek"
    )

    exang = st.selectbox(
        "Egzersize Bağlı Anjina",
        options=[0, 1],
        format_func=lambda x: "Hayır" if x == 0 else "Evet"
    )

with col2:
    slope = st.selectbox(
        "ST Segment Eğimi (slope)",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Yukarı eğimli",
            2: "2 - Düz",
            3: "3 - Aşağı eğimli"
        }[x]
    )

    oldpeak = st.number_input(
        "Oldpeak Değeri",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1
    )

    thalach = st.number_input(
        "Maksimum Kalp Atış Hızı (thalach)",
        min_value=70.0,
        max_value=220.0,
        value=150.0,
        step=1.0
    )

    trestbps = st.number_input(
        "Dinlenme Kan Basıncı (trestbps)",
        min_value=80.0,
        max_value=220.0,
        value=130.0,
        step=1.0
    )

# Tahmin butonu
if st.button("Tahmin Yap"):
    input_data = pd.DataFrame([{
        "cp": cp,
        "thal": thal,
        "ca": ca,
        "sex": sex,
        "exang": exang,
        "slope": slope,
        "oldpeak": oldpeak,
        "thalach": thalach,
        "trestbps": trestbps
    }])

    # Eğitimde kullanılan IQR clipping mantığına uygun sınırlandırma
    input_data["trestbps"] = input_data["trestbps"].clip(lower=90.0, upper=170.0)
    input_data["thalach"] = input_data["thalach"].clip(lower=84.75, upper=214.75)
    input_data["oldpeak"] = input_data["oldpeak"].clip(lower=-2.40, upper=4.00)

    input_scaled = scaler.transform(input_data[selected_features])
    input_scaled_df = pd.DataFrame(input_scaled, columns=selected_features)

    prediction = model.predict(input_scaled_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled_df)[0][1]
    else:
        probability = None

    st.subheader("Tahmin Sonucu")

    if prediction == 1:
        st.error("Model tahmini: Kalp hastalığı riski VAR")
    else:
        st.success("Model tahmini: Kalp hastalığı riski YOK")

    if probability is not None:
        st.write(f"Modelin hastalık var sınıfı için tahmini olasılığı: **{probability:.2%}**")

    st.write("---")
    st.write("Girilen değerler:")
    st.dataframe(input_data)

st.write("---")
st.caption(
    "Final model: Support Vector Machine | Test Accuracy: 0.8913 | "
    "Test Precision: 0.8333 | Test Recall: 0.9524"
)
