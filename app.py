import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ─── Sayfa Ayarı ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Destekli Kalp Hastalığı Risk Analizi",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Dosya Yolları & Model Yükleme ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

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
    "trestbps",
]

# ─── CSS Stilleri ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Genel Sayfa ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(160deg, #0a0f1e 0%, #111827 40%, #0f172a 100%);
}

/* ── Ana container padding ── */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* ── Hero Bölümü ── */
.hero-section {
    text-align: center;
    padding: 2.5rem 1rem 2rem 1rem;
    margin-bottom: 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #60a5fa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 400;
}

/* ── Metrik Kartları ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
@media (max-width: 768px) {
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .hero-title { font-size: 1.6rem; }
}
.metric-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}
.metric-icon {
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #f1f5f9;
}
.metric-value-blue { color: #60a5fa; }
.metric-value-green { color: #34d399; }
.metric-value-amber { color: #fbbf24; }
.metric-value-purple { color: #a78bfa; }

/* ── Bölüm Başlıkları ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
}
.section-header-icon {
    font-size: 1.3rem;
}
.section-header-text {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
}

/* ── Form Kartı ── */
.form-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(8px);
}

/* ── Streamlit Selectbox / Number Input ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background-color: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}
label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── Tahmin Butonu ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.02em;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(37, 99, 235, 0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Sonuç Kartları ── */
.result-card {
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
    backdrop-filter: blur(10px);
    animation: fadeSlideIn 0.5s ease-out;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-risk {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.result-safe {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.result-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.result-title {
    font-size: 1.3rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.result-title-risk { color: #f87171; }
.result-title-safe { color: #34d399; }
.result-desc {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.5;
}
.result-prob {
    display: inline-block;
    margin-top: 1rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-size: 0.9rem;
    color: #e2e8f0;
}
.prob-value {
    font-weight: 800;
    font-size: 1.1rem;
}

/* ── Bilgi Kartları ── */
.info-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.info-card h4 {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.info-card p, .info-card li {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.7;
}
.info-card ul {
    padding-left: 1.2rem;
    margin: 0;
}

/* ── Uyarı Kartı ── */
.warning-card {
    background: rgba(251, 191, 36, 0.06);
    border: 1px solid rgba(251, 191, 36, 0.2);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}
.warning-card-icon {
    font-size: 1.4rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.warning-card-text {
    color: #fcd34d;
    font-size: 0.85rem;
    line-height: 1.6;
    font-weight: 500;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.footer-text {
    color: #475569;
    font-size: 0.78rem;
    line-height: 1.8;
}
.footer-brand {
    color: #64748b;
    font-weight: 600;
}

/* ── Streamlit varsayılan gizleme ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2rem 0;
    border: none;
}

/* ── Data table styling ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Girdi bilgi etiketi ── */
.input-hint {
    font-size: 0.73rem;
    color: #64748b;
    margin-top: -0.5rem;
    margin-bottom: 0.8rem;
}

/* ── Değişken Açıklama Expander ── */
.glossary-container {
    margin-top: 0.5rem;
}
.glossary-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    transition: background 0.2s ease;
}
.glossary-item:hover {
    background: rgba(255, 255, 255, 0.06);
}
.glossary-var {
    font-weight: 700;
    color: #60a5fa;
    font-size: 0.88rem;
    margin-bottom: 0.25rem;
}
.glossary-desc {
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.55;
    margin: 0;
}
.glossary-desc code {
    background: rgba(59, 130, 246, 0.12);
    color: #93c5fd;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.78rem;
    font-family: 'Inter', monospace;
}
.glossary-note {
    background: rgba(251, 191, 36, 0.06);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-top: 0.8rem;
    color: #fcd34d;
    font-size: 0.78rem;
    line-height: 1.5;
    font-weight: 500;
}

/* Expander stil uyumu */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── HERO BÖLÜMÜ ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">BIM 322 — Makine Öğrenmesi ve Uygulamaları</div>
    <div class="hero-title">AI Destekli Kalp Hastalığı<br>Risk Analizi</div>
    <div class="hero-subtitle">
        Heart Disease Cleveland veri seti ile geliştirilmiş makine öğrenmesi tabanlı
        eğitim amaçlı risk tahmin sistemi
    </div>
</div>
""", unsafe_allow_html=True)

# ─── MODEL PERFORMANS METRİKLERİ ────────────────────────────────────────────────
st.markdown("""
<div class="metrics-grid">
    <div class="metric-card">
        <div class="metric-icon">🤖</div>
        <div class="metric-label">Final Model</div>
        <div class="metric-value metric-value-blue">SVM</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">🎯</div>
        <div class="metric-label">Test Accuracy</div>
        <div class="metric-value metric-value-green">89.13%</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">📊</div>
        <div class="metric-label">Test Precision</div>
        <div class="metric-value metric-value-amber">83.33%</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">🔍</div>
        <div class="metric-label">Test Recall</div>
        <div class="metric-value metric-value-purple">95.24%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── AYIRICI ────────────────────────────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ─── HASTA BİLGİLERİ FORMU & TAHMİN SONUCU ─────────────────────────────────────
col_form, col_spacer, col_result = st.columns([5, 0.5, 4.5])

with col_form:
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">🩺</span>
        <span class="section-header-text">Hasta Klinik Bilgileri</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)

    with fc1:
        cp = st.selectbox(
            "Göğüs Ağrısı Tipi (cp)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "1 — Tipik anjina",
                2: "2 — Atipik anjina",
                3: "3 — Anjinal olmayan ağrı",
                4: "4 — Asemptomatik",
            }[x],
        )

        thal = st.selectbox(
            "Thal Değeri",
            options=[3, 6, 7],
            format_func=lambda x: {
                3: "3 — Normal",
                6: "6 — Sabit defekt",
                7: "7 — Geri dönüşümlü defekt",
            }[x],
        )

        ca = st.selectbox(
            "Floroskopi Damar Sayısı (ca)",
            options=[0, 1, 2, 3],
        )

        sex = st.selectbox(
            "Cinsiyet",
            options=[0, 1],
            format_func=lambda x: "👩 Kadın" if x == 0 else "👨 Erkek",
        )

        exang = st.selectbox(
            "Egzersize Bağlı Anjina",
            options=[0, 1],
            format_func=lambda x: "Hayır" if x == 0 else "Evet",
        )

    with fc2:
        slope = st.selectbox(
            "ST Segment Eğimi (slope)",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1 — Yukarı eğimli",
                2: "2 — Düz",
                3: "3 — Aşağı eğimli",
            }[x],
        )

        oldpeak = st.number_input(
            "Oldpeak Değeri",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1,
        )

        thalach = st.number_input(
            "Maks. Kalp Atış Hızı (thalach)",
            min_value=70.0,
            max_value=220.0,
            value=150.0,
            step=1.0,
        )

        trestbps = st.number_input(
            "Dinlenme Kan Basıncı (trestbps)",
            min_value=80.0,
            max_value=220.0,
            value=130.0,
            step=1.0,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")  # küçük boşluk
    predict_clicked = st.button("🔬  Risk Analizi Yap", use_container_width=True)

    # ─── DEĞİŞKEN AÇIKLAMA BÖLÜMÜ ──────────────────────────────────────────
    with st.expander("ℹ️  Girilen Değerler Ne Anlama Geliyor?", expanded=False):
        st.markdown("""
        <div class="glossary-container">
            <div class="glossary-item">
                <div class="glossary-var">cp — Göğüs Ağrısı Tipi</div>
                <div class="glossary-desc">
                    Göğüs ağrısı tipini ifade eder. Kalp hastalığı riskini değerlendirmede önemli bir klinik belirtidir.<br>
                    <code>1</code> Tipik anjina · <code>2</code> Atipik anjina · <code>3</code> Anjinal olmayan ağrı · <code>4</code> Asemptomatik
                </div>
            </div>
            <div class="glossary-item">
                <div class="glossary-var">thal — Talasemi / Kan Akışı Testi</div>
                <div class="glossary-desc">
                    Kalp ile ilgili talasemi / kan akışı test sonucunu temsil eder.<br>
                    <code>3</code> Normal · <code>6</code> Sabit defekt · <code>7</code> Geri dönüşümlü defekt
                </div>
            </div>
            <div class="glossary-item">
                <div class="glossary-var">ca — Floroskopi Damar Sayısı</div>
                <div class="glossary-desc">
                    Floroskopi ile görüntülenen ana damar sayısını ifade eder. <code>0</code> ile <code>3</code> arasında değer alır.
                    Damar sayısı arttıkça kalp hastalığı riskiyle ilişkili olabilir.
                </div>
            </div>
            <div class="glossary-item">
                <div class="glossary-var">sex — Cinsiyet</div>
                <div class="glossary-desc">
                    Cinsiyet bilgisidir. <code>0</code> Kadın · <code>1</code> Erkek
                </div>
            </div>
            <div class="glossary-item">
                <div class="glossary-var">exang — Egzersize Bağlı Anjina</div>
                <div class="glossary-desc">
                    Egzersize bağlı anjina olup olmadığını gösterir. <code>0</code> Hayır · <code>1</code> Evet
                </div>
            </div>
            <div class="glossary-item">
                <div class="glossary-var">slope — ST Segment Eğimi</div>
                <div class="glossary-desc">
                    Egzersiz sırasında ST segmentinin eğimini ifade eder.<br>
                    <code>1</code> Yukarı eğimli · <code>2</code> Düz · <code>3</code> Aşağı eğimli
                </div>
            </div>
            <div class="glossary-item">
                <div class="glossary-var">oldpeak — ST Depresyon Değeri</div>
                <div class="glossary-desc">
                    Egzersize bağlı ST depresyon değeridir. Kalbin egzersiz sırasında verdiği elektriksel tepkiyle ilişkilidir.
                </div>
            </div>
            <div class="glossary-item">
                <div class="glossary-var">thalach — Maksimum Kalp Atış Hızı</div>
                <div class="glossary-desc">
                    Kişinin ulaştığı maksimum kalp atış hızıdır (bpm).
                </div>
            </div>
            <div class="glossary-item">
                <div class="glossary-var">trestbps — Dinlenme Kan Basıncı</div>
                <div class="glossary-desc">
                    Dinlenme halindeki kan basıncıdır (mm Hg).
                </div>
            </div>
            <div class="glossary-note">
                ⚕️ Bu değişken açıklamaları eğitim amaçlı sadeleştirilmiştir. Tıbbi yorum veya teşhis için doktor değerlendirmesi gerekir.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── TAHMİN MANTIĞI (orijinal mantık korunmuştur) ──────────────────────────────
with col_result:
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">📋</span>
        <span class="section-header-text">Tahmin Sonucu</span>
    </div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        # DataFrame oluştur — özellik sırası korunuyor
        input_data = pd.DataFrame([{
            "cp": cp,
            "thal": thal,
            "ca": ca,
            "sex": sex,
            "exang": exang,
            "slope": slope,
            "oldpeak": oldpeak,
            "thalach": thalach,
            "trestbps": trestbps,
        }])

        # Eğitimde kullanılan IQR clipping mantığına uygun sınırlandırma
        input_data["trestbps"] = input_data["trestbps"].clip(lower=90.0, upper=170.0)
        input_data["thalach"] = input_data["thalach"].clip(lower=84.75, upper=214.75)
        input_data["oldpeak"] = input_data["oldpeak"].clip(lower=-2.40, upper=4.00)

        # Ölçekleme
        input_scaled = scaler.transform(input_data[selected_features])
        input_scaled_df = pd.DataFrame(input_scaled, columns=selected_features)

        # Tahmin
        prediction = model.predict(input_scaled_df)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled_df)[0][1]
        else:
            probability = None

        # Sonuç Kartı
        if prediction == 1:
            prob_display = f"{probability:.1%}" if probability is not None else "—"
            st.markdown(f"""
            <div class="result-card result-risk">
                <div class="result-icon">⚠️</div>
                <div class="result-title result-title-risk">Kalp Hastalığı Riski Tespit Edildi</div>
                <div class="result-desc">
                    Model, girilen klinik değerlere göre kalp hastalığı riski olduğunu tahmin etmektedir.
                    Lütfen bu sonucun yalnızca eğitim amaçlı olduğunu unutmayın.
                </div>
                <div class="result-prob">
                    Hastalık Olasılığı: <span class="prob-value" style="color:#f87171;">{prob_display}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            prob_display = f"{probability:.1%}" if probability is not None else "—"
            safe_prob = f"{1 - probability:.1%}" if probability is not None else "—"
            st.markdown(f"""
            <div class="result-card result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title result-title-safe">Risk Tespit Edilmedi</div>
                <div class="result-desc">
                    Model, girilen klinik değerlere göre kalp hastalığı riski bulunmadığını tahmin etmektedir.
                    Sağlığınızı korumak için düzenli kontrollere devam edin.
                </div>
                <div class="result-prob">
                    Sağlıklı Olasılığı: <span class="prob-value" style="color:#34d399;">{safe_prob}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Girilen değerler tablosu
        st.markdown("")
        with st.expander("📄 Girilen Klinik Değerler", expanded=False):
            st.dataframe(input_data, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="result-card" style="background: rgba(255,255,255,0.03); border: 1px dashed rgba(255,255,255,0.12);">
            <div class="result-icon">🫀</div>
            <div class="result-title" style="color: #64748b;">Sonuç Bekleniyor</div>
            <div class="result-desc">
                Sol taraftaki formu doldurup <strong>"Risk Analizi Yap"</strong> butonuna tıklayın.
                Model tahmin sonucu burada görüntülenecektir.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── AYIRICI ────────────────────────────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ─── UYARI KARTI ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="warning-card">
    <div class="warning-card-icon">⚕️</div>
    <div class="warning-card-text">
        <strong>Önemli Uyarı:</strong> Bu uygulama yalnızca eğitim amaçlıdır.
        Tıbbi teşhis, tedavi veya klinik karar destek sistemi olarak kullanılmamalıdır.
        Gerçek sağlık değerlendirmeleri için uzman doktora başvurulmalıdır.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── BİLGİ KARTLARI ─────────────────────────────────────────────────────────────
info1, info2 = st.columns(2)

with info1:
    st.markdown("""
    <div class="info-card">
        <h4>🧠 Model Hakkında</h4>
        <p>
            Bu projede <strong>Support Vector Machine (SVM)</strong> modeli kullanılmıştır.
            Model, Heart Disease Cleveland veri setindeki 303 hasta kaydı üzerinde eğitilmiştir.
            9 klinik özellik kullanılarak kalp hastalığı riski binary sınıflandırma ile tahmin edilmektedir.
        </p>
        <ul>
            <li>Hedef değişken: 0 (Risk yok) / 1 (Risk var)</li>
            <li>Eğitim öncesi IQR tabanlı outlier clipping uygulanmıştır</li>
            <li>Özellikler standart ölçekleme ile normalize edilmiştir</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="info-card">
        <h4>📁 Veri Seti Bilgisi</h4>
        <p>
            <strong>Heart Disease Cleveland Dataset</strong> — UCI Machine Learning Repository
            kaynaklı bu veri seti, kardiyolojik araştırmalarda sıklıkla kullanılan referans bir veri setidir.
        </p>
        <ul>
            <li><strong>303</strong> hasta kaydı</li>
            <li><strong>14</strong> orijinal özellik</li>
            <li><strong>9</strong> seçilmiş özellik ile model eğitimi</li>
            <li>Hedef değişken binary hale dönüştürülmüştür</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-text">
        <span class="footer-brand">❤️ Kalp Hastalığı Risk Analizi</span><br>
        BIM 322 — Makine Öğrenmesi ve Uygulamaları · Final Projesi<br>
        Heart Disease Cleveland Dataset · Support Vector Machine · Python & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
