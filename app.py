import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="HeartCare AI - Diagnostic Center",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    
    [data-testid="stMetric"] {
        background-color: #1a1a1a !important;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid #333;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: bold;
    }

    [data-testid="stMetricLabel"] {
        color: #bbbbbb !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource 
def load_assets():
    try:
        with open('best_heart_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler_heart.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: File model atau scaler tidak ditemukan!")
        return None, None

model, scaler = load_assets()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("Informasi Sistem")
    st.info("""
    Sistem ini menggunakan algoritma **Random Forest** untuk memprediksi risiko penyakit jantung berdasarkan data klinis pasien. 
    """)
    st.divider()
    st.write("**Metrik Model:**")
    st.write("- Accuracy: 97.30%")
    st.write("- ROC-AUC: 0.9720")

col_header1, col_header2 = st.columns([1, 4])
with col_header1:
    st.write("") 
with col_header2:
    st.title("🏥 HeartCare Analytics Center")
    st.subheader("Sistem Cerdas Klasifikasi Risiko Penyakit Jantung")

st.write("---")

with st.container():
    with st.form("input_form"):
        st.markdown("### 📋 Form Entri Data Klinis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### **Indikator Vital**")
            age = st.number_input("1. Usia (20-85)", 20, 85, 44)
            gender = st.selectbox("2. Gender", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
            cp = st.selectbox("3. Tipe Nyeri Dada", 
                              options=[(0, "Typical Angina"), (1, "Atypical Angina"), (2, "Non-anginal Pain"), (3, "Asymptomatic")],
                              format_func=lambda x: x[1])[0]
            rbp = st.number_input("4. Tekanan Darah Sistolik (mmHg)", 80, 200, 120)
            chol = st.number_input("5. Kolesterol Total (mg/dl)", 100, 500, 200)
            fbs = st.selectbox("6. Gula Darah Puasa > 120 mg/dl", options=[(1, "Ya (True)"), (0, "Tidak (False)")], format_func=lambda x: x[1])[0]

        with col2:
            st.markdown("##### **Gaya Hidup & Fisik**")
            mhr = st.number_input("7. Detak Jantung Maksimal (MaxHR)", 60, 220, 150)
            exang = st.selectbox("8. Nyeri Dada saat Olahraga", options=[(1, "Ya"), (0, "Tidak")], format_func=lambda x: x[1])[0]
            smoke = st.selectbox("9. Merokok", options=[(1, "Ya"), (0, "Tidak")], format_func=lambda x: x[1])[0]
            bmi = st.number_input("10. Indeks Massa Tubuh (BMI)", 10.0, 50.0, 25.0)
            fh = st.selectbox("11. Riwayat Keluarga Penyakit Jantung", options=[(1, "Ada"), (0, "Tidak Ada")], format_func=lambda x: x[1])[0]
            stress = st.slider("12. Tingkat Stress (1-10)", 1, 10, 5)
            phys = st.selectbox("13. Aktivitas Fisik", options=[(0, "Rendah"), (1, "Sedang"), (2, "Tinggi")], format_func=lambda x: x[1])[0]
        
        st.write("")
        submitted = st.form_submit_button("ANALISIS RISIKO SEKARANG")

if submitted and model is not None:
    data = [age, gender, cp, rbp, chol, fbs, mhr, exang, smoke, bmi, fh, stress, phys]
    df_input = pd.DataFrame([data])
    scaled_data = scaler.transform(df_input)
    
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]

    st.write("### 🩺 Hasil Analisis Diagnostik")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if prediction[0] == 1:
            st.error(f"## STATUS: POSITIF / BERISIKO TINGGI")
            st.write(f"Berdasarkan data yang dimasukkan, pasien memiliki indikator klinis yang kuat terhadap risiko penyakit jantung.")
        else:
            st.success(f"## STATUS: NEGATIF / RENDAH RISIKO")
            st.write(f"Berdasarkan data yang dimasukkan, pasien berada dalam kondisi kesehatan jantung yang stabil.")

    with res_col2:
        if prediction[0] == 1:
            st.metric("Probability Risk", f"{prob:.2%}", delta="High Risk", delta_color="inverse")
        else:
            st.metric("Safety Score", f"{(1-prob):.2%}", delta="Safe", delta_color="normal")

    st.divider()
    
    if prediction[0] == 1:
        st.warning("⚠️ **REKOMENDASI KLINIS:**\n1. Segera jadwalkan konsultasi dengan Dokter Spesialis Jantung.\n2. Lakukan pemeriksaan penunjang (EKG/Treadmill Test).\n3. Kurangi asupan lemak jenuh dan monitor tekanan darah harian.")
    else:
        st.info("**SARAN PENCEGAHAN:**\n1. Pertahankan pola makan sehat rendah kolesterol.\n2. Lanjutkan rutinitas olahraga minimal 150 menit per minggu.\n3. Pertahankan berat badan ideal dan kelola tingkat stres.")

st.write("---")
st.caption("© 2026 HeartCare Analytics Center | Sistem Berbasis AI")