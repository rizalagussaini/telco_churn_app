import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("📊 Aplikasi Prediksi Pelanggan Churn (Telco)")

st.sidebar.header("📝 Isi Data Pelanggan")

def user_input():
    SeniorCitizen = st.sidebar.radio("Apakah pelanggan lansia?", ['Tidak', 'Iya'])
    Partner = st.sidebar.radio("Memiliki pasangan?", ['Iya', 'Tidak'])
    Dependents = st.sidebar.radio("Memiliki tanggungan?", ['Iya', 'Tidak'])
    PhoneService = st.sidebar.radio("Menggunakan layanan telepon?", ['Iya', 'Tidak'])
    MultipleLines = st.sidebar.selectbox("Layanan telepon ganda", ['Tidak ada layanan telepon', 'Iya', 'Tidak'])
    InternetService = st.sidebar.selectbox("Jenis layanan internet", ['DSL', 'Fiber optic', 'Tidak ada'])
    OnlineSecurity = st.sidebar.selectbox("Keamanan online", ['Iya', 'Tidak', 'Tidak ada layanan internet'])
    OnlineBackup = st.sidebar.selectbox("Backup online", ['Iya', 'Tidak', 'Tidak ada layanan internet'])
    DeviceProtection = st.sidebar.selectbox("Perlindungan perangkat", ['Iya', 'Tidak', 'Tidak ada layanan internet'])
    TechSupport = st.sidebar.selectbox("Bantuan teknis", ['Iya', 'Tidak', 'Tidak ada layanan internet'])
    StreamingTV = st.sidebar.selectbox("Layanan streaming TV", ['Iya', 'Tidak', 'Tidak ada layanan internet'])
    StreamingMovies = st.sidebar.selectbox("Layanan streaming film", ['Iya', 'Tidak', 'Tidak ada layanan internet'])
    Contract = st.sidebar.selectbox("Jenis kontrak", ['Bulanan', '1 Tahun', '2 Tahun'])
    PaperlessBilling = st.sidebar.radio("Penagihan tanpa kertas?", ['Iya', 'Tidak'])
    PaymentMethod = st.sidebar.selectbox("Metode pembayaran", [
        'Cek elektronik', 'Cek pos', 'Transfer otomatis (Bank)', 'Kartu kredit otomatis'
    ])
    #MonthlyCharges = st.sidebar.number_input("Biaya bulanan (USD)", min_value=0.0)
    #TotalCharges = st.sidebar.number_input("Total biaya selama ini (USD)", min_value=0.0)
    #monthly_rp = st.sidebar.number_input("Biaya bulanan (Rp)", min_value=0.0, value=160000.0, step=1000.0)
    #total_rp = st.sidebar.number_input("Total biaya selama ini (Rp)", min_value=0.0, value=1600000.0, step=1000.0)

    # Konversi ke USD
    #MonthlyCharges = monthly_rp / 16000
    #TotalCharges = total_rp / 16000

    import locale
    locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')  # Format Indonesia

    # Input sebagai teks dengan format ribuan
    monthly_rp_str = st.sidebar.text_input("Biaya bulanan (Rp)", "160.000")
    total_rp_str = st.sidebar.text_input("Total biaya selama ini (Rp)", "1.600.000")

    # Hapus titik dan konversi ke float
    monthly_rp = float(monthly_rp_str.replace(".", "").replace(",", "."))
    total_rp = float(total_rp_str.replace(".", "").replace(",", "."))

    # Konversi ke USD
    MonthlyCharges = monthly_rp / 16000
    TotalCharges = total_rp / 16000



    tenure_group = st.sidebar.selectbox("Lama berlangganan", [
        '1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'
    ])

    # Konversi input ke bentuk yang sesuai model
    data = {
        'SeniorCitizen': 1 if SeniorCitizen == 'Iya' else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': 'Month-to-month' if Contract == 'Bulanan' else ('One year' if Contract == '1 Tahun' else 'Two year'),
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': (
            'Electronic check' if PaymentMethod == 'Cek elektronik' else
            'Mailed check' if PaymentMethod == 'Cek pos' else
            'Bank transfer (automatic)' if PaymentMethod == 'Transfer otomatis (Bank)' else
            'Credit card (automatic)'
        ),
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'tenure_group': tenure_group
    }

    return pd.DataFrame([data])

# Ambil input user
input_df = user_input()

# One-hot encoding
input_encoded = pd.get_dummies(input_df)

# Sesuaikan fitur agar cocok dengan model
model_features = model.get_booster().feature_names
for col in set(model_features) - set(input_encoded.columns):
    input_encoded[col] = 0
input_encoded = input_encoded[model_features]

# Prediksi
prediction = model.predict(input_encoded)
proba = model.predict_proba(input_encoded)

# Output hasil prediksi
st.subheader("🔍 Hasil Prediksi")
st.write("**Status Pelanggan:**", "🚫 Churn" if prediction[0] == 1 else "✅ Tidak Churn")
st.write("**Probabilitas Churn:**", f"{proba[0][1]*100:.2f}%")
