import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('model.pkl')

st.title("📊 Prediksi Churn Pelanggan Telco ")

st.write("Masukkan data pelanggan untuk memprediksi apakah pelanggan akan churn atau tidak.")

# Form input data
with st.form("churn_form"):
    SeniorCitizen = st.selectbox("Apakah pelanggan lansia?", ["Yes", "No"])
    Partner = st.selectbox("Memiliki pasangan?", ["Yes", "No"])
    Dependents = st.selectbox("Memiliki tanggungan?", ["Yes", "No"])
    PhoneService = st.selectbox("Menggunakan layanan telepon?", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Jenis layanan internet", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Jenis kontrak", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
    PaymentMethod = st.selectbox("Metode Pembayaran", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)
    tenure_group = st.selectbox("Tenure Group (lama berlangganan)", [
        "1 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"
    ])

    submitted = st.form_submit_button("Prediksi")

# Ketika tombol ditekan
if submitted:
    input_dict = {
        'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
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
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'tenure_group': tenure_group
    }

    # Konversi ke DataFrame
    df_input = pd.DataFrame([input_dict])

    # One-hot encoding
    cat_cols = df_input.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df_input, columns=cat_cols)

    # Lengkapi fitur yang dibutuhkan model
    model_features = model.get_booster().feature_names
    for col in set(model_features) - set(df_encoded.columns):
        df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    # Prediksi
    prediction = model.predict(df_encoded)[0]
    proba = model.predict_proba(df_encoded)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Pelanggan kemungkinan akan **CHURN** (Probabilitas: {proba:.2f})")
    else:
        st.success(f"✅ Pelanggan kemungkinan **TIDAK CHURN** (Probabilitas: {proba:.2f})")
