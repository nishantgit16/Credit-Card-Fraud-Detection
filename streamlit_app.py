import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------------- UI Setup ----------------
st.set_page_config(page_title="Credit Card Fraud Detection System", layout="centered")

st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        h1 { color: #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Credit Card Fraud Detection System")

# ---------------- Real-Time Prediction ----------------
st.header("🔎 Enter Transaction Details")

NUM_FEATURES = 30  # Total number of features

inputs = []
with st.form("fraud_form"):
    for i in range(1, NUM_FEATURES + 1):
        value = st.number_input(f'Feature {i}', value=0.0)
        inputs.append(value)

    submitted = st.form_submit_button("Predict")

if submitted:
    data = np.array([inputs])

    # Scale 'time' (index 0) and 'amount' (index 1)
    data[:, 0] = scaler.transform(data[:, 0].reshape(-1, 1)).flatten()
    data[:, 1] = scaler.transform(data[:, 1].reshape(-1, 1)).flatten()

    prob = model.predict_proba(data)[0][1]
    threshold = 0.6
    prediction = int(prob >= threshold)

    st.write("---")
    if prediction:
        st.error(f"⚠️ Fraud Detected! Probability: {prob:.4f}")
    else:
        st.success(f"✅ Transaction is Safe. Probability: {prob:.4f}")

# ---------------- Batch Prediction ----------------
st.write("---")
st.header("📁 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV for Batch Fraud Prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
        X = df.values

        # Scale 'time' and 'amount' columns
        X[:, 0] = scaler.transform(X[:, 0].reshape(-1, 1)).flatten()
        X[:, 1] = scaler.transform(X[:, 1].reshape(-1, 1)).flatten()

        probs = model.predict_proba(X)[:, 1]
        threshold = 0.6
        predictions = (probs >= threshold).astype(int)

        df["Fraud Probability"] = probs
        df["Fraud Detected"] = predictions

        st.write("Prediction Results:")
        st.dataframe(df)

        # Option to download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", data=csv, file_name="predictions.csv", mime="text/csv")
