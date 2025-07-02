import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load trained model and scalers
model = joblib.load('fraud_detection_model.pkl')
scaler_time = joblib.load('scaler_time.pkl')
scaler_amount = joblib.load('scaler_amount.pkl')

# Extract XGBoost model from ensemble
xgb_model = model.named_estimators_['xgb']

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Option", ["Single Prediction", "Batch Prediction"])

# ---------------- Single Prediction ----------------
if option == "Single Prediction":
    st.title("🔍 Single Transaction Fraud Check")

    time = st.number_input("Transaction Time", value=0.0)
    amount = st.number_input("Transaction Amount", value=0.0)

    st.subheader("Other Features (28 Total)")
    inputs = []
    for i in range(1, 29):
        value = st.number_input(f'V{i}', value=0.0)
        inputs.append(value)

    if st.button("Predict Transaction"):
        scaled_time = scaler_time.transform(np.array([[time]]))[0][0]
        scaled_amount = scaler_amount.transform(np.array([[amount]]))[0][0]

        data = [scaled_time, scaled_amount] + inputs
        data_np = np.array([data])

        prob = model.predict_proba(data_np)[0][1]
        prediction = int(prob >= 0.6)

        st.write("---")
        if prediction:
            st.error(f"⚠️ Fraud Detected! Probability: {prob:.4f}")
        else:
            st.success(f"✅ Transaction is Safe. Probability: {prob:.4f}")

        st.subheader("Model Explanation (SHAP Values)")
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(data_np)

        # SHAP Barplot
        st.write("**Global Feature Importance (Barplot)**")
        fig_bar, ax = plt.subplots(figsize=(8, 5))
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig_bar)

        # SHAP Waterfall
        st.write("**Individual Prediction Explanation (Waterfall)**")
        fig_water, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_water)

# ---------------- Batch Prediction ----------------
elif option == "Batch Prediction":
    st.title("📁 Batch Fraud Detection")

    uploaded_file = st.file_uploader("Upload CSV with Transactions", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Original Uploaded Data:**")
        st.dataframe(df.head())

        if st.button("Predict Batch Transactions"):
            df['scaled_time'] = scaler_time.transform(df[['Time']])
            df['scaled_amount'] = scaler_amount.transform(df[['Amount']])

            df.drop(['Time', 'Amount'], axis=1, inplace=True)
            df.insert(0, 'scaled_time', df.pop('scaled_time'))
            df.insert(1, 'scaled_amount', df.pop('scaled_amount'))

            X = df.values
            probs = model.predict_proba(X)[:, 1]
            predictions = (probs >= 0.6).astype(int)

            df['Fraud Probability'] = probs
            df['Fraud Detected'] = predictions

            st.success("✅ Prediction Completed:")
            st.dataframe(df.head())

            # SHAP Barplot for batch
            st.subheader("📊 SHAP Global Feature Importance (Barplot)")
            background_size = min(100, X.shape[0])
            background = X[np.random.choice(X.shape[0], background_size, replace=False)]

            explainer = shap.Explainer(xgb_model)
            shap_values = explainer(background)

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, background, plot_type="bar", show=False)
            st.pyplot(fig)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
