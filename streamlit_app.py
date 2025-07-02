import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------------- Load model and scalers ----------------
model = joblib.load('fraud_detection_model.pkl')
scaler_time = joblib.load('scaler_time.pkl')
scaler_amount = joblib.load('scaler_amount.pkl')

# ---------------- UI Setup ----------------
st.set_page_config(page_title="Credit Card Fraud Detection System", layout="wide")
st.title("🚨 Credit Card Fraud Detection System")

# ---------------- Tabs for Navigation ----------------
tab1, tab2 = st.tabs(["🔍 Single Transaction Prediction", "📁 Batch Prediction"])

# ---------------- Single Transaction Prediction ----------------
with tab1:
    st.header("Enter Transaction Details")

    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=0.0)
    features = []
    for i in range(1, 29):
        val = st.number_input(f"V{i}", value=0.0)
        features.append(val)

    if st.button("Predict Transaction"):
        scaled_time = scaler_time.transform(np.array([[time]]))[0][0]
        scaled_amount = scaler_amount.transform(np.array([[amount]]))[0][0]

        data = np.array([[scaled_time, scaled_amount] + features])

        prob = model.predict_proba(data)[0][1]
        prediction = int(prob >= 0.6)

        st.write("---")
        if prediction:
            st.error(f"⚠️ Fraud Detected! Probability: {prob:.4f}")
        else:
            st.success(f"✅ Transaction is Safe. Probability: {prob:.4f}")

        # ----------------- SHAP Explanation -----------------
        st.subheader("Model Explanation (SHAP Values)")
        explainer = shap.Explainer(model.named_estimators_['xgb'])
        shap_values = explainer(data)

        st.write("**Waterfall Plot (Individual Explanation):**")
        fig_w = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_w)

        st.write("**Bar Plot (Feature Importance):**")
        fig_b = plt.figure()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig_b)

# ---------------- Batch Prediction ----------------
with tab2:
    st.header("📁 Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload CSV (Original Kaggle Format)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Uploaded Data Preview:**")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            progress = st.progress(0)

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

            progress.progress(100)
            st.success("Prediction Completed:")

            def color_rows(row):
                color = 'background-color: #ff0000' if row['Fraud Detected'] == 1 else ''
                return [color] * len(row)

            st.dataframe(df.style.apply(color_rows, axis=1))

            st.subheader("📊 SHAP Global Feature Importance (Barplot)")
            background_size = min(100, X.shape[0])
            background = X[np.random.choice(X.shape[0], background_size, replace=False)]

            explainer = shap.Explainer(model.named_estimators_['xgb'])
            shap_values = explainer(background)

            fig_shap, ax = plt.subplots()
            shap.summary_plot(shap_values, background, plot_type="bar", show=False)
            st.pyplot(fig_shap)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")
