# 📄 Credit Card Fraud Detection System

A production-ready, machine learning-powered solution for detecting fraudulent credit card transactions. This project combines model development, interpretability with SHAP, and an interactive Streamlit web app offering both real-time single predictions and batch processing.

---

## 📝 Project Highlights

🔀 Real-world Kaggle dataset for fraud detection\
🌐 Fully functional Streamlit web application\
🔧 Separate scaling pipelines for `Time` and `Amount` using RobustScaler\
🔢 Ensemble model: XGBoost, LightGBM, CatBoost combined via Voting Classifier\
🔀 SMOTE applied for class imbalance handling\
🔍 SHAP interpretability for both single and batch predictions\
🌍 Clean UI with barplots, waterfall plots, CSV uploads, and batch fraud detection

---

## 📊 Problem Statement

Credit card fraud is a major financial threat with billions lost annually. Detecting fraud is complicated due to:

- Extremely imbalanced data distributions
- Constantly evolving fraud patterns
- The need for high precision and high recall simultaneously

This project builds a robust, interpretable fraud detection pipeline to combat these issues.

---

## 📁 Dataset Overview

**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions
- 492 labeled as fraudulent (\~0.17% fraud rate)
- Features:
  - `Time`: Seconds elapsed since first transaction
  - `Amount`: Transaction amount
  - `V1` to `V28`: PCA-transformed anonymized features
  - `Class`: 1 = Fraud, 0 = Legitimate

---

## ⚙️ Technologies & Tools

| Tool/Library                | Description                              |
| --------------------------- | ---------------------------------------- |
| Python                      | Programming Language                     |
| Scikit-learn                | ML algorithms, preprocessing             |
| XGBoost, LightGBM, CatBoost | Gradient boosting & ensemble models      |
| SMOTE (imblearn)            | Synthetic data generation for balancing  |
| Streamlit                   | Frontend for real-time predictions       |
| SHAP                        | Explainable AI with feature attributions |
| Pandas, NumPy               | Data handling and transformations        |
| Matplotlib, Seaborn         | Visualization libraries                  |

---

## 🌉 Project Structure

```
Fraud-Detection-Streamlit/
├── app_streamlit.py              # Streamlit web app
├── fraud_detection_model.pkl     # Trained Voting Classifier model
├── scaler_time.pkl               # RobustScaler for 'Time'
├── scaler_amount.pkl             # RobustScaler for 'Amount'
├── requirements.txt              # Project dependencies
├── README.md                     # This project overview file
└── .gitignore                    # Ignored files setup
```

---

## 📃 Model Development Workflow

### 1. Data Preprocessing

- Outlier handling using IQR method
- Separate RobustScaler for `Time` and `Amount` features
- PCA-transformed features used without modification

### 2. Class Imbalance Handling

- Applied SMOTE to generate synthetic minority class samples
- Achieved balanced training distribution

### 3. Model Training & Evaluation

- Ensemble with XGBoost, LightGBM, and CatBoost via Voting Classifier
- Hyperparameter tuning performed to optimize Recall and Precision
- Model selection prioritized:
  - High Recall (detect more fraud)
  - Balanced Precision (minimize false alarms)

### 4. Performance Metrics (Final Model)

- **Recall:** \~81%
- **Precision:** Balanced for real-world deployment
- **ROC-AUC:** Robust class discrimination
- SHAP values validated with visual explanations

---

## 🌐 Streamlit Web Application

The web app offers:

🔹 Single Transaction Prediction:

- Enter transaction features manually
- Scales `Time` and `Amount` internally
- Displays fraud probability, SHAP Barplot, and Waterfall explanation

🔹 Batch Prediction (CSV Upload):

- Upload Kaggle-format `.csv` with raw `Time`, `Amount`, and features `V1` to `V28`
- Shows raw data preview
- Predicts fraud on all records
- Provides fraud probabilities and flags
- Displays SHAP Barplot for global feature importance
- Results downloadable as `.csv`

**Run Locally:**

```bash
streamlit run app_streamlit.py
```

---

## 🚀 Live Demo

Hosted Application: [Streamlit Cloud Link](https://credit-card-fraud-detection-by-nishant.streamlit.app/)

---

## 📦 Installation & Setup

```bash
git clone https://github.com/your-username/fraud-detection-streamlit.git
cd fraud-detection-streamlit

python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

pip install -r requirements.txt

streamlit run app_streamlit.py
```

---

## 🌽 Future Enhancements

- SHAP explanations integrated for both single and batch predictions (Implemented)
- UI/UX improvements for better user experience
- Precision-recall threshold optimization
- Cloud deployment to platforms like AWS, Azure, Streamlit Cloud
- Additional security features like authentication

---

## 📅 Acknowledgements

- [Kaggle Dataset Contributors](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Open-source ML and visualization communities

---

## 📩 Contact

**Nishant Gupta**\
[LinkedIn](https://www.linkedin.com/in/nishantgupta68)

