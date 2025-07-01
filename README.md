# 📄 Credit Card Fraud Detection System

A real-world, machine learning-powered solution for detecting fraudulent credit card transactions. This project not only builds an accurate fraud detection model but also deploys it as an interactive web application using Streamlit — providing both real-time single predictions and batch predictions via CSV uploads.

---

## 📝 Project Highlights

👉 Real-world dataset for fraud detection\
👉 Preprocessing with outlier handling and scaling\
👉 Multiple ML models tested: XGBoost, LightGBM, Voting Classifier\
👉 SMOTE applied to handle class imbalance\
👉 Model optimized for **high recall** without sacrificing precision\
👉 Fully functional Streamlit web app\
👉 Deployed on Streamlit Cloud

---

## 📊 Problem Statement

Credit card fraud is a critical financial crime, leading to billions in losses globally. Detecting fraud in real-time is challenging due to:

- Highly imbalanced datasets
- Evolving fraud patterns
- Need for both high precision and high recall

This project addresses these challenges by building a robust ML pipeline capable of accurately flagging fraudulent transactions while minimizing false positives.

---

## 📁 Dataset Overview

**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions
- 492 fraudulent cases (highly imbalanced)
- Features:
  - `Time`: Seconds elapsed between transactions
  - `Amount`: Transaction amount
  - `V1` to `V28`: PCA-transformed anonymized features
  - `Class`: Target variable (1 = Fraud, 0 = Genuine)

---

## ⚙️ Technologies & Tools

| Tool/Library        | Version | Description                          |
| ------------------- | ------- | ------------------------------------ |
| Python              | 3.x     | Programming language                 |
| scikit-learn        | latest  | ML algorithms, preprocessing         |
| XGBoost             | latest  | Gradient boosting model              |
| LightGBM            | latest  | High-performance gradient boosting   |
| CatBoost            | latest  | Categorical boosting (optional)      |
| SMOTE (imblearn)    | latest  | Synthetic Minority Oversampling      |
| Streamlit           | latest  | Web application framework            |
| Pandas, NumPy       | latest  | Data handling & numerical operations |
| Matplotlib, Seaborn | latest  | Data visualization                   |

---

## 🏗️ Project Structure

```
Fraud-Detection-Streamlit/
├── app_streamlit.py              # Streamlit web app
├── fraud_detection_model.pkl     # Trained ML model
├── scaler.pkl                    # Scaler for 'Time' and 'Amount'
├── requirements.txt              # Project dependencies
├── README.md                     # Project overview (this file)
└── .gitignore                    # Exclude unnecessary files
```

---

## 🧻 Model Building Workflow

1. **Data Preprocessing**

   - Outlier handling using IQR method
   - RobustScaler applied to `Time` and `Amount` features
   - PCA-transformed features used as-is

2. **Handling Class Imbalance**

   - Applied **SMOTE** to generate synthetic fraud cases
   - Balanced training set for better model generalization

3. **Model Training & Evaluation**

   - Multiple models tested: Logistic Regression, Decision Tree, XGBoost, LightGBM
   - Hyperparameter tuning performed
   - Final model selected based on:\
     ✅ High Recall (catch more fraud)\
     ✅ Good Precision (limit false alarms)\
     ✅ Robustness on real-world data

4. **Performance Metrics (Final Model)**

   - **Recall:** \~81%
   - **Precision:** Balanced to avoid excessive false positives
   - **ROC-AUC:** Robust discrimination ability
   - SMOTE visualizations validated synthetic data distribution

---

## 🌐 Streamlit Web Application

Interactive fraud detection system with:

👉 Real-time transaction risk prediction\
👉 Batch predictions via CSV upload\
👉 Probability-based fraud alerts\
👉 Clean, responsive UI

**Local Run:**

```bash
streamlit run app_streamlit.py
```

**App Features:**

- Input all 30 transaction features manually
- Scaled `Time` and `Amount` features internally
- Model outputs fraud probability and risk classification
- Upload `.csv` files for bulk transaction analysis
- Download results with fraud predictions

---

## 🚀 Live Demo

🌍 [Deployed Application Link](https://your-streamlit-app-link-here)

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/your-username/fraud-detection-streamlit.git
cd fraud-detection-streamlit
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app_streamlit.py
```

---

## 🎍 Future Enhancements

- Integrate SHAP for model explainability
- Use advanced ensemble/hybrid models
- Deploy on production-grade platforms (e.g., AWS, Azure)
- Add user authentication for secure access

---

## 👌 Acknowledgements

Special thanks to:

- [Kaggle Dataset Contributors](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Open-source community for incredible ML libraries

---

## 📬 Contact

**Nishant Gupta**\
LinkedIn: [www.linkedin.com/in/nishantgupta68](https://www.linkedin.com/in/nishantgupta68)

