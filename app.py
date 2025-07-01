from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')  # Uncomment if using scaling

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data = scaler.transform([data])  
    prob = model.predict_proba([data])[0][1]
    threshold = 0.6
    prediction = int(prob >= threshold)
    
    return render_template('index.html', prediction_text=f'Fraud Detected: {bool(prediction)} | Probability: {prob:.4f}')

if __name__ == "__main__":
    app.run(debug=True)
