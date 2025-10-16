from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('housing_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract features in correct order
    features = np.array([[
        data['area'], data['bedrooms'], data['bathrooms'], data['stories'],
        data['mainroad'], data['guestroom'], data['basement'],
        data['hotwaterheating'], data['airconditioning'], data['parking'],
        data['prefarea'], data['furnishingstatus_semi-furnished'],
        data['furnishingstatus_unfurnished']
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
