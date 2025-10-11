from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Gradient Boosting Housing Price API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([list(data.values())])
    prediction = model.predict(features)
    return jsonify({'predicted_price': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
