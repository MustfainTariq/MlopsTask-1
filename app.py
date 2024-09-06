# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return "Model API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([np.array(data['input'])])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
