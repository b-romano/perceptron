from flask import Flask, request, jsonify
import joblib
from sklearn.linear_model import Perceptron

app = Flask(__name__)

# Załaduj model perceptron
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)