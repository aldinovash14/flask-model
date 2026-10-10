import json

import joblib
import numpy as np
from flask import Flask, jsonify, request 


# Load model
MODEL_PATH = "models/best_regressor_model.pkl"

model = joblib.load(MODEL_PATH)

# Nama fitur
FEATURES_NAMES = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6"
]

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Diabetes Regression API",
        "status": "running"
    })

@app.route("/api/model")
def model_info():
    # Membaca informasi model
    with open(
        "models/model_metrics.json",
        "r"
    ) as file:

        metrics = json.load(file)

    return jsonify(metrics)

@app.route("/api/predict", methods=["POST"])
def predict():
    # Mengambil data dari request
    data = request.get_json()

    # Mengambil nilai fitur sesuai urutan
    features = [
        data["age"],
        data["sex"],
        data["bmi"],
        data["bp"],
        data["s1"],
        data["s2"],
        data["s3"],
        data["s4"],
        data["s5"],
        data["s6"]
    ]

    # Mengubah data menjadi array
    input_data = np.array([features])

    # Melakukan prediksi
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    return jsonify({
        "prediction": int(prediction),
        "prediction_proba": {
            str(label): float(probability)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }
    })

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5001
        )