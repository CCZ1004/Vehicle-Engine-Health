from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Load the model and scaler
model = load_model("engine_health_model.h5")

# Ensure the scaler matches what was used during training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    # Serve the HTML page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)  # Reshape for a single prediction

        # Scale input features
        input_scaled = scaler.transform(features)

        # Make a prediction
        prediction = model.predict(input_scaled)
        binary_output = int(prediction >= 0.47)  # Threshold for binary classification

        # Create a response with the result
        response = {
            "predicted_label": "Good Health" if binary_output == 1 else "Low Health",
            "confidence": float(prediction[0])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
