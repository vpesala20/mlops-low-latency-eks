import numpy as np
from flask import Flask, request, jsonify

# Load the dummy model (replace with your actual model loading logic)
# Since we don't have a real model, we create a dummy prediction function.
def dummy_predict(data):
    """Simulates a model prediction by returning the sum of features."""
    return np.sum(data, axis=1) * 1.5

# Initialize the Flask application
app = Flask(__name__)

# --- NEW ROOT ROUTE ADDED ---
@app.route('/', methods=['GET'])
def status_check():
    """Endpoint for a simple status check."""
    return jsonify({"status": "OK", "message": "MLOps Model Service is running. Use /predict POST endpoint for inference."}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for receiving data and returning predictions."""
    if request.is_json:
        data = request.get_json(silent=True)
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid input format. Expecting JSON with 'features' key."}), 400

        try:
            features = np.array(data['features'])

            if features.ndim == 1:
                features = features.reshape(1, -1)

            predictions = dummy_predict(features)
            results = predictions.tolist()

            return jsonify({"predictions": results})
        except Exception as e:
            return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

    return jsonify({"error": "Request body must be JSON."}), 400

# Run the app if executed directly
if __name__ == '__main__':
    # Flask runs inside the container, so it must bind to 0.0.0.0
    app.run(host='0.0.0.0', port=8080)
