import time
import json
import psutil
from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

# --- Configuration ---
MODEL_PATH = 'model.joblib'
MODEL_LOADED = False
MODEL = None

# --- Health Metrics Class ---
class HealthMetrics:
    """Stores and computes rolling health metrics for the service."""
    
    def __init__(self):
        # Tracking prediction performance
        self.prediction_count = 0
        self.total_latency_ms = 0
        self.last_prediction_time = None
        self.start_time = time.time() # Service start time
        
        # Tracking internal errors (like data parsing issues)
        self.error_count = 0

    def record_prediction(self, latency_ms):
        """Records a successful prediction and its latency."""
        self.prediction_count += 1
        self.total_latency_ms += latency_ms
        self.last_prediction_time = time.time()

    def record_error(self):
        """Records a prediction failure."""
        self.error_count += 1

    def get_metrics(self):
        """Returns all collected metrics, including computed and system metrics."""
        
        # Compute derived metrics
        avg_latency_ms = (self.total_latency_ms / self.prediction_count) if self.prediction_count > 0 else 0
        
        # Get system metrics using psutil
        try:
            # CPU usage relative to the container/system limits
            cpu_percent = psutil.cpu_percent(interval=None) 
            # Memory usage in MB
            memory_info = psutil.virtual_memory()
            memory_used_mb = memory_info.used / (1024 * 1024)
            memory_total_mb = memory_info.total / (1024 * 1024)
            memory_percent = memory_info.percent
        except Exception:
            # Fallback if psutil fails (e.g., restricted container environment)
            cpu_percent = "N/A"
            memory_used_mb = "N/A"
            memory_percent = "N/A"

        return {
            "model_status": "LOADED" if MODEL_LOADED else "FAILED_TO_LOAD",
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "prediction_metrics": {
                "total_predictions": self.prediction_count,
                "average_latency_ms": round(avg_latency_ms, 2),
                "total_error_count": self.error_count,
                "last_prediction_timestamp": self.last_prediction_time,
            },
            "system_metrics": {
                "cpu_utilization_percent": cpu_percent,
                "memory_used_mb": round(memory_used_mb, 2) if isinstance(memory_used_mb, (int, float)) else memory_used_mb,
                "memory_percent": memory_percent,
            }
        }

# --- Initialize App and Metrics ---
app = Flask(__name__)
metrics = HealthMetrics()

# --- Model Loading ---
try:
    MODEL = load(MODEL_PATH)
    MODEL_LOADED = True
    print(f"INFO: Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    MODEL_LOADED = False

# --- Endpoints ---

@app.route('/', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    if not MODEL_LOADED:
        return jsonify({"message": "Model service running, but model failed to load.", "status": "ERROR"}), 503
        
    return jsonify({
        "message": "MLOps Model Service is running. Use /predict POST endpoint for inference.", 
        "status": "OK"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Inference endpoint. Expects JSON with features."""
    if not MODEL_LOADED:
        return jsonify({"message": "Model not available.", "status": "ERROR"}), 503

    start_time = time.time() * 1000 # Start time in milliseconds

    try:
        data = request.get_json(force=True)
        # Assuming the input data is a list of features or a single observation
        features = pd.DataFrame(data)
        
        # Ensure model expects the correct number of features (A robust check should be here)
        if features.empty:
             metrics.record_error()
             return jsonify({"error": "Input data is empty or invalid format."}), 400

        # Perform prediction
        predictions = MODEL.predict(features).tolist()
        
        end_time = time.time() * 1000 # End time in milliseconds
        latency_ms = end_time - start_time
        
        # Record successful prediction metrics
        metrics.record_prediction(latency_ms)
        
        return jsonify({
            "predictions": predictions, 
            "latency_ms": round(latency_ms, 2)
        })

    except Exception as e:
        metrics.record_error()
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Exposes current performance and system health metrics."""
    return jsonify(metrics.get_metrics())

if __name__ == '__main__':
    # When running locally for testing
    app.run(debug=True, host='0.0.0.0', port=8080)
