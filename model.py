import numpy as np
import joblib
import time
import psutil
from flask import Flask, request, jsonify
from prometheus_client import generate_latest, Gauge

# --- Prometheus Metrics Setup ---
PREDICTION_GAUGE = Gauge('ml_model_predictions_total', 'Total number of predictions made')
ERROR_GAUGE = Gauge('ml_model_errors_total', 'Total number of prediction errors')
LATENCY_GAUGE = Gauge('ml_model_average_latency_ms', 'Average prediction latency in milliseconds')
LAST_PREDICTION_TIMESTAMP = Gauge('ml_model_last_prediction_timestamp', 'Timestamp of the last prediction')
CPU_GAUGE = Gauge('system_cpu_utilization_percent', 'CPU utilization percentage')
MEMORY_GAUGE = Gauge('system_memory_percent', 'Memory utilization percentage')
MEMORY_USED_MB_GAUGE = Gauge('system_memory_used_mb', 'Memory used in MB')
UPTIME_GAUGE = Gauge('system_uptime_seconds', 'Uptime in seconds')
MODEL_STATUS_GAUGE = Gauge('model_status', 'Status of the model (1=LOADED, 0=FAILED)')
START_TIME = time.time()

# --- Model Loading and Status ---
MODEL = None
MODEL_LOADED = False
try:
    # Use joblib to load the model (as confirmed by the logs)
    MODEL = joblib.load('model.joblib')
    MODEL_LOADED = True
    MODEL_STATUS_GAUGE.set(1)
    print("SUCCESS: Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    MODEL_STATUS_GAUGE.set(0)

app = Flask(__name__)

# --- Core Prediction Logic ---
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if not MODEL_LOADED:
        ERROR_GAUGE.inc()
        return jsonify({"error": "Model is not loaded."}), 503

    try:
        data = request.get_json(force=True)
        # Expecting data format: {"features": [[10, 20, 30, 40]]}
        features = data.get('features')
        
        if features is None:
            ERROR_GAUGE.inc()
            return jsonify({"error": "Invalid input format. Expecting JSON with 'features' key."}), 400

        # FIX: Ensure input is a 2D NumPy array of float type
        input_array = np.array(features, dtype=float)

        # Make prediction
        prediction = MODEL.predict(input_array).tolist()

        # Log metrics
        latency = (time.time() - start_time) * 1000  # ms
        PREDICTION_GAUGE.inc()
        LATENCY_GAUGE.set(latency)
        LAST_PREDICTION_TIMESTAMP.set(int(time.time()))

        return jsonify({"predictions": prediction})

    except Exception as e:
        ERROR_GAUGE.inc()
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# --- Metrics Endpoint ---
@app.route('/metrics', methods=['GET'])
def metrics():
    # Update system metrics before serving
    CPU_GAUGE.set(psutil.cpu_percent())
    MEMORY_GAUGE.set(psutil.virtual_memory().percent)
    MEMORY_USED_MB_GAUGE.set(psutil.virtual_memory().used / (1024 * 1024))
    UPTIME_GAUGE.set(time.time() - START_TIME)
    
    # Custom aggregation for the JSON view (we are retrieving the current gauge values)
    prediction_count = PREDICTION_GAUGE.collect()[0].samples[0].value if PREDICTION_GAUGE.collect() else 0
    error_count = ERROR_GAUGE.collect()[0].samples[0].value if ERROR_GAUGE.collect() else 0
    avg_latency = LATENCY_GAUGE.collect()[0].samples[0].value if LATENCY_GAUGE.collect() else 0
    last_timestamp = LAST_PREDICTION_TIMESTAMP.collect()[0].samples[0].value if LAST_PREDICTION_TIMESTAMP.collect() else None
    
    # Convert uptime, memory, cpu to current values
    uptime_seconds = time.time() - START_TIME
    model_status_str = "LOADED" if MODEL_LOADED else "FAILED_TO_LOAD"

    return jsonify({
        "model_status": model_status_str,
        "prediction_metrics": {
            "total_predictions": int(prediction_count),
            "total_error_count": int(error_count),
            "average_latency_ms": round(avg_latency, 2),
            "last_prediction_timestamp": int(last_timestamp) if last_timestamp else None
        },
        "system_metrics": {
            "cpu_utilization_percent": round(psutil.cpu_percent(), 1),
            "memory_percent": round(psutil.virtual_memory().percent, 1),
            "memory_used_mb": round(psutil.virtual_memory().used / (1024 * 1024), 2)
        },
        "uptime_seconds": round(uptime_seconds, 2)
    })

if __name__ == '__main__':
    # Flask development server runs on port 8080 (as defined in Dockerfile)
    app.run(host='0.0.0.0', port=8080, debug=True)
