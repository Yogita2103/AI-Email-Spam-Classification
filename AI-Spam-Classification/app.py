"""
app.py - Flask backend.

Routes:
    GET  /          the website
    POST /predict   JSON {"text": "..."}  ->  prediction
    GET  /metrics   model evaluation results (from model/metrics.json)
    GET  /health    used by Render to check the app is alive
"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.predict import MAX_CHARS, load_model, predict_message

BASE_DIR = Path(__file__).resolve().parent
METRICS_FILE = BASE_DIR / "model" / "metrics.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # reject requests over 1 MB


@app.route("/")
def index():
    return render_template("index.html", max_chars=MAX_CHARS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please paste an email to analyse."}), 400
    try:
        return jsonify(predict_message(text))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Something went wrong while analysing the email."}), 500


@app.route("/metrics")
def metrics():
    if not METRICS_FILE.exists():
        return jsonify({"error": "Metrics not found. Run:  python src/train.py"}), 404
    return jsonify(json.loads(METRICS_FILE.read_text(encoding="utf-8")))


@app.route("/health")
def health():
    try:
        load_model()
        return jsonify({"status": "ok", "model_loaded": True})
    except FileNotFoundError:
        return jsonify({"status": "model_missing", "model_loaded": False}), 503


if __name__ == "__main__":
    app.run(debug=True)
