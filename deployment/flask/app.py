import os
from pathlib import Path

from flask import Flask, jsonify, request

from model_runtime import load_artifact, predict_records


MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).with_name("model.pkl")))

app = Flask(__name__)
_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run deployment/export_model.py first."
            )
        _artifact = load_artifact(MODEL_PATH)
    return _artifact


def extract_records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
        if isinstance(records, dict):
            return [records]
        return records
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Send a JSON object, a JSON list, or {'records': [...]} payload.")


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists()
    return jsonify({"status": "ok", "model_ready": model_ready})


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        records = extract_records(payload)
        predictions = predict_records(records, get_artifact())
        return jsonify({"count": len(predictions), "predictions": predictions})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
