import json
import os
from pathlib import Path

from model_runtime import load_artifact, predict_records


MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).with_name("model.pkl")))
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


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def extract_payload(event):
    if "body" in event:
        body = event["body"]
        if isinstance(body, str):
            return json.loads(body)
        return body
    return event


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


def lambda_handler(event, context):
    try:
        payload = extract_payload(event)
        records = extract_records(payload)
        predictions = predict_records(records, get_artifact())
        return response(200, {"count": len(predictions), "predictions": predictions})
    except Exception as exc:
        return response(400, {"error": str(exc)})
