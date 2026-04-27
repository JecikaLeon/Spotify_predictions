import os
from pathlib import Path

from flask import Flask, jsonify, request

from model_runtime import load_artifact, predict_records


MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).with_name("model.pkl")))

app = Flask(__name__)
_artifact = None

DEFAULT_RECORD = {
    "track_id": "request_track",
    "artists": "unknown_artist",
    "album_name": "unknown_album",
    "track_name": "unknown_track",
    "track_genre": "pop",
    "duration_ms": 210000,
    "explicit": False,
    "danceability": 0.5,
    "energy": 0.5,
    "key": 0,
    "loudness": -8.0,
    "mode": 1,
    "speechiness": 0.05,
    "acousticness": 0.3,
    "instrumentalness": 0.0,
    "liveness": 0.1,
    "valence": 0.5,
    "tempo": 120.0,
    "time_signature": 4,
}

QUERY_ALIASES = {
    "GENERO": ("track_genre", str),
    "DURACION": ("duration_ms", int),
    "TRACK_ID": ("track_id", str),
    "ARTISTS": ("artists", str),
    "ARTISTA": ("artists", str),
    "ALBUM": ("album_name", str),
    "TRACK_NAME": ("track_name", str),
    "CANCION": ("track_name", str),
    "EXPLICIT": ("explicit", lambda value: str(value).lower() in {"1", "true", "yes", "si"}),
    "DANCEABILITY": ("danceability", float),
    "ENERGY": ("energy", float),
    "KEY": ("key", int),
    "LOUDNESS": ("loudness", float),
    "MODE": ("mode", int),
    "SPEECHINESS": ("speechiness", float),
    "ACOUSTICNESS": ("acousticness", float),
    "INSTRUMENTALNESS": ("instrumentalness", float),
    "LIVENESS": ("liveness", float),
    "VALENCE": ("valence", float),
    "TEMPO": ("tempo", float),
    "TIME_SIGNATURE": ("time_signature", int),
}


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


def record_from_query(args):
    record = DEFAULT_RECORD.copy()
    normalized_args = {key.upper(): value for key, value in args.items()}

    for query_name, (field_name, caster) in QUERY_ALIASES.items():
        if query_name in normalized_args:
            raw_value = normalized_args[query_name]
            if raw_value == "":
                continue
            record[field_name] = caster(raw_value)

    return record


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists()
    return jsonify({"status": "ok", "model_ready": model_ready})


@app.get("/")
def index():
    return jsonify(
        {
            "status": "ok",
            "endpoints": {
                "health": "/health",
                "predict_get": "/predict/?GENERO=pop&DURACION=210000",
                "predict_post": "/predict",
            },
        }
    )


@app.get("/predict")
@app.get("/predict/")
def predict_from_query():
    try:
        record = record_from_query(request.args)
        predictions = predict_records([record], get_artifact())
        return jsonify({"predictions": predictions})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


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
