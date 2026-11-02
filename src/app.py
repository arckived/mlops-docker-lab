"""Flask service for the Wine Quality Sommelier.

Pages:   /            prediction form (plain HTML form, no JavaScript)
         /metrics     model evaluation from training
         /history     predictions logged to the database
API:     POST /api/predict, GET /api/metrics, GET /api/history, GET /health
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import db

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
PORT = int(os.getenv("PORT", "5000"))

model = joblib.load(MODEL_DIR / "model.joblib")
with open(MODEL_DIR / "metrics.json") as f:
    METRICS = json.load(f)

FEATURES = METRICS["features"]
INPUT_FEATURES = [f for f in FEATURES if f != "is_red"]
CLASSES = list(model.classes_)

db.init_db()

app = Flask(__name__)


@app.template_filter("pretty")
def pretty_feature(name: str) -> str:
    """Display name for a feature column, e.g. 'is_red' -> 'Wine type'."""
    if name == "is_red":
        return "Wine type"
    if name == "pH":
        return name
    return name[:1].upper() + name[1:]


@app.context_processor
def inject_metrics():
    return {"metrics": METRICS}


def parse_input(payload: dict) -> tuple[str, dict]:
    """Validate a request body and return (wine_type, feature dict)."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    wine_type = str(payload.get("wine_type", "red")).lower()
    if wine_type not in ("red", "white"):
        raise ValueError("wine_type must be 'red' or 'white'")
    values = {}
    missing = []
    for name in INPUT_FEATURES:
        if name not in payload:
            missing.append(name)
            continue
        try:
            values[name] = float(payload[name])
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number")
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    return wine_type, values


def typical_values(wine_type: str) -> dict:
    """Median measurements of a red or white wine, used to prefill the form."""
    key = "median_red" if wine_type == "red" else "median_white"
    return {f: round(METRICS["feature_stats"][f][key], 4) for f in INPUT_FEATURES}


def predict(wine_type: str, values: dict) -> dict:
    """Run the model on one wine, save the result to the database and return it."""
    row = {**values, "is_red": 1 if wine_type == "red" else 0}
    X = pd.DataFrame([row], columns=FEATURES)
    probs = model.predict_proba(X)[0]
    probabilities = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}
    predicted = max(probabilities, key=probabilities.get)
    prediction_id = db.log_prediction(wine_type, values, predicted, probabilities[predicted])
    return {
        "id": prediction_id,
        "wine_type": wine_type,
        "predicted_class": predicted,
        "confidence": probabilities[predicted],
        "probabilities": {c: probabilities[c] for c in METRICS["classes"]},
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        # The HTML form sends every field as text; parse_input converts and checks them.
        values = request.form.to_dict()
        wine_type = values.get("wine_type", "red")
        try:
            wine_type, parsed = parse_input(values)
            result = predict(wine_type, parsed)
        except ValueError as e:
            error = str(e)
    else:
        wine_type = request.args.get("type", "red")
        if wine_type not in ("red", "white"):
            wine_type = "red"
        values = typical_values(wine_type)

    return render_template(
        "index.html",
        features=INPUT_FEATURES,
        stats=METRICS["feature_stats"],
        importance=METRICS["feature_importance"],
        wine_type=wine_type,
        values=values,
        result=result,
        error=error,
    )


@app.get("/metrics")
def metrics_page():
    return render_template("metrics.html", classes=METRICS["classes"])


@app.get("/history")
def history_page():
    return render_template("history.html", rows=db.recent_predictions(50), counts=db.class_counts())


@app.post("/api/predict")
def api_predict():
    try:
        wine_type, values = parse_input(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(predict(wine_type, values))


@app.get("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.get("/api/history")
def api_history():
    limit = min(int(request.args.get("limit", 50)), 500)
    return jsonify(db.recent_predictions(limit))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": METRICS["selected_model"], "trained_at": METRICS["trained_at"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
