"""Streamlit dashboard for the Wine Quality Sommelier.

Three tabs:
    Predict  enter a wine's measurements and get Low / Medium / High quality
    Model    evaluation results saved by the trainer container
    History  predictions saved in the database
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import db

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "artifacts"))

st.set_page_config(page_title="Wine Quality Sommelier", page_icon="🍷", layout="wide")


# ---------- Load the model, metrics and database once ----------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model.joblib")


@st.cache_data
def load_metrics():
    with open(MODEL_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_resource
def setup_database():
    db.init_db()
    return True


model = load_model()
metrics = load_metrics()
setup_database()

FEATURES = metrics["features"]                          # all 12 model inputs
INPUT_FEATURES = [f for f in FEATURES if f != "is_red"]  # the 11 measurements
CLASSES = metrics["classes"]                            # ["Low", "Medium", "High"]
STATS = metrics["feature_stats"]


def pretty(name: str) -> str:
    """Display name for a feature column."""
    if name == "is_red":
        return "Wine type"
    if name == "pH":
        return name
    return name[:1].upper() + name[1:]


def predict(wine_type: str, values: dict) -> dict:
    """Run the model on one wine and return the class and probabilities."""
    row = {**values, "is_red": 1 if wine_type == "Red" else 0}
    X = pd.DataFrame([row], columns=FEATURES)
    probs = model.predict_proba(X)[0]
    probabilities = {cls: float(p) for cls, p in zip(model.classes_, probs)}
    predicted = max(probabilities, key=probabilities.get)
    return {"predicted_class": predicted, "confidence": probabilities[predicted], "probabilities": probabilities}


# ---------- Page ----------

st.title("🍷 Wine Quality Sommelier")
st.caption(
    f"Model: {metrics['selected_model']}, trained {metrics['trained_at']} "
    f"on {metrics['n_train'] + metrics['n_test']} wines from the UCI Wine Quality dataset"
)

predict_tab, model_tab, history_tab = st.tabs(["Predict", "Model", "History"])


# ---------- Tab 1: Predict ----------

with predict_tab:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Describe the wine")
        wine_type = st.radio("Wine type", ["Red", "White"], horizontal=True)
        st.write("The sliders start at the typical value for the selected wine type.")

        median_key = "median_red" if wine_type == "Red" else "median_white"
        values = {}
        slider_cols = st.columns(2)
        for i, feature in enumerate(INPUT_FEATURES):
            s = STATS[feature]
            with slider_cols[i % 2]:
                values[feature] = st.slider(
                    pretty(feature),
                    min_value=float(s["min"]),
                    max_value=float(s["max"]),
                    value=float(s[median_key]),
                    step=(s["max"] - s["min"]) / 200,
                    format="%.3f",
                    # A different key per wine type resets the sliders when the type changes.
                    key=f"{wine_type}_{feature}",
                )

        clicked = st.button("Rate this wine", type="primary")

    with right:
        st.subheader("Prediction")
        if clicked:
            result = predict(wine_type, values)
            prediction_id = db.log_prediction(
                wine_type.lower(), values, result["predicted_class"], result["confidence"]
            )
            st.metric("Predicted quality", result["predicted_class"])
            st.write(f"The model is **{result['confidence']:.0%}** confident.")
            for cls in CLASSES:
                p = result["probabilities"][cls]
                st.progress(p, text=f"{cls}: {p:.0%}")
            st.success(f"Saved as prediction #{prediction_id}. See it in the History tab.")
        else:
            st.info("Set the measurements and press **Rate this wine**.")

        st.subheader("What the model looks at")
        st.write("How much the macro F1 score drops when each feature is shuffled. Bigger means more important.")
        importance = pd.DataFrame(metrics["feature_importance"])
        importance["feature"] = importance["feature"].map(pretty)
        st.dataframe(
            importance,
            hide_index=True,
            use_container_width=True,
            height=(len(importance) + 1) * 35 + 3,  # tall enough to show every row
            column_config={
                "feature": "Feature",
                "importance": st.column_config.ProgressColumn(
                    "Importance", format="%.3f", min_value=0, max_value=float(importance["importance"].max())
                ),
            },
        )


# ---------- Tab 2: Model ----------

with model_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", metrics["selected_model"])
    c2.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    c3.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    c4.metric("Train / test rows", f"{metrics['n_train']} / {metrics['n_test']}")

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Candidate models")
        st.write("The trainer container trains both. The one with the higher macro F1 is saved and served.")
        candidates = pd.DataFrame(metrics["candidates"]).T.rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro F1"})
        st.dataframe(candidates, use_container_width=True)

        st.subheader("Scores for each class")
        per_class = pd.DataFrame(metrics["per_class"]).T[["precision", "recall", "f1-score"]]
        per_class.columns = ["Precision", "Recall", "F1"]
        per_class["Wines in dataset"] = [metrics["class_distribution"][c] for c in per_class.index]
        st.dataframe(per_class, use_container_width=True)
        st.caption("Low = quality 3 to 5, Medium = 6, High = 7 to 9.")

    with right:
        st.subheader("Confusion matrix")
        st.write("Rows are the true class, columns are what the model predicted on the test set.")
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=[f"True {c}" for c in CLASSES],
            columns=[f"Predicted {c}" for c in CLASSES],
        )
        st.dataframe(cm, use_container_width=True)


# ---------- Tab 3: History ----------

with history_tab:
    counts = db.class_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions saved", sum(counts.values()))
    c2.metric("Low", counts.get("Low", 0))
    c3.metric("Medium", counts.get("Medium", 0))
    c4.metric("High", counts.get("High", 0))

    rows = db.recent_predictions(50)
    if rows:
        history = pd.DataFrame(
            {
                "#": r["id"],
                "Time (UTC)": r["created_at"],
                "Type": r["wine_type"].capitalize(),
                "Alcohol": r["features"]["alcohol"],
                "Volatile acidity": r["features"]["volatile acidity"],
                "Sulphates": r["features"]["sulphates"],
                "Prediction": r["predicted_class"],
                "Confidence": f"{r['confidence']:.0%}",
            }
            for r in rows
        )
        st.dataframe(history, hide_index=True, use_container_width=True)
    else:
        st.info("No predictions yet. Make one in the Predict tab.")
