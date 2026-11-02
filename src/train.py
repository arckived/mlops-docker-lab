"""Train a wine quality classifier and save it with its evaluation metrics.

Reads the red and white UCI Wine Quality CSVs, turns the 0 to 10 quality
score into three classes (Low, Medium, High), trains two candidate models,
keeps the one with the best macro F1 on the test split and writes:

    <MODEL_DIR>/model.joblib   the fitted sklearn Pipeline
    <MODEL_DIR>/metrics.json   scores, confusion matrix, feature importance
                               and per feature stats used by the dashboard
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
RANDOM_STATE = 42

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "is_red",
]
CLASSES = ["Low", "Medium", "High"]


def quality_to_class(q: int) -> str:
    if q <= 5:
        return "Low"
    if q == 6:
        return "Medium"
    return "High"


def load_data() -> pd.DataFrame:
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red["is_red"] = 1
    white["is_red"] = 0
    df = pd.concat([red, white], ignore_index=True)
    df["label"] = df["quality"].apply(quality_to_class)
    return df


def feature_stats(df: pd.DataFrame) -> dict:
    """Slider ranges and typical values for the dashboard."""
    stats = {}
    for col in FEATURES[:-1]:
        s = df[col]
        stats[col] = {
            "min": float(s.quantile(0.005)),
            "max": float(s.quantile(0.995)),
            "median_red": float(df.loc[df.is_red == 1, col].median()),
            "median_white": float(df.loc[df.is_red == 0, col].median()),
        }
    return stats


def main() -> None:
    start = time.time()
    df = load_data()
    print(f"Loaded {len(df)} wines ({int(df.is_red.sum())} red, {int((1 - df.is_red).sum())} white)")

    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        # The scaler lives inside the pipeline, so serving applies the exact
        # same preprocessing as training.
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        }
        fitted[name] = pipe
        print(f"{name:>16}: accuracy={results[name]['accuracy']:.4f} macro_f1={results[name]['macro_f1']:.4f}")

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best = fitted[best_name]
    print(f"Selected model: {best_name}")

    pred = best.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    report = classification_report(y_test, pred, labels=CLASSES, output_dict=True)

    perm = permutation_importance(
        best, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1
    )
    importance = sorted(
        ({"feature": f, "importance": round(float(v), 4)} for f, v in zip(FEATURES, perm.importances_mean)),
        key=lambda d: d["importance"],
        reverse=True,
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(time.time() - start, 1),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": CLASSES,
        "features": FEATURES,
        "class_distribution": {c: int((y == c).sum()) for c in CLASSES},
        "candidates": results,
        "selected_model": best_name,
        "accuracy": results[best_name]["accuracy"],
        "macro_f1": results[best_name]["macro_f1"],
        "per_class": {
            c: {k: round(float(report[c][k]), 4) for k in ("precision", "recall", "f1-score")} for c in CLASSES
        },
        "confusion_matrix": cm.tolist(),
        "feature_importance": importance,
        "feature_stats": feature_stats(df),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model and metrics to {MODEL_DIR} in {metrics['training_seconds']}s")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
