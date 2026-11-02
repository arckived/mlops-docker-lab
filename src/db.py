"""Prediction log storage.

Uses Postgres when DATABASE_URL is set (the docker compose setup) and falls
back to a local SQLite file otherwise (plain `docker run` or local dev).
"""

import json
import os
import time
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
)
from sqlalchemy.exc import OperationalError

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predictions.db")

metadata = MetaData()
predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("wine_type", String(5), nullable=False),
    Column("features", Text, nullable=False),
    Column("predicted_class", String(10), nullable=False),
    Column("confidence", Float, nullable=False),
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def init_db(retries: int = 15, delay: float = 2.0) -> None:
    """Create the table, waiting for the database to accept connections."""
    for attempt in range(1, retries + 1):
        try:
            metadata.create_all(engine)
            # Drop pooled connections so forked gunicorn workers open their own.
            engine.dispose()
            return
        except OperationalError:
            if attempt == retries:
                raise
            print(f"Database not ready (attempt {attempt}/{retries}), retrying in {delay}s")
            time.sleep(delay)


def log_prediction(wine_type: str, features: dict, predicted_class: str, confidence: float) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            insert(predictions).values(
                created_at=datetime.now(timezone.utc),
                wine_type=wine_type,
                features=json.dumps(features),
                predicted_class=predicted_class,
                confidence=confidence,
            )
        )
        return int(result.inserted_primary_key[0])


def recent_predictions(limit: int = 50) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(select(predictions).order_by(predictions.c.id.desc()).limit(limit)).mappings().all()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "created_at": r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
                "wine_type": r["wine_type"],
                "features": json.loads(r["features"]),
                "predicted_class": r["predicted_class"],
                "confidence": round(r["confidence"], 4),
            }
        )
    return out


def class_counts() -> dict:
    with engine.connect() as conn:
        rows = conn.execute(
            select(predictions.c.predicted_class, func.count()).group_by(predictions.c.predicted_class)
        ).all()
    return {cls: int(n) for cls, n in rows}
