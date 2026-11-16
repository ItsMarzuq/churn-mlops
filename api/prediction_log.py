"""Log every prediction to a local SQLite table (FR-A7).

Stored: timestamp, model version, input features, probability, and label.
Rows older than the retention period are deleted by prune().
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

RETENTION_DAYS = 90


class PredictionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    churn_probability REAL NOT NULL,
                    churn_label INTEGER NOT NULL,
                    features TEXT NOT NULL
                )
                """
            )

    def record(self, model_version: str, features: dict, probability: float, label: bool) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(timestamp, model_version, churn_probability, churn_label, features) "
                "VALUES (?, ?, ?, ?, ?)",
                (now, model_version, probability, int(label), json.dumps(features)),
            )

    def prune(self, days: int = RETENTION_DAYS) -> int:
        """Delete rows older than `days`. Returns how many were removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff,))
            return cursor.rowcount

    def recent(self, days: int = 7) -> list[dict]:
        """Rows from the last `days`, used later by the drift report."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM predictions WHERE timestamp >= ? ORDER BY timestamp", (cutoff,)
            ).fetchall()
        return [dict(row) | {"features": json.loads(row["features"])} for row in rows]