"""Tests for the prediction log. Run with: pytest"""

import sqlite3
from datetime import datetime, timedelta, timezone

from api.prediction_log import PredictionLog


def test_record_and_read_back(tmp_path):
    log = PredictionLog(tmp_path / "p.db")
    log.record("7", {"tenure": 2}, 0.8, True)

    rows = log.recent(days=1)
    assert len(rows) == 1
    assert rows[0]["model_version"] == "7"
    assert rows[0]["features"] == {"tenure": 2}
    assert rows[0]["churn_label"] == 1


def test_prune_removes_only_old_rows(tmp_path):
    path = tmp_path / "p.db"
    log = PredictionLog(path)
    log.record("7", {"tenure": 2}, 0.8, True)

    old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    with sqlite3.connect(path) as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(timestamp, model_version, churn_probability, churn_label, features) "
                "VALUES (?, '6', 0.1, 0, '{}')",
                (old,),
            )

    removed = log.prune(days=90)
    assert removed == 1
    assert len(log.recent(days=365)) == 1