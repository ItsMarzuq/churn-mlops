"""M6: drift check on the last 7 days of logged requests, compared with the training data.

Run from the project root:
    python scripts/drift_report.py

Exit status: 0 if the drift share is within the limit, 1 if it is too high
or there is not enough logged data.
"""

import os
import sys
from pathlib import Path

import pandas as pd

from churn.data import TARGET
from churn.drift import MAX_DRIFT_SHARE, drift_share, drift_summary
from churn.prediction_log import PredictionLog

REFERENCE = Path("data/processed/train.csv")
REPORT = Path("reports/drift_report.html")
WINDOW_DAYS = 7
MIN_ROWS = 30


def save_evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    snapshot.save_html(str(REPORT))
    print(f"saved {REPORT}")


def main() -> None:
    reference = pd.read_csv(REFERENCE, index_col=0).drop(columns=[TARGET])

    log = PredictionLog(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.db"))
    rows = log.recent(days=WINDOW_DAYS)
    rows = log.recent(days=WINDOW_DAYS)
    if len(rows) < MIN_ROWS:
        print(
            f"Only {len(rows)} logged predictions in the last {WINDOW_DAYS} days; "
            f"need {MIN_ROWS}."
        )
        sys.exit(1)

    current = pd.DataFrame([r["features"] for r in rows])[reference.columns]

    flags = drift_summary(reference, current)
    share = drift_share(flags)
    for col, drifted in flags.items():
        print(f"{col:20s} {'DRIFT' if drifted else 'ok'}")
    print(f"\ndrifted share: {share:.0%} (limit {MAX_DRIFT_SHARE:.0%})")

    save_evidently_report(reference, current)

    if share > MAX_DRIFT_SHARE:
        print("FAILED: too many features drifted. Consider retraining.")
        sys.exit(1)
    print("passed")


if __name__ == "__main__":
    main()