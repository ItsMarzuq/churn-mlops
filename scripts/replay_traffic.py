"""Send held-out test customers through the running API so the drift report has data.

Start the API first (uvicorn api.main:app), then from the project root run:
    python scripts/replay_traffic.py            # normal traffic
    python scripts/replay_traffic.py --drift    # shifted MonthlyCharges, to test the alert
"""

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

TEST = Path("data/processed/test.csv")
URL = "http://127.0.0.1:8000/predict"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(TEST, index_col=0).drop(columns=["Churn"])
    df = df.sample(n=min(args.n, len(df)), random_state=0)
    if args.drift:
        df["MonthlyCharges"] = df["MonthlyCharges"] * 1.5
        df["TotalCharges"] = df["TotalCharges"] * 1.5
        df["tenure"] = (df["tenure"] + 24).clip(upper=72)
        df["Contract"] = "Month-to-month"
        df["InternetService"] = "Fiber optic"
        df["PaymentMethod"] = "Electronic check"

    sent, failed = 0, 0
    for record in df.to_dict(orient="records"):
        payload = json.dumps(record, default=lambda v: v.item()).encode()
        request = urllib.request.Request(
            URL, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
            sent += 1
        except Exception:
            failed += 1

    print(f"sent {sent} requests, {failed} failed")


if __name__ == "__main__":
    main()