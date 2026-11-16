"""DVC stage 2: train both candidates on the saved split and write metrics."""

import json
from pathlib import Path

import pandas as pd
import yaml

from churn.baseline import evaluate, fit_baseline
from churn.models import fit_xgboost

PROC = Path("data/processed")
METRICS = Path("metrics/metrics.json")


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    train = pd.read_csv(PROC / "train.csv", index_col=0)
    test = pd.read_csv(PROC / "test.csv", index_col=0)

    results = {
        "logistic_regression": evaluate(fit_baseline(train), test),
        "xgboost": evaluate(fit_xgboost(train, params["xgboost"]), test),
    }

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(results, indent=2))

    for model, metrics in results.items():
        print(f"\n{model}")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()