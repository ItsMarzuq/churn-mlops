import json
from pathlib import Path

import mlflow

from churn.baseline import evaluate, fit_baseline
from churn.data import (
    SEED,
    TARGET,
    TEST_SIZE,
    clean,
    load_raw,
    save_split_indices,
    split,
)
from churn.schema import validate_raw

RAW = Path("data/raw/telco.csv")
SPLIT_FILE = Path("data/processed/split.json")
METRICS_FILE = Path("metrics/baseline.json")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "churn-classifier"


def main() -> None:
    raw = validate_raw(load_raw(RAW))
    cleaned = clean(raw)
    train, test = split(cleaned)
    save_split_indices(train, test, SPLIT_FILE)

    model = fit_baseline(train)
    metrics = evaluate(model, test)

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="logreg-baseline"):
        mlflow.log_params({
            "model": "logistic_regression",
            "seed": SEED,
            "test_size": TEST_SIZE,
            "train_rows": len(train),
            "test_rows": len(test),
        })
        mlflow.log_metrics(metrics)
        features = list(train.drop(columns=[TARGET]).columns)
        mlflow.log_text("\n".join(features), "features.txt")
        mlflow.set_tag("data_source", "data/raw/telco.csv")

    print(f"train rows: {len(train)}  test rows: {len(test)}")
    print(f"test churn rate: {test[TARGET].mean():.3f}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()