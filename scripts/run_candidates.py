"""M2: train both candidates, log each run to MLflow, and register the models.

Run from the project root:
    python scripts/run_candidates.py
"""

from pathlib import Path

import mlflow
from mlflow import MlflowClient

from churn.baseline import evaluate, fit_baseline
from churn.data import SEED, TEST_SIZE, clean, load_raw, split
from churn.models import LOGREG, MODEL_TYPE_TAG, XGBOOST, XGB_PARAMS, fit_xgboost
from churn.schema import validate_raw

RAW = Path("data/raw/telco.csv")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "churn-classifier"
REGISTERED_NAME = "churn-classifier"

# Types we created ourselves during training, so safe to load back.
XGB_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def run_and_log(run_name, model, model_type, train, test, params, trusted_types=None):
    """Evaluate one model, log it to MLflow, and tag its registered version with its type."""
    metrics = evaluate(model, test)
    extra = {"skops_trusted_types": trusted_types} if trusted_types else {}
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "seed": SEED,
                "test_size": TEST_SIZE,
                "train_rows": len(train),
                "test_rows": len(test),
                **params,
            }
        )
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=REGISTERED_NAME, **extra
        )
    MlflowClient().set_model_version_tag(
        REGISTERED_NAME, str(info.registered_model_version), MODEL_TYPE_TAG, model_type
    )
    return metrics


def main() -> None:
    raw = validate_raw(load_raw(RAW))
    train, test = split(clean(raw))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = {
        "logreg-candidate": run_and_log(
            "logreg-candidate",
            fit_baseline(train),
            LOGREG,
            train,
            test,
            {"model": LOGREG},
        ),
        "xgboost-candidate": run_and_log(
            "xgboost-candidate",
            fit_xgboost(train),
            XGBOOST,
            train,
            test,
            {"model": XGBOOST, **XGB_PARAMS},
            trusted_types=XGB_TRUSTED_TYPES,
        ),
    }

    for run_name, metrics in results.items():
        print(f"\n{run_name}")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()