"""Promote a candidate model version if it beats production under the CV rule.

Run from the project root:
    python scripts/promote.py --version 2

Exit status: 0 if promoted (or already production), 1 if rejected.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churn.cv import cross_validate
from churn.data import clean, load_raw, split
from churn.models import FIT_FUNCTIONS, MODEL_TYPE_TAG
from churn.promotion import decide
from churn.schema import validate_raw

RAW = Path("data/raw/telco.csv")
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-classifier"
PRODUCTION_ALIAS = "production"


def model_type_of(client: MlflowClient, version: str) -> str:
    mv = client.get_model_version(MODEL_NAME, version)
    if MODEL_TYPE_TAG not in mv.tags:
        raise SystemExit(
            f"Version {version} has no '{MODEL_TYPE_TAG}' tag. "
            "Delete the registered model and rerun run_candidates.py."
        )
    return mv.tags[MODEL_TYPE_TAG]


def production_version(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except MlflowException:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a candidate if it passes the CV rule.")
    parser.add_argument("--version", required=True, type=int)
    args = parser.parse_args()
    version = str(args.version)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    prod = production_version(client)
    if prod is not None and prod.version == version:
        print(f"Version {version} is already production. Nothing to do.")
        sys.exit(0)

    cand_type = model_type_of(client, version)
    train, _ = split(clean(validate_raw(load_raw(RAW))))
    cand_pr, cand_brier = cross_validate(FIT_FUNCTIONS[cand_type], train)

    if prod is None:
        promote = True
        reason = "No production model exists yet; first candidate promoted."
    else:
        prod_type = model_type_of(client, prod.version)
        prod_pr, prod_brier = cross_validate(FIT_FUNCTIONS[prod_type], train)
        print(f"candidate v{version} ({cand_type}) mean PR-AUC: {cand_pr.mean():.4f}")
        print(f"production v{prod.version} ({prod_type}) mean PR-AUC: {prod_pr.mean():.4f}")
        decision = decide(cand_pr, prod_pr, cand_brier, prod_brier)
        promote, reason = decision.promote, decision.reason

    status = "promoted" if promote else "rejected"
    timestamp = datetime.now(timezone.utc).isoformat()
    client.set_model_version_tag(MODEL_NAME, version, "promotion_status", status)
    client.set_model_version_tag(MODEL_NAME, version, "promotion_reason", reason)
    client.set_model_version_tag(MODEL_NAME, version, "promotion_timestamp", timestamp)
    print(f"version {version} ({cand_type}): {status}\n{reason}")

    if promote:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()