"""Train a small model on the committed sample so CI can build and run the Docker image.

The real model comes from the MLflow registry (scripts/export_model.py). CI has no
registry, so it uses this throwaway model instead.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow.sklearn

from churn.baseline import fit_baseline
from churn.data import clean, load_raw

SAMPLE = Path("tests/data/telco_sample.csv")
OUT = Path("model")


def main() -> None:
    model = fit_baseline(clean(load_raw(SAMPLE)))

    if OUT.exists():
        shutil.rmtree(OUT)
    mlflow.sklearn.save_model(model, path=str(OUT), serialization_format="cloudpickle")

    metadata = {
        "name": "ci-model",
        "version": "ci",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote CI model to {OUT}/")


if __name__ == "__main__":
    main()