import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-classifier"
OUT = Path("model")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "production")

    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")

    if OUT.exists():
        shutil.rmtree(OUT)
    mlflow.sklearn.save_model(pipeline, path=str(OUT), serialization_format="cloudpickle")

    created = datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc)
    metadata = {"name": MODEL_NAME, "version": str(mv.version), "created_at": created.isoformat()}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"exported {MODEL_NAME} version {mv.version} to {OUT}/")


if __name__ == "__main__":
    main()