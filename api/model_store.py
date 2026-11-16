import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from api.config import Settings

PRODUCTION_ALIAS = "production"


@dataclass
class LoadedModel:
    pipeline: Any
    version: str
    stage: str
    created_at: str


def load_model(settings: Settings) -> LoadedModel:
    if settings.model_path:
        return _load_from_folder(settings.model_path)
    return _load_from_registry(settings)


def _load_from_folder(path: str) -> LoadedModel:
    """Used in the Docker image. The export script wrote metadata.json next to the model."""
    metadata = json.loads((Path(path) / "metadata.json").read_text())
    return LoadedModel(
        pipeline=mlflow.sklearn.load_model(path),
        version=metadata["version"],
        stage=PRODUCTION_ALIAS,
        created_at=metadata["created_at"],
    )


def _load_from_registry(settings: Settings) -> LoadedModel:
    """Used in local development, reading mlflow.db directly."""
    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient()

    if settings.model_version:
        version = str(settings.model_version)
        uri = f"models:/{settings.model_name}/{version}"
        stage = "pinned"
    else:
        version = str(client.get_model_version_by_alias(settings.model_name, PRODUCTION_ALIAS).version)
        uri = f"models:/{settings.model_name}@{PRODUCTION_ALIAS}"
        stage = PRODUCTION_ALIAS

    mv = client.get_model_version(settings.model_name, version)
    created = datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc)
    return LoadedModel(
        pipeline=mlflow.sklearn.load_model(uri),
        version=version,
        stage=stage,
        created_at=created.isoformat(),
    )