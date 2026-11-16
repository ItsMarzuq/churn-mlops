import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    tracking_uri: str
    model_name: str
    model_version: str | None
    model_path: str | None
    threshold: float


def load_settings() -> Settings:
    return Settings(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        model_name=os.getenv("MODEL_NAME", "churn-classifier"),
        # If MODEL_VERSION is set, serve that version. Otherwise serve the production alias.
        model_version=os.getenv("MODEL_VERSION") or None,
        # If MODEL_PATH is set, load the model baked into the image instead of the registry.
        model_path=os.getenv("MODEL_PATH") or None,
        threshold=float(os.getenv("CHURN_THRESHOLD", "0.5")),
    )