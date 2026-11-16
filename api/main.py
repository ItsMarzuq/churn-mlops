from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.config import load_settings
from api.model_store import load_model
from api.schemas import (
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load settings and the model once at startup, not on every request."""
    settings = load_settings()
    state["settings"] = settings
    state["model"] = load_model(settings)
    yield
    state.clear()


app = FastAPI(title="Churn prediction API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="model" in state)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        name=state["settings"].model_name,
        version=model.version,
        stage=model.stage,
        created_at=model.created_at,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([customer.model_dump()])
    probability = float(model.pipeline.predict_proba(row)[0, 1])
    threshold = state["settings"].threshold
    return PredictionResponse(
        churn_probability=probability,
        churn_label=probability >= threshold,
        model_version=model.version,
    )