import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main
from api.model_store import LoadedModel

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65,
}


class FakePipeline:
    """Always predicts an 80% churn probability."""

    def predict_proba(self, X):
        p = np.full(len(X), 0.8)
        return np.column_stack([1 - p, p])


@pytest.fixture
def client(monkeypatch):
    def fake_load(settings):
        return LoadedModel(
            pipeline=FakePipeline(),
            version="7",
            stage="production",
            created_at="2026-11-16T00:00:00+00:00",
        )

    monkeypatch.setattr(main, "load_model", fake_load)
    with TestClient(main.app) as c:
        yield c


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "7"
    assert body["stage"] == "production"


def test_predict_returns_probability_label_and_version(client):
    r = client.post("/predict", json=VALID_CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_probability"] == pytest.approx(0.8)
    assert body["churn_label"] is True
    assert body["model_version"] == "7"


def test_invalid_category_returns_422_naming_the_field(client):
    r = client.post("/predict", json={**VALID_CUSTOMER, "Contract": "Weekly"})
    assert r.status_code == 422
    assert "Contract" in str(r.json())


def test_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_negative_charges_rejected(client):
    r = client.post("/predict", json={**VALID_CUSTOMER, "MonthlyCharges": -5})
    assert r.status_code == 422


def test_threshold_is_configurable(monkeypatch, client):
    # Restart the app with a higher threshold, so the 0.8 prediction is no longer a churn label.
    monkeypatch.setenv("CHURN_THRESHOLD", "0.9")
    with TestClient(main.app) as c:
        r = c.post("/predict", json=VALID_CUSTOMER)
    assert r.json()["churn_label"] is False