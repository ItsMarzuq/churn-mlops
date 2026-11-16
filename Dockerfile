FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The API needs only these packages, not DVC, pandera or the notebook tools.
RUN pip install \
    pandas scikit-learn xgboost mlflow fastapi uvicorn pydantic pyyaml

COPY api ./api
COPY model ./model

ENV MODEL_PATH=/app/model

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]