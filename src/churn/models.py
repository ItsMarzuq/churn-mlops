from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.baseline import fit_baseline, split_features
from churn.data import SEED

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = list(X.select_dtypes(exclude="number").columns)
    numeric = list(X.select_dtypes(include="number").columns)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def fit_xgboost(train: pd.DataFrame) -> Pipeline:
    X, y = split_features(train)
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(X)),
            ("model", XGBClassifier(**XGB_PARAMS, random_state=SEED, eval_metric="logloss")),
        ]
    )
    model.fit(X, y)
    return model

LOGREG = "logistic_regression"
XGBOOST = "xgboost"
MODEL_TYPE_TAG = "model_type"
FIT_FUNCTIONS = {LOGREG: fit_baseline, XGBOOST: fit_xgboost}