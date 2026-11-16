from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import SEED, TARGET


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


def fit_baseline(train: pd.DataFrame) -> Pipeline:
    X, y = split_features(train)
    categorical = list(X.select_dtypes(exclude="number").columns)
    numeric = list(X.select_dtypes(include="number").columns)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    model.fit(X, y)
    return model


def top_k_recall(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.2) -> float:
    k = int(round(len(scores) * fraction))
    top_idx = np.argsort(-scores)[:k]
    total_churners = y_true.sum()
    if total_churners == 0:
        raise ValueError("No churners in y_true; recall is undefined")
    return float(y_true[top_idx].sum() / total_churners)


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    X, y = split_features(test)
    probs = model.predict_proba(X)[:, 1]
    y_arr = y.to_numpy()
    return {
        "pr_auc": float(average_precision_score(y_arr, probs)),
        "brier": float(brier_score_loss(y_arr, probs)),
        "recall_top_20pct": top_k_recall(y_arr, probs, 0.2),
    }