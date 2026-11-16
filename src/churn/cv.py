"""Stratified k-fold cross-validation on the training split."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from churn.baseline import split_features
from churn.data import SEED

N_FOLDS = 5


def cross_validate(
    fit_fn: Callable[[pd.DataFrame], object],
    train: pd.DataFrame,
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PR-AUC and Brier score for each fold.

    Both models get the same folds because the seed is fixed.
    """
    X, y = split_features(train)
    y_arr = y.to_numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    pr_aucs, briers = [], []
    for tr_idx, va_idx in skf.split(X, y_arr):
        model = fit_fn(train.iloc[tr_idx])
        probs = model.predict_proba(X.iloc[va_idx])[:, 1]
        pr_aucs.append(average_precision_score(y_arr[va_idx], probs))
        briers.append(brier_score_loss(y_arr[va_idx], probs))
    return np.array(pr_aucs), np.array(briers)