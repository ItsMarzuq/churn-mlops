from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from churn.baseline import fit_baseline, split_features
from churn.data import SEED, clean, load_raw, split
from churn.models import fit_xgboost
from churn.schema import validate_raw

RAW = Path("data/raw/telco.csv")
N_FOLDS = 5


def cross_validate(fit_fn, train):
    X, y = split_features(train)
    y = y.to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pr_aucs, briers = [], []
    for tr_idx, va_idx in skf.split(X, y):
        model = fit_fn(train.iloc[tr_idx])
        probs = model.predict_proba(X.iloc[va_idx])[:, 1]
        pr_aucs.append(average_precision_score(y[va_idx], probs))
        briers.append(brier_score_loss(y[va_idx], probs))
    return np.array(pr_aucs), np.array(briers)


def main() -> None:
    raw = validate_raw(load_raw(RAW))
    train, _ = split(clean(raw))
    lr_pr, lr_brier = cross_validate(fit_baseline, train)
    xgb_pr, xgb_brier = cross_validate(fit_xgboost, train)

    for name, pr, brier in [
        ("logistic regression", lr_pr, lr_brier),
        ("xgboost", xgb_pr, xgb_brier),
    ]:
        print(f"\n{name}")
        print(f"  PR-AUC mean {pr.mean():.4f}  std {pr.std(ddof=1):.4f}")
        print(f"  Brier  mean {brier.mean():.4f}  std {brier.std(ddof=1):.4f}")

    diff = xgb_pr - lr_pr
    print(f"\nPR-AUC gain of xgboost over logistic regression, per fold: {np.round(diff, 4)}")
    print(
        f"  mean {diff.mean():.4f}  std {diff.std(ddof=1):.4f}  "
        f"xgboost wins {(diff > 0).sum()} of {N_FOLDS} folds"
    )


if __name__ == "__main__":
    main()