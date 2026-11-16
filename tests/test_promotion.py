"""Tests for the promotion rule. Run with: pytest"""

import numpy as np

from churn.promotion import decide

FOLDS = 5


def test_clear_improvement_is_promoted():
    production_pr = np.full(FOLDS, 0.66)
    candidate_pr = production_pr + np.array([0.02, 0.021, 0.019, 0.02, 0.022])
    brier = np.full(FOLDS, 0.135)

    decision = decide(candidate_pr, production_pr, brier, brier)
    assert decision.promote


def test_tied_models_are_rejected():
    # The per-fold gains from our actual cross-validation run.
    production_pr = np.full(FOLDS, 0.66)
    candidate_pr = production_pr + np.array([0.0039, -0.004, 0.0057, -0.0084, 0.0063])
    production_brier = np.full(FOLDS, 0.1350)
    candidate_brier = np.full(FOLDS, 0.1360)

    decision = decide(candidate_pr, production_pr, candidate_brier, production_brier)
    assert not decision.promote
    assert "not larger than fold spread" in decision.reason


def test_brier_regression_is_rejected_even_with_better_pr_auc():
    production_pr = np.full(FOLDS, 0.66)
    candidate_pr = production_pr + 0.03
    production_brier = np.full(FOLDS, 0.135)
    candidate_brier = np.full(FOLDS, 0.150)  # 0.015 worse, above the 0.01 limit

    decision = decide(candidate_pr, production_pr, candidate_brier, production_brier)
    assert not decision.promote
    assert "Brier" in decision.reason