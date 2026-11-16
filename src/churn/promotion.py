"""Promotion rule: decide whether a candidate should replace the production model.

Rule chosen for this project:
- Candidate and production are scored on the same cross-validation folds.
- Promote only if the average PR-AUC gain is larger than the fold-to-fold spread of that gain.
- The Brier score may be at most 0.01 worse (PRD FR-P2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_BRIER_REGRESSION = 0.01


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


def decide(
    candidate_pr: np.ndarray,
    production_pr: np.ndarray,
    candidate_brier: np.ndarray,
    production_brier: np.ndarray,
) -> Decision:
    """Compare two models scored on the same folds and return a promote/reject decision."""
    gains = candidate_pr - production_pr
    mean_gain = float(gains.mean())
    spread = float(gains.std(ddof=1))
    brier_change = float(candidate_brier.mean() - production_brier.mean())  # positive means worse

    if brier_change > MAX_BRIER_REGRESSION:
        return Decision(
            False,
            f"Brier score worse by {brier_change:.4f}; limit is {MAX_BRIER_REGRESSION}",
        )
    if mean_gain <= spread:
        return Decision(
            False,
            f"PR-AUC gain {mean_gain:.4f} is not larger than fold spread {spread:.4f}",
        )
    return Decision(
        True,
        f"PR-AUC gain {mean_gain:.4f} exceeds fold spread {spread:.4f}; "
        f"Brier change {brier_change:+.4f}",
    )