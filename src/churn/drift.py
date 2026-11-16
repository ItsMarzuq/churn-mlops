"""Per-feature drift tests. The pass/fail decision does not depend on Evidently."""

from __future__ import annotations

import pandas as pd
from scipy import stats

P_VALUE = 0.05
MAX_DRIFT_SHARE = 0.30


def column_drifted(reference: pd.Series, current: pd.Series) -> bool:
    """Numeric columns use a two-sample Kolmogorov-Smirnov test; text columns use chi-square."""
    if pd.api.types.is_numeric_dtype(reference):
        _, p = stats.ks_2samp(reference.dropna(), current.dropna())
        return bool(p < P_VALUE)

    ref_text = reference.astype(str)
    cur_text = current.astype(str)
    categories = sorted(set(ref_text) | set(cur_text))
    if len(categories) < 2:
        return False

    ref_counts = ref_text.value_counts().reindex(categories, fill_value=0)
    cur_counts = cur_text.value_counts().reindex(categories, fill_value=0)
    _, p, _, _ = stats.chi2_contingency([ref_counts.values, cur_counts.values])
    return bool(p < P_VALUE)


def drift_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, bool]:
    """Return a drifted/not-drifted flag for every feature column."""
    return {col: column_drifted(reference[col], current[col]) for col in reference.columns}


def drift_share(flags: dict[str, bool]) -> float:
    """Fraction of features that drifted."""
    return sum(flags.values()) / len(flags) 