"""Tests for the drift tests. Run with: pytest"""

import numpy as np
import pandas as pd

from churn.drift import column_drifted, drift_share


def test_same_distribution_rarely_flagged():
    # The test uses a 5% significance level, so about 1 in 20 same-distribution
    # checks is flagged by chance. Check the rate over many seeds, not one draw.
    flagged = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        reference = pd.Series(rng.normal(50, 10, 1000))
        current = pd.Series(rng.normal(50, 10, 500))
        flagged += column_drifted(reference, current)
    assert flagged <= 2


def test_shifted_numeric_flagged():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(50, 10, 1000))
    current = pd.Series(rng.normal(80, 10, 500))
    assert column_drifted(reference, current)


def test_shifted_category_flagged():
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 100 + ["b"] * 400)
    assert column_drifted(reference, current)


def test_share_is_fraction_of_drifted_features():
    flags = {"a": True, "b": False, "c": False, "d": False}
    assert drift_share(flags) == 0.25