from pathlib import Path

import pandas as pd
import pytest

from churn.data import ID_COL, TARGET, clean, load_raw, split

RAW = Path(__file__).resolve().parent / "data" / "telco_sample.csv"


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    return clean(load_raw(RAW))


def test_row_count_is_unchanged(cleaned):
    assert len(cleaned) == 200


def test_customer_id_is_dropped(cleaned):
    assert ID_COL not in cleaned.columns


def test_target_is_binary(cleaned):
    assert set(cleaned[TARGET].unique()) == {0, 1}


def test_total_charges_is_numeric_with_no_missing_values(cleaned):
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_zero_tenure_customers_get_zero_total_charges(cleaned):
    zero_tenure = cleaned[cleaned["tenure"] == 0]
    assert len(zero_tenure) == 11
    assert (zero_tenure["TotalCharges"] == 0.0).all()


def test_split_is_stratified_and_deterministic(cleaned):
    train_a, test_a = split(cleaned)
    _train_b, test_b = split(cleaned)

    assert list(test_a.index) == list(test_b.index)
    assert abs(test_a[TARGET].mean() - cleaned[TARGET].mean()) < 0.03
    assert len(train_a) + len(test_a) == len(cleaned)
    assert set(train_a.index).isdisjoint(test_a.index)