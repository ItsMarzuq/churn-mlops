from pathlib import Path

import pytest
from pandera.errors import SchemaErrors

from churn.data import load_raw
from churn.schema import validate_raw

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"


def test_raw_data_passes_schema():
    validate_raw(load_raw(RAW))  


def test_unknown_contract_type_fails():
    df = load_raw(RAW)
    df.loc[0, "Contract"] = "Weekly"
    with pytest.raises(SchemaErrors):
        validate_raw(df)


def test_missing_column_fails():
    df = load_raw(RAW).drop(columns=["tenure"])
    with pytest.raises(SchemaErrors):
        validate_raw(df)