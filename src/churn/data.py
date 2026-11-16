"""Load, clean, and split the Telco churn dataset.

Pipeline order: load_raw -> validate -> clean -> split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Churn"
ID_COL = "customerID"
TEST_SIZE = 0.2
SEED = 42


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV exactly as it is. No cleaning happens here."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    - TotalCharges is text in the CSV; blanks become NaN, then zero-tenure
      customers (never billed yet) get 0.0.
    - Churn becomes 1/0.
    - The customer ID is dropped so it cannot be used as a feature.
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].str.strip(), errors="coerce")
    never_billed = out["TotalCharges"].isna() & (out["tenure"] == 0)
    out.loc[never_billed, "TotalCharges"] = 0.0

    unexpected_nans = out["TotalCharges"].isna().sum()
    if unexpected_nans:
        raise ValueError(f"{unexpected_nans} TotalCharges values are missing and not explained by zero tenure")

    out[TARGET] = (out[TARGET] == "Yes").astype(int)
    out = out.drop(columns=[ID_COL])
    return out


def split(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split with a fixed seed.

    Stratifying keeps the churn rate the same in both halves.
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET],
        random_state=SEED,
    )
    return train, test


def save_split_indices(train: pd.DataFrame, test: pd.DataFrame, path: str | Path) -> None:
    """Record which rows went to each side, so the split can be reproduced and audited."""
    payload = {
        "seed": SEED,
        "test_fraction": round(len(test) / (len(train) + len(test)), 4),
        "train_rows": [int(i) for i in train.index],
        "test_rows": [int(i) for i in test.index],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload))