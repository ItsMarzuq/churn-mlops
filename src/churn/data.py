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
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
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


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=SEED,
    )
    return train, test


def save_split_indices(train: pd.DataFrame, test: pd.DataFrame, path: str | Path) -> None:
    payload = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "train_rows": [int(i) for i in train.index],
        "test_rows": [int(i) for i in test.index],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload))