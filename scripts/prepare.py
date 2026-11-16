"""DVC stage 1: validate raw data, clean it, split it, and save the results."""

from pathlib import Path

import yaml

from churn.data import clean, load_raw, save_split_indices, split
from churn.schema import validate_raw

RAW = Path("data/raw/telco.csv")
OUT = Path("data/processed")


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())

    raw = validate_raw(load_raw(RAW))
    cleaned = clean(raw)
    train, test = split(cleaned, test_size=params["test_size"])

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv")
    test.to_csv(OUT / "test.csv")
    save_split_indices(train, test, OUT / "split.json")
    print(f"train rows: {len(train)}  test rows: {len(test)}")


if __name__ == "__main__":
    main()