"""Create a small committed sample of the Telco data for CI tests.

Keeps every row with a blank TotalCharges (zero tenure), plus random rows,
so the edge cases the tests check are always present. Run from the project root:
    python scripts/make_test_sample.py
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/telco.csv")
OUT = Path("tests/data/telco_sample.csv")
N_ROWS = 200
SEED = 42


def main() -> None:
    raw = pd.read_csv(RAW)
    blanks = raw[raw["TotalCharges"].str.strip() == ""]
    rest = raw.drop(blanks.index)
    sample = pd.concat([blanks, rest.sample(n=N_ROWS - len(blanks), random_state=SEED)])
    sample = sample.sort_index()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)
    print(f"wrote {len(sample)} rows ({len(blanks)} with blank TotalCharges) to {OUT}")


if __name__ == "__main__":
    main()