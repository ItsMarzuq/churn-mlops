import json
from pathlib import Path

from churn.baseline import evaluate, fit_baseline
from churn.data import clean, load_raw, save_split_indices, split
from churn.schema import validate_raw

RAW = Path("data/raw/telco.csv")
SPLIT_FILE = Path("data/processed/split.json")
METRICS_FILE = Path("metrics/baseline.json")


def main() -> None:
    raw = validate_raw(load_raw(RAW))
    cleaned = clean(raw)
    train, test = split(cleaned)
    save_split_indices(train, test, SPLIT_FILE)

    model = fit_baseline(train)
    metrics = evaluate(model, test)

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    print(f"train rows: {len(train)}  test rows: {len(test)}")
    print(f"test churn rate: {test['Churn'].mean():.3f}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()