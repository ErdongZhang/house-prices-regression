"""Predict sale prices for Kaggle's test.csv with a saved model.

    python -m house_prices.predict               # writes submissions/submission.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import clean_house_data, load_house_data
from .features import make_features
from .settings import MODELS_DIR, RAW_DATA_DIR, SUBMISSIONS_DIR


def make_submission(model, test_raw):
    """Kaggle submission: columns Id and SalePrice, one row per house in test.csv order."""
    test_clean = clean_house_data(test_raw)
    submission = pd.DataFrame({"Id": test_clean["Id"], "SalePrice": model.predict(make_features(test_clean))})

    assert list(submission.columns) == ["Id", "SalePrice"]
    assert submission["Id"].equals(test_raw["Id"]) and submission["Id"].is_unique
    assert submission["SalePrice"].notna().all() and (submission["SalePrice"] > 0).all()
    return submission


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODELS_DIR / "final_model.joblib", help="model saved by train.py")
    parser.add_argument("--test", default=RAW_DATA_DIR / "test.csv", help="Kaggle test.csv")
    parser.add_argument("--output", default=SUBMISSIONS_DIR / "submission.csv", help="submission file to write")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"{args.model} not found: run `python -m house_prices.train` first")
    submission = make_submission(joblib.load(args.model), load_house_data(args.test))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {args.output}: {len(submission)} rows, "
          f"median predicted price ${submission['SalePrice'].median():,.0f}")


if __name__ == "__main__":
    main()
