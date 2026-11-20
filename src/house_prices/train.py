"""Train the final model on all labelled houses and save it.

    python -m house_prices.train                 # train and save models/final_model.joblib
    python -m house_prices.train --cv           # also report 5-fold CV RMSLE first
"""

import argparse
import json
import platform
from pathlib import Path
from importlib.metadata import version

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from .data import clean_house_data, is_outlier, load_house_data
from .features import make_features
from .metrics import rmsle_score
from .models import build_model, load_config, unusual_sale_weights
from .settings import CONFIG_DIR, MODELS_DIR, RANDOM_STATE, RAW_DATA_DIR


def cross_validate(model, X, y, sample_weight, n_splits=5):
    """RMSLE per fold, with the same weights used when fitting each fold."""
    scores = []
    for fit_rows, held_rows in KFold(n_splits, shuffle=True, random_state=RANDOM_STATE).split(X):
        fold_model = clone(model).fit(X.iloc[fit_rows], y.iloc[fit_rows], sample_weight=sample_weight[fit_rows])
        scores.append(rmsle_score(y.iloc[held_rows], fold_model.predict(X.iloc[held_rows])))
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_DIR / "final_model.toml", help="model config (TOML)")
    parser.add_argument("--train", default=RAW_DATA_DIR / "train.csv", help="Kaggle train.csv")
    parser.add_argument("--output", default=MODELS_DIR / "final_model.joblib", help="where to save the model")
    parser.add_argument("--cv", action="store_true", help="report 5-fold CV RMSLE before the final fit")
    args = parser.parse_args()

    config = load_config(args.config)
    labelled = clean_house_data(load_house_data(args.train))
    outliers = is_outlier(labelled)
    train = labelled.loc[~outliers].reset_index(drop=True)
    X, y = make_features(train), train["SalePrice"]
    sample_weight = unusual_sale_weights(train, config["unusual_sale_weight"])
    model = build_model(config)
    print(f"Training on {len(train)} houses ({int(outliers.sum())} outliers removed)")

    metadata = {
        "config": config,
        "training_rows": len(train),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
    }
    if args.cv:
        scores = cross_validate(model, X, y, sample_weight)
        print(f"5-fold CV RMSLE: {scores.mean():.4f} (std {scores.std():.4f})")
        metadata["cv_rmsle"] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4),
                                "folds": [round(s, 4) for s in scores]}

    model.fit(X, y, sample_weight=sample_weight)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output} and {args.output.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
