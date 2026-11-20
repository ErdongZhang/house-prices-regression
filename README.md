# House Price Prediction (Kaggle: House Prices – Advanced Regression Techniques)

[![tests](https://github.com/ErdongZhang/house-prices-regression/actions/workflows/tests.yml/badge.svg)](https://github.com/ErdongZhang/house-prices-regression/actions/workflows/tests.yml)

Predicting the sale prices of houses in Ames, Iowa, from 79 descriptive features: an end-to-end regression project, from raw CSV files to a Kaggle submission.

| Measure | Random Forest baseline | Final model |
|---|---|---|
| 5-fold CV RMSLE | 0.1311 | **0.1054** |
| Holdout RMSLE (20%, used once) | 0.1451 | **0.1273** |
| Kaggle public leaderboard | | **0.1224**, top 16% of leak-free submissions (681 / 4,209)\* |

\* The leaderboard's top 59 scores (rank 56: 0.032) are far below what any model reaches on this data and come from leaked test prices, so they are excluded. Including them, the rank is 740 of 4,268. The best score that looks leak-free is 0.1067.

**Final model:** an equal-weight blend (in log space) of Ridge regression and Gradient Boosting, trained on log prices, with unusual sales (foreclosures, family sales) at half weight.

![Predicted vs. actual price, percentage error, and error distribution for the final model](reports/figures/predicted_vs_actual.png)

*Final model on houses it never saw during fitting: out-of-fold predictions on the training part (blue) and the 20% holdout (orange). The median error is about 5%; the largest misses are over-predicted cheap houses, mostly unusual sales.*

## Quick start: reproduce the submission

Requires Python 3.11 or newer (results were produced with 3.14).

```bash
# 1. Install the exact package versions used, and this project's package
pip install -r requirements.txt
pip install -e .

# 2. Download the data into data/raw/ (see data/README.md)

# 3. Train the final model and write the submission
python -m house_prices.train          # ~20 s; saves models/final_model.joblib
python -m house_prices.predict        # writes submissions/submission.csv
```

`python -m house_prices.train --cv` also reports a 5-fold CV score before training. Both scripts take `--help`.

`requirements.txt` pins the exact versions behind the reported results. To install only what training and prediction need, with any recent versions, use `pip install -e .`; add `pip install -e ".[notebook]"` for the notebook and `".[dev]"` for the tests.

The model is defined in [`configs/final_model.toml`](configs/final_model.toml): blend members, hyperparameters, blend weights, and the weight of unusual sales. Every value there was chosen in the notebook, and the notebook's last code cell checks that the config still matches its selection.

## How the project is organised

```
├── configs/
│   └── final_model.toml          # the final model's settings, chosen in the notebook
├── data/
│   ├── README.md                 # how to download the data
│   └── raw/                      # Kaggle CSV files (not in git)
├── models/
│   ├── final_model.joblib        # trained model (not in git; rebuilt by train)
│   └── final_model.json          # config, package versions, and CV score of that model
├── notebooks/
│   └── house_price_analysis.ipynb   # the full analysis: why every choice was made
├── reports/
│   ├── house_price_analysis.pdf  # the notebook as a 24-page report without code
│   └── figures/                  # figures used in this README
├── src/house_prices/             # code shared by the notebook and the scripts
│   ├── data.py                   # reading and rule-based cleaning
│   ├── features.py               # feature encoding, imputers, preprocessing pipeline
│   ├── models.py                 # blend model and building it from the config
│   ├── metrics.py                # RMSLE, the competition metric
│   ├── settings.py               # project folders and the random seed
│   ├── train.py                  # command line: train and save the final model
│   └── predict.py                # command line: write the Kaggle submission
├── submissions/
│   └── submission.csv            # predictions submitted to Kaggle
├── tests/                        # pytest checks of cleaning rules and the submission format
├── .github/workflows/tests.yml   # runs the tests on every push
├── pyproject.toml                # makes src/house_prices installable
└── requirements.txt              # exact package versions used
```

**Notebook vs. package.** The notebook tells the story: it compares seven candidate models, analyses errors, and justifies each decision. Code the final model depends on lives in `src/house_prices` and is imported by the notebook, so the analysis and the training script cannot drift apart. Code that only serves the analysis (model comparison, plots, error tables) stays in the notebook.

## Read the analysis

- [`notebooks/house_price_analysis.ipynb`](notebooks/house_price_analysis.ipynb): the full analysis with code. To re-run it (about 6 minutes), install as in the quick start and run `jupyter lab notebooks/house_price_analysis.ipynb`.
- [`reports/house_price_analysis.pdf`](reports/house_price_analysis.pdf): the same analysis as a report without code.

## Highlights

- **Leakage-free pipeline.** Cleaning rules read only each house's own values; everything learned from data (medians, category lists, scaling) is fitted inside a scikit-learn pipeline on training folds only. A 20% holdout stayed untouched until the final check.
- **Missing values by meaning.** "Absent" (no pool), "unknown" (not recorded), and "not applicable" (garage year for a house without a garage) get different treatments, with absence confirmed by related measurements.
- **One decision rule.** Every choice is made by cross-validated RMSLE on the same five folds; candidates within 0.0002 of the best are a tie, and a tie goes to the simpler model.
- **Model comparison.** Random Forest, Lasso / Ridge / ElasticNet, Gradient Boosting, XGBoost, a two-stage "classify price tier, then regress" model, and fixed and learned blends. Regularised linear and boosting models beat the Random Forest by 14–16%; Lasso matches the best linear accuracy with 105 of 256 encoded features.
- **Error analysis.** About a third of the squared error comes from the 9% of houses sold under unusual conditions. Giving those sales half weight in training improved CV RMSLE in all five folds.
- **Bias vs. regression to the mean.** Grouping by predicted rather than actual price shows that most of the apparent under-prediction of expensive houses is a selection effect (−6.0% vs. −2.2%). A post-hoc calibration was tested and not adopted.

![Cross-validated RMSLE of every candidate model](reports/figures/cv_comparison.png)

*Cross-validated RMSLE of every candidate, before the sample-weighting step (which lowers the selected blend from 0.1068 to 0.1054). Blends of a linear and a boosting model lead; their differences are a tenth of the fold standard deviation, so the simplest one is selected.*

## What didn't work

Ideas that were tested and not adopted, and why:

| Idea | Result | Decision |
|---|---|---|
| Two-stage model: classify the price tier, then regress within the tier | CV 0.1313 (soft) and 0.1453 (hard), against 0.1311 for the plain Random Forest. The classifier recognised only 56% of low-price and 65% of high-price houses | Dropped: misrouted houses are predicted by a model that never saw houses like them |
| XGBoost instead of scikit-learn's Gradient Boosting | CV 0.1132 against 0.1116 | Not used: no gain on about 1,000 houses |
| Adding the Random Forest to a blend | CV 0.1094 against 0.1068 without it | Dropped: learned blend weights give it 0–3% anyway |
| Calibrating the top quality grades upwards | CV 0.1068 → 0.1065, better in only 3 of 5 folds | Not adopted: inconsistent gain, based on fewer than 50 houses |
| Ignoring unusual sales entirely (weight 0) | CV 0.1073, worse than full weight (0.1068) | Half weight used instead (0.1054): noisy prices still carry information |
| Engineered features (total area, house age, total bathrooms, quality × area) | CV change of at most 0.0007 for the selected blend | Not added: below the noise between folds |
| Replacing Ridge with a tuned SVR, plus the engineered features *(tested separately, not in the notebook)* | CV 0.1054 → 0.1031, holdout 0.1273 → 0.1265, Kaggle 0.1224 → 0.1220 | Not adopted: a more complex model for a leaderboard gain of 0.0004, within the leaderboard's noise |

## Reproducibility notes

- **Same predictions.** `python -m house_prices.train` followed by `predict` reproduces the notebook's `submission.csv` to within floating-point rounding (largest difference below $0.000001). Random seeds are fixed (`random_state=42`).
- **Why `train --cv` prints 0.1089, not 0.1054.** The notebook's 0.1054 is measured on the 80% training part with the notebook's folds, the setting in which all models were compared. The script cross-validates on all 1,458 labelled houses with different folds, so its number differs; it is a sanity check, not the reported result.
- **Tests.** `pytest` checks the cleaning rules and that a small version of the model writes a valid submission (about 10 seconds). Tests that need the Kaggle data are skipped when `data/raw/` is empty, as on GitHub Actions.

## Data

The data comes from the Kaggle competition and is not included in this repository; see [`data/README.md`](data/README.md). It is based on the Ames Housing dataset compiled by Dean De Cock (*Journal of Statistics Education* 19(3), 2011).
