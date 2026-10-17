# House Price Prediction (Kaggle: House Prices – Advanced Regression Techniques)

Predicting the sale prices of houses in Ames, Iowa, from 79 descriptive features: an end-to-end regression project, from raw CSV files to a Kaggle submission.

| Measure | Random Forest baseline | Final model |
|---|---|---|
| 5-fold CV RMSLE | 0.1311 | **0.1054** |
| Holdout RMSLE (20%, used once) | 0.1451 | **0.1273** |
| Kaggle public leaderboard | | **0.1224**, top 16% of leak-free submissions (681 / 4,209) |

**Final model:** an equal-weight blend (in log space) of Ridge regression and Gradient Boosting, trained on log prices, with unusual sales (foreclosures, family sales) at half weight.

## Read the analysis

- [`house_price_analysis.ipynb`](house_price_analysis.ipynb): the full notebook with code
- [`house_price_analysis.pdf`](house_price_analysis.pdf): the same analysis as a 24-page report without code

## Highlights

- **Leakage-free pipeline.** Cleaning rules read only each house's own values; everything learned from data (medians, category lists, scaling) is fitted inside a scikit-learn pipeline on training folds only. A 20% holdout stayed untouched until the final check.
- **Missing values by meaning.** "Absent" (no pool), "unknown" (not recorded), and "not applicable" (garage year for a house without a garage) get different treatments, with absence confirmed by related measurements.
- **One decision rule.** Every choice is made by cross-validated RMSLE on the same five folds; candidates within 0.0002 of the best are a tie, and a tie goes to the simpler model.
- **Model comparison.** Random Forest, Lasso / Ridge / ElasticNet, Gradient Boosting, XGBoost, a two-stage "classify price tier, then regress" model, and fixed and learned blends. Regularised linear and boosting models beat the Random Forest by 14–16%; Lasso matches the best linear accuracy with 105 of 256 encoded features.
- **Error analysis.** About a third of the squared error comes from the 9% of houses sold under unusual conditions. Giving those sales half weight in training improved CV RMSLE in all five folds.
- **Bias vs. regression to the mean.** Grouping by predicted rather than actual price shows that most of the apparent under-prediction of expensive houses is a selection effect (−6.0% vs. −2.2%). A post-hoc calibration was tested and not adopted.

## Project structure

| Path | Content |
|---|---|
| `house_price_analysis.ipynb` | Full analysis: cleaning, features, model comparison, error analysis, final model |
| `house_price_analysis.pdf` | Report version of the notebook (no code) |
| `submission.csv` | Predictions for the Kaggle test set |
| `requirements.txt` | Python packages and versions used |

## How to run

1. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) (a free Kaggle account is required) and put them in this folder.
2. Install the packages (tested with Python 3.14):
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook and run all cells (about 6 minutes on a laptop):
   ```bash
   jupyter lab house_price_analysis.ipynb
   ```
   The last cell writes `submission.csv`.

## Data

The data comes from the Kaggle competition and is not included in this repository. It is based on the Ames Housing dataset compiled by Dean De Cock (*Journal of Statistics Education* 19(3), 2011).
