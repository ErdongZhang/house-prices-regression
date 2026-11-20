"""Feature encoding and the preprocessing pipeline (notebook sections 6-7)."""

import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .settings import RANDOM_STATE

QUALITY_SCALE = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ABSENCE_ZERO = {label: 0 for label in ["NoBasement", "NoGarage", "NoFireplace", "NoPool"]}
ORDINAL_MAPS = {
    **{column: {**QUALITY_SCALE, **ABSENCE_ZERO} for column in [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"
    ]},
    "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4, **ABSENCE_ZERO},
    "BsmtFinType1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, **ABSENCE_ZERO},
    "BsmtFinType2": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, **ABSENCE_ZERO},
    "GarageFinish": {"Unf": 1, "RFn": 2, "Fin": 3, **ABSENCE_ZERO},
}

# Identifiers, targets, and audit columns are not predictors
NON_FEATURE_COLS = ["Id", "SalePrice", "VeneerConflict"]

FRONTAGE_CATS = ["Neighborhood", "BldgType", "LotConfig", "LotShape", "MSSubClass"]


def make_features(df):
    """Select predictors and encode them with fixed rules."""
    X = df.drop(columns=NON_FEATURE_COLS, errors="ignore").copy()
    X["MSSubClass"] = X["MSSubClass"].astype(str)
    for column, mapping in ORDINAL_MAPS.items():
        X[column] = X[column].map(mapping).astype(float)
    return X


class LotFrontageImputer(BaseEstimator, TransformerMixin):
    """Fill LotFrontage using values learned from the training data only.

    method: "global", "neighborhood", or "model"
    add_indicator: add a 0/1 column marking originally missing values
    """

    def __init__(self, method="model", add_indicator=False):
        self.method = method
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        observed = X.loc[X["LotFrontage"].notna()]
        self.global_median_ = observed["LotFrontage"].median()

        if self.method == "neighborhood":
            self.by_neighborhood_ = observed.groupby("Neighborhood")["LotFrontage"].median()
        elif self.method == "model":
            self.model_ = Pipeline([
                ("encode", ColumnTransformer([
                    ("area", FunctionTransformer(np.log), ["LotArea"]),
                    ("cats", OneHotEncoder(handle_unknown="ignore"), FRONTAGE_CATS),
                ])),
                ("forest", RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
                )),
            ])
            self.model_.fit(observed[["LotArea"] + FRONTAGE_CATS], observed["LotFrontage"])
        return self

    def estimate(self, X):
        """Estimated frontage for every row of X."""
        if self.method == "neighborhood":
            return X["Neighborhood"].map(self.by_neighborhood_).fillna(self.global_median_).to_numpy()
        if self.method == "model":
            return self.model_.predict(X[["LotArea"] + FRONTAGE_CATS])
        return np.full(len(X), self.global_median_)

    def transform(self, X):
        result = X.copy()
        missing = result["LotFrontage"].isna()
        if self.add_indicator:
            result["LotFrontage_missing"] = missing.astype(int)
        if missing.any():
            result.loc[missing, "LotFrontage"] = self.estimate(result.loc[missing])
        return result


class GroupModeImputer(BaseEstimator, TransformerMixin):
    """Fill a categorical column with its most common value within a group."""

    def __init__(self, column="MSZoning", group="Neighborhood"):
        self.column = column
        self.group = group

    def fit(self, X, y=None):
        observed = X.dropna(subset=[self.column])
        self.by_group_ = observed.groupby(self.group)[self.column].agg(lambda s: s.mode().iloc[0])
        self.overall_ = observed[self.column].mode().iloc[0]
        return self

    def transform(self, X):
        result = X.copy()
        missing = result[self.column].isna()
        result.loc[missing, self.column] = (
            result.loc[missing, self.group].map(self.by_group_).fillna(self.overall_)
        )
        return result


class SkewedLogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to columns that were strongly right-skewed in the training data."""

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        is_binary = np.array([len(np.unique(column)) <= 2 for column in X.T])
        self.log_columns_ = (skew(X, axis=0) > self.threshold) & (X.min(axis=0) >= 0) & ~is_binary
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.log_columns_] = np.log1p(X[:, self.log_columns_])
        return X


def make_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log_skewed", SkewedLogTransformer()),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ])


def make_pipeline(regressor, frontage_imputer=None):
    """Full pipeline: imputers -> preprocessing -> model trained on log prices.

    frontage_imputer defaults to the model-based LotFrontageImputer chosen in notebook section 7.
    """
    return Pipeline([
        ("frontage", clone(frontage_imputer) if frontage_imputer is not None else LotFrontageImputer("model")),
        ("zoning", GroupModeImputer("MSZoning", "Neighborhood")),
        ("preprocess", make_preprocessor()),
        ("model", TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)),
    ])
