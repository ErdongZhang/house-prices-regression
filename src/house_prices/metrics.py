"""The competition metric."""

import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error


def rmse_score(y_true, y_pred):
    """Return RMSE in the same units as the target."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def rmsle_score(y_true, y_pred):
    """Return RMSE between log prices (the competition metric)."""
    return rmse_score(np.log1p(y_true), np.log1p(y_pred))


rmsle_scorer = make_scorer(rmsle_score, greater_is_better=False)
