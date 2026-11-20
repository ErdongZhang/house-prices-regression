"""The final model: a weighted blend of pipelines (notebook sections 8, 13 and 14)."""

import tomllib

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from .features import make_pipeline
from .settings import RANDOM_STATE

UNUSUAL_SALES = ["Abnorml", "Family", "Alloca", "AdjLand"]

# Model types a config file may name; add an entry here to make a new type configurable
MODEL_TYPES = {
    "Ridge": Ridge,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}


def blend(predictions_list, weights=None):
    """Weighted average in log space (equal weights by default), returned in dollars."""
    return np.expm1(np.average([np.log1p(p) for p in predictions_list], axis=0, weights=weights))


class BlendRegressor(BaseEstimator, RegressorMixin):
    """Weighted average of several pipelines' log-price predictions."""

    def __init__(self, members, weights):
        self.members = members  # list of (name, unfitted pipeline)
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        params = {} if sample_weight is None else {"model__sample_weight": sample_weight}
        self.fitted_ = [clone(pipeline).fit(X, y, **params) for _, pipeline in self.members]
        return self

    def predict(self, X):
        return blend([model.predict(X) for model in self.fitted_], self.weights)


def unusual_sale_weights(df, unusual_weight):
    """Training weight per house: unusual_weight for unusual sales, 1 for all others."""
    return np.where(df["SaleCondition"].isin(UNUSUAL_SALES), unusual_weight, 1.0)


def load_config(path):
    with open(path, "rb") as file:
        return tomllib.load(file)


def build_model(config):
    """Unfitted BlendRegressor described by a config dictionary (see configs/final_model.toml)."""
    members = []
    for member in config["members"]:
        params = dict(member.get("params", {}))
        regressor_class = MODEL_TYPES[member["type"]]
        if "random_state" in regressor_class().get_params():
            params.setdefault("random_state", RANDOM_STATE)
        members.append((member["name"], make_pipeline(regressor_class(**params))))
    weights = np.array([member["weight"] for member in config["members"]], dtype=float)
    return BlendRegressor(members, weights)
