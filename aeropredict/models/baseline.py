"""Tabular baselines on the engineered rolling features.

Both models predict the *capped* RUL directly and clip their output to
``[0, cap]`` — an engine cannot have negative life, and nothing in the sensors can
tell you an engine has 300 cycles left rather than 125 (see ``RUL_CAP``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RUL_CAP


class LinearRUL:
    """Ridge regression — the 'is a straight line already good enough?' check."""

    name = "linear"

    def __init__(self, cap: int = RUL_CAP, alpha: float = 1.0):
        self.cap = cap
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y) -> "LinearRUL":
        self.columns = list(X.columns)
        self.model.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(X[self.columns].to_numpy(dtype=float))
        return np.clip(pred, 0, self.cap)


class LightGBMRUL:
    """Gradient boosting on the same features — the strong tabular baseline."""

    name = "lgbm"

    def __init__(self, cap: int = RUL_CAP, **params):
        self.cap = cap
        defaults = dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=0,
            verbose=-1,
        )
        defaults.update(params)
        self.model = LGBMRegressor(**defaults)
        self.columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y) -> "LightGBMRUL":
        self.columns = list(X.columns)
        self.model.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(X[self.columns])
        return np.clip(pred, 0, self.cap)

    def feature_importance(self) -> pd.Series:
        imp = pd.Series(self.model.feature_importances_, index=self.columns)
        return imp.sort_values(ascending=False)
