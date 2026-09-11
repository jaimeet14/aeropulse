"""Preprocessing and feature engineering.

Two things make C-MAPSS awkward for a naive pipeline:

1. **Operating regimes.** FD002/FD004 cycle through six flight conditions (altitude,
   Mach, throttle). A sensor reading only means something relative to the regime it
   was taken in, so we cluster the three ``setting_*`` columns into regimes with
   k-means and standardise every sensor *within* its regime. For FD001/FD003 there is
   a single regime and this collapses to plain standardisation.

2. **Flat sensors.** Several of the 21 sensors are constant (or constant within each
   regime). They carry no information and add noise to a model; we detect them from
   the training data rather than hard-coding a list, so the choice is auditable.

Rolling features are strictly backward-looking (``rolling(...)`` over past cycles of
the *same* unit), so a feature at cycle *t* never sees cycle *t+1*. That is the one
leakage risk this dataset actually has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import (
    FLAT_SENSOR_STD_THRESHOLD,
    N_REGIMES,
    SENSOR_COLS,
    SETTING_COLS,
    WINDOW,
)


@dataclass
class Preprocessor:
    """Regime detection + per-regime sensor standardisation + flat-sensor pruning."""

    dataset: str
    n_regimes: int | None = None
    window: int = WINDOW
    flat_threshold: float = FLAT_SENSOR_STD_THRESHOLD
    # fitted state
    kmeans: KMeans | None = None
    regime_mean: pd.DataFrame | None = None  # index = regime, columns = SENSOR_COLS
    regime_std: pd.DataFrame | None = None
    sensors: list[str] = field(default_factory=list)  # sensors kept after pruning
    dropped_sensors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ fitting
    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        if self.n_regimes is None:
            self.n_regimes = N_REGIMES.get(self.dataset.upper(), 1)

        regimes = self._fit_regimes(train)
        grouped = train[SENSOR_COLS].groupby(regimes)
        self.regime_mean = grouped.mean()
        std = grouped.std(ddof=0)

        # A sensor is flat if, inside *every* regime, it either barely moves or only
        # ever takes one or two distinct values (a quantised constant such as s6 on
        # FD001, which flips between 21.60 and 21.61 — noise, not signal).
        nunique = grouped.nunique()
        flat = ((std < self.flat_threshold) | (nunique <= 2)).all(axis=0)
        self.dropped_sensors = [s for s in SENSOR_COLS if flat[s]]
        self.sensors = [s for s in SENSOR_COLS if not flat[s]]
        # Guard flat/zero-variance entries so transform never divides by zero.
        self.regime_std = std.where(std >= self.flat_threshold, 1.0)
        return self

    def _fit_regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_regimes <= 1:
            self.kmeans = None
            return np.zeros(len(df), dtype=int)
        self.kmeans = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=0)
        return self.kmeans.fit_predict(df[SETTING_COLS].to_numpy())

    # --------------------------------------------------------------- transform
    def assign_regime(self, df: pd.DataFrame) -> np.ndarray:
        if self.kmeans is None:
            return np.zeros(len(df), dtype=int)
        return self.kmeans.predict(df[SETTING_COLS].to_numpy())

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return unit, cycle, regime and the kept sensors standardised per regime."""
        self._check_fitted()
        regime = self.assign_regime(df)
        mean = self.regime_mean.loc[regime, self.sensors].to_numpy()
        std = self.regime_std.loc[regime, self.sensors].to_numpy()
        out = df[["unit", "cycle"]].copy().reset_index(drop=True)
        out["regime"] = regime
        out[self.sensors] = (df[self.sensors].to_numpy() - mean) / std
        return out

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tabular features for the gradient-boosting / linear baselines.

        Per kept sensor: current value, rolling mean, rolling std and rolling slope
        (least-squares trend over the window). Windows only look backwards within the
        same unit, and shrink at the start of a trajectory (``min_periods=1``).
        """
        norm = self.normalise(df)
        norm = norm.sort_values(["unit", "cycle"]).reset_index(drop=True)
        w = self.window
        g = norm.groupby("unit", sort=False)

        feats = {"unit": norm["unit"], "cycle": norm["cycle"], "regime": norm["regime"]}
        for s in self.sensors:
            r = g[s].rolling(w, min_periods=1)
            feats[s] = norm[s]
            feats[f"{s}_mean"] = r.mean().reset_index(level=0, drop=True)
            feats[f"{s}_std"] = r.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
            feats[f"{s}_slope"] = _rolling_slope(norm, s, w)
        return pd.DataFrame(feats)

    def feature_columns(self) -> list[str]:
        cols = ["cycle", "regime"]
        for s in self.sensors:
            cols += [s, f"{s}_mean", f"{s}_std", f"{s}_slope"]
        return cols

    def windows(self, df: pd.DataFrame, last_only: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
        """Sliding windows of normalised sensors for the sequence model.

        Returns ``(X, meta)`` where ``X`` has shape ``(n, window, n_sensors)`` and
        ``meta`` holds the ``unit``/``cycle`` of each window's *last* row, so labels
        can be joined back on. Units shorter than the window are left-padded by
        repeating their first reading. With ``last_only`` only the final window of
        each unit is returned (what the test set is scored on).
        """
        norm = self.normalise(df).sort_values(["unit", "cycle"])
        w = self.window
        xs, units, cycles = [], [], []
        for unit, grp in norm.groupby("unit", sort=False):
            arr = grp[self.sensors].to_numpy(dtype=np.float32)
            cyc = grp["cycle"].to_numpy()
            if len(arr) < w:
                pad = np.repeat(arr[:1], w - len(arr), axis=0)
                arr = np.vstack([pad, arr])
                cyc = np.concatenate([np.full(w - len(cyc), cyc[0]), cyc])
            starts = [len(arr) - w] if last_only else range(0, len(arr) - w + 1)
            for i in starts:
                xs.append(arr[i : i + w])
                units.append(unit)
                cycles.append(int(cyc[i + w - 1]))
        X = np.stack(xs) if xs else np.empty((0, w, len(self.sensors)), dtype=np.float32)
        meta = pd.DataFrame({"unit": units, "cycle": cycles})
        return X, meta

    def _check_fitted(self) -> None:
        if self.regime_mean is None:
            raise RuntimeError("Preprocessor.fit() must be called first")


def _rolling_slope(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Backward-looking least-squares slope of ``col`` against cycle, per unit.

    Uses rolling sums so it is O(n) rather than fitting a polyfit per row:
    slope = (n*sum(ty) - sum(t)*sum(y)) / (n*sum(t^2) - sum(t)^2).
    """
    t = df["cycle"].astype(float)
    y = df[col].astype(float)
    tmp = pd.DataFrame(
        {"unit": df["unit"], "t": t, "y": y, "ty": t * y, "tt": t * t, "one": 1.0}
    )
    r = tmp.groupby("unit", sort=False).rolling(window, min_periods=2)
    sums = r[["t", "y", "ty", "tt", "one"]].sum().reset_index(level=0, drop=True)
    n = sums["one"]
    num = n * sums["ty"] - sums["t"] * sums["y"]
    den = n * sums["tt"] - sums["t"] ** 2
    slope = (num / den.replace(0.0, np.nan)).fillna(0.0)
    return slope.reindex(df.index)
