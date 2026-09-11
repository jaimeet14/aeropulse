"""Evaluation metrics for RUL prediction.

RMSE is the obvious metric but it treats early and late predictions symmetrically.
In the real world they are not symmetric: predicting *more* life than an engine
actually has (a late prediction, ``pred > true``) means maintenance is scheduled
after the failure — the expensive, dangerous outcome. Predicting less life (early)
just costs some unused engine life. The PHM08 / C-MAPSS challenge therefore scores
with an asymmetric exponential penalty that grows faster for late predictions.

We compute and report both, and never optimise for one while hiding the other.
"""

from __future__ import annotations

import numpy as np

from .config import URGENCY_THRESHOLDS


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric scoring function (lower is better).

    d = pred - true.  s = sum( exp(-d/13) - 1 ) for d < 0 (early, mild penalty)
                        + sum( exp( d/10) - 1 ) for d >= 0 (late, harsh penalty)
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = np.exp(-d[d < 0] / 13.0) - 1.0
    late = np.exp(d[d >= 0] / 10.0) - 1.0
    return float(early.sum() + late.sum())


def late_fraction(y_true, y_pred) -> float:
    """Share of predictions that over-estimate remaining life (the costly direction)."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(d > 0))


def urgency(rul: float) -> str:
    """Map a RUL (in cycles) to a maintenance-urgency bucket."""
    if rul <= URGENCY_THRESHOLDS["critical"]:
        return "critical"
    if rul <= URGENCY_THRESHOLDS["warning"]:
        return "warning"
    return "healthy"


def urgency_accuracy(y_true, y_pred) -> float:
    """Accuracy of the classification variant: did we land in the right bucket?"""
    t = [urgency(v) for v in np.asarray(y_true, float)]
    p = [urgency(v) for v in np.asarray(y_pred, float)]
    return float(np.mean([a == b for a, b in zip(t, p)]))


def summarise(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": round(rmse(y_true, y_pred), 3),
        "mae": round(mae(y_true, y_pred), 3),
        "nasa_score": round(nasa_score(y_true, y_pred), 1),
        "late_fraction": round(late_fraction(y_true, y_pred), 3),
        "urgency_accuracy": round(urgency_accuracy(y_true, y_pred), 3),
        "n": int(len(np.asarray(y_true))),
    }
