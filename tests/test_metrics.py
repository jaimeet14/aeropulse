import math

import numpy as np

from aeropredict.metrics import (
    late_fraction,
    nasa_score,
    rmse,
    summarise,
    urgency,
    urgency_accuracy,
)


def test_perfect_prediction_scores_zero():
    y = np.array([10, 50, 100])
    assert rmse(y, y) == 0.0
    assert nasa_score(y, y) == 0.0
    assert late_fraction(y, y) == 0.0


def test_nasa_score_penalises_late_more_than_early():
    """Same absolute error, opposite sign: over-estimating remaining life (late)
    must cost more than under-estimating it (early)."""
    truth = np.array([50.0])
    early = nasa_score(truth, truth - 20)  # predicted 30, engine has 50 left
    late = nasa_score(truth, truth + 20)  # predicted 70, engine has 50 left
    assert rmse(truth, truth - 20) == rmse(truth, truth + 20)  # RMSE can't tell
    assert late > early
    assert math.isclose(early, math.exp(20 / 13) - 1)
    assert math.isclose(late, math.exp(20 / 10) - 1)


def test_nasa_score_grows_exponentially_with_error():
    truth = np.array([50.0])
    s10 = nasa_score(truth, truth + 10)
    s20 = nasa_score(truth, truth + 20)
    s40 = nasa_score(truth, truth + 40)
    assert s20 / s10 > 2 and s40 / s20 > 2


def test_late_fraction():
    truth = np.array([10, 10, 10, 10])
    pred = np.array([12, 8, 10, 15])
    assert late_fraction(truth, pred) == 0.5


def test_urgency_buckets():
    assert urgency(0) == "critical"
    assert urgency(20) == "critical"
    assert urgency(20.1) == "warning"
    assert urgency(50) == "warning"
    assert urgency(51) == "healthy"
    assert urgency(125) == "healthy"


def test_urgency_accuracy_and_summary():
    truth = [5, 30, 100, 60]
    pred = [10, 55, 90, 45]  # first & third right, second & fourth wrong bucket
    assert urgency_accuracy(truth, pred) == 0.5
    s = summarise(truth, pred)
    assert set(s) == {"rmse", "mae", "nasa_score", "late_fraction", "urgency_accuracy", "n"}
    assert s["n"] == 4
