import numpy as np
import pytest

from aeropredict.config import SENSOR_COLS
from aeropredict.features import Preprocessor
from tests.conftest import FLAT_SENSORS, QUANTISED


@pytest.fixture(scope="module")
def pre(subset):
    return Preprocessor("FD001", n_regimes=2, window=10).fit(subset.train)


def test_flat_and_quantised_sensors_are_dropped(pre):
    assert set(pre.dropped_sensors) == FLAT_SENSORS | {QUANTISED}
    assert set(pre.sensors) | set(pre.dropped_sensors) == set(SENSOR_COLS)
    assert len(pre.sensors) == 21 - len(FLAT_SENSORS) - 1


def test_regimes_are_recovered_from_settings(pre, subset):
    regime = pre.assign_regime(subset.train)
    assert set(regime) == {0, 1}
    # The synthetic data encodes the regime in setting_1 (0 vs 10); k-means must
    # split on it perfectly.
    truth = (subset.train["setting_1"] > 5).astype(int).to_numpy()
    agreement = max(np.mean(regime == truth), np.mean(regime != truth))
    assert agreement == 1.0


def test_normalisation_is_per_regime(pre, subset):
    norm = pre.normalise(subset.train)
    # Within each regime every kept sensor is standardised on the training data.
    for r, grp in norm.groupby("regime"):
        assert np.allclose(grp[pre.sensors].mean(), 0.0, atol=1e-6)
        assert np.allclose(grp[pre.sensors].std(ddof=0), 1.0, atol=1e-6)
    # Regimes have very different raw offsets; without per-regime scaling the
    # pooled std of a sensor would be far from 1.
    assert set(norm.columns) == {"unit", "cycle", "regime", *pre.sensors}


def test_engineered_features_do_not_look_forward(pre, subset):
    """Perturbing cycles after t must not change any feature at or before t."""
    df = subset.train[subset.train.unit == 1].copy()
    t = 25
    base = pre.engineer(df)
    tampered = df.copy()
    future = tampered["cycle"] > t
    tampered.loc[future, pre.sensors] += 1000.0
    changed = pre.engineer(tampered)
    cols = pre.feature_columns()
    before = base["cycle"] <= t
    assert np.allclose(base.loc[before, cols], changed.loc[before, cols])
    # sanity: the perturbation did change something after t
    assert not np.allclose(base.loc[~before, cols], changed.loc[~before, cols])


def test_engineered_features_shape_and_columns(pre, subset):
    feats = pre.engineer(subset.train)
    assert len(feats) == len(subset.train)
    cols = pre.feature_columns()
    assert len(cols) == 2 + 4 * len(pre.sensors)
    assert not feats[cols].isna().any().any()
    # rolling std at the very first cycle of a unit is 0, not NaN
    first = feats.groupby("unit").head(1)
    assert (first[[f"{s}_std" for s in pre.sensors]] == 0).all().all()


def test_windows_shapes_and_padding(pre, subset):
    X, meta = pre.windows(subset.train)
    n_units = subset.train["unit"].nunique()
    per_unit = subset.train.groupby("unit").size()
    expected = int((per_unit - pre.window + 1).clip(lower=1).sum())
    assert X.shape == (expected, pre.window, len(pre.sensors))
    assert len(meta) == expected

    X_last, meta_last = pre.windows(subset.test, last_only=True)
    assert X_last.shape[0] == subset.test["unit"].nunique() == len(meta_last)
    # the last window ends at each unit's final cycle
    last_cycles = subset.test.groupby("unit")["cycle"].max()
    assert (meta_last.set_index("unit")["cycle"] == last_cycles).all()

    # a unit shorter than the window is left-padded with its first reading
    short = subset.test[subset.test.unit == subset.test.unit.iloc[0]].head(4)
    Xs, _ = pre.windows(short, last_only=True)
    assert Xs.shape == (1, pre.window, len(pre.sensors))
    assert np.allclose(Xs[0, 0], Xs[0, pre.window - 4])


def test_unfitted_preprocessor_raises(subset):
    with pytest.raises(RuntimeError):
        Preprocessor("FD001").normalise(subset.train)
