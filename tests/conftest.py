"""Shared fixtures: a small synthetic C-MAPSS-shaped dataset written to a temp dir,
plus a tiny trained bundle. Nothing here needs the NASA download, so the suite
runs offline and in CI in well under a minute.

Environment variables are set *before* aeropredict is imported so config.py picks
up the temp locations.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TMP = Path(tempfile.mkdtemp(prefix="aeropredict-test-"))
os.environ["AEROPREDICT_DATA_DIR"] = str(_TMP / "data")
os.environ["AEROPREDICT_ARTIFACT_DIR"] = str(_TMP / "artifacts")
os.environ["AEROPREDICT_AUTO_DOWNLOAD"] = "0"

from aeropredict.config import RAW_COLUMNS  # noqa: E402  (after env setup)

FLAT_SENSORS = {"s1", "s5", "s10", "s16", "s18", "s19"}
QUANTISED = "s6"  # flips between two values, like the real s6 on FD001


def make_trajectories(n_units: int, rng: np.random.Generator, n_regimes: int = 2,
                      min_len: int = 60, max_len: int = 140, truncate: bool = False):
    """Synthetic engines whose informative sensors drift with wear fraction."""
    rows, ruls = [], {}
    regime_offsets = np.linspace(0, 50, n_regimes)
    for unit in range(1, n_units + 1):
        life = int(rng.integers(min_len, max_len + 1))
        n_obs = life if not truncate else int(rng.integers(35, life))
        for cycle in range(1, n_obs + 1):
            wear = cycle / life
            regime = int(rng.integers(0, n_regimes))
            settings = [regime * 10.0 + rng.normal(0, 0.01), regime * 0.2 + rng.normal(0, 0.001), 100.0]
            sensors = []
            for i in range(1, 22):
                name = f"s{i}"
                if name in FLAT_SENSORS:
                    sensors.append(1.0 * i)
                elif name == QUANTISED:
                    sensors.append(21.60 + 0.01 * (rng.random() < 0.3))
                else:
                    base = 500.0 + i + regime_offsets[regime]
                    sensors.append(base + 8.0 * wear**2 * (1 if i % 2 else -1) + rng.normal(0, 0.3))
            rows.append([unit, cycle, *settings, *sensors])
        ruls[unit] = life - n_obs
    return pd.DataFrame(rows, columns=RAW_COLUMNS), pd.Series(ruls)


def write_subset(data_dir: Path, name: str, train: pd.DataFrame, test: pd.DataFrame, rul: pd.Series) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / f"train_{name}.txt", sep=" ", header=False, index=False)
    test.to_csv(data_dir / f"test_{name}.txt", sep=" ", header=False, index=False)
    np.savetxt(data_dir / f"RUL_{name}.txt", rul.sort_index().to_numpy(), fmt="%d")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    d = Path(os.environ["AEROPREDICT_DATA_DIR"])
    rng = np.random.default_rng(42)
    train, _ = make_trajectories(30, rng, n_regimes=2)
    test, rul = make_trajectories(8, rng, n_regimes=2, truncate=True)
    write_subset(d, "FD001", train, test, rul)
    return d


@pytest.fixture(scope="session")
def subset(data_dir):
    from aeropredict.data import load_subset

    return load_subset("FD001", data_dir)


@pytest.fixture(scope="session")
def bundle(subset):
    from aeropredict.train import train_subset

    # n_regimes=2 because the synthetic FD001 has two regimes (the real one has one).
    return train_subset(
        subset, lstm_epochs=3, lstm_kwargs={"hidden": 16, "layers": 1}, device="cpu", n_regimes=2
    )


@pytest.fixture(scope="session")
def artifact_dir(bundle) -> Path:
    from aeropredict.models import save_bundle

    d = Path(os.environ["AEROPREDICT_ARTIFACT_DIR"])
    save_bundle(bundle, d)
    return d
