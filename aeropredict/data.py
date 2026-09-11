"""Loading and labelling the C-MAPSS sub-datasets.

Each sub-dataset FD00x ships three whitespace-separated files:

* ``train_FD00x.txt`` — full run-to-failure trajectories, one row per (unit, cycle).
* ``test_FD00x.txt``  — trajectories truncated some time *before* failure.
* ``RUL_FD00x.txt``   — one integer per test unit: the true remaining cycles at the
  point of truncation, used only for scoring.

Train and test engines are disjoint units, so the usual time-series split leakage
is not a concern here; the leakage to guard against is feature engineering that
looks *forward* within a unit's own trajectory (see :mod:`aeropredict.features`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, RAW_COLUMNS, RUL_CAP


def _read_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(RAW_COLUMNS)]  # some mirrors add trailing empty columns
    df.columns = RAW_COLUMNS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def add_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Label a run-to-failure frame with remaining useful life per row.

    RUL at a row = (last cycle of that unit) - (this cycle). With ``cap`` set, the
    label becomes piecewise-linear: ``min(RUL, cap)``. The uncapped value is kept in
    ``rul_true`` for honest reporting.
    """
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    out = df.copy()
    out["rul_true"] = (max_cycle - out["cycle"]).astype(int)
    out["rul"] = out["rul_true"].clip(upper=cap) if cap is not None else out["rul_true"]
    return out


@dataclass
class Subset:
    """One C-MAPSS sub-dataset, loaded and labelled."""

    name: str
    train: pd.DataFrame  # labelled with rul / rul_true
    test: pd.DataFrame  # truncated trajectories, unlabelled per-row
    test_rul: pd.Series  # true RUL at truncation, indexed by unit

    @property
    def test_last(self) -> pd.DataFrame:
        """The last observed row of every test unit — the rows that get scored."""
        return self.test.groupby("unit").tail(1).set_index("unit")


def load_subset(name: str, data_dir: Path = DATA_DIR, cap: int | None = RUL_CAP) -> Subset:
    name = name.upper()
    train = add_rul(_read_txt(data_dir / f"train_{name}.txt"), cap=cap)
    test = _read_txt(data_dir / f"test_{name}.txt")
    rul_values = np.loadtxt(data_dir / f"RUL_{name}.txt", dtype=int).reshape(-1)
    units = sorted(test["unit"].unique())
    if len(rul_values) != len(units):
        raise ValueError(
            f"{name}: RUL file has {len(rul_values)} rows but test set has {len(units)} units"
        )
    test_rul = pd.Series(rul_values, index=pd.Index(units, name="unit"), name="rul_true")
    return Subset(name=name, train=train, test=test, test_rul=test_rul)
