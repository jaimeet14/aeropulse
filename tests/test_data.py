import numpy as np
import pandas as pd
import pytest

from aeropredict.config import RAW_COLUMNS, RUL_CAP
from aeropredict.data import add_rul, load_subset


def _frame(lengths: dict[int, int]) -> pd.DataFrame:
    rows = [[u, c, 0, 0, 100, *([1.0] * 21)] for u, n in lengths.items() for c in range(1, n + 1)]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_add_rul_counts_down_to_zero_and_caps():
    df = add_rul(_frame({1: 5, 2: 200}), cap=125)
    u1 = df[df.unit == 1]
    assert list(u1["rul_true"]) == [4, 3, 2, 1, 0]
    assert list(u1["rul"]) == [4, 3, 2, 1, 0]
    u2 = df[df.unit == 2]
    assert u2["rul_true"].max() == 199
    assert u2["rul"].max() == 125  # piecewise-linear target
    assert (u2["rul"] == 125).sum() == 75  # first 75 cycles are all "at least 125 left"


def test_add_rul_without_cap():
    df = add_rul(_frame({1: 200}), cap=None)
    assert df["rul"].max() == 199


def test_load_subset_parses_files(data_dir, subset):
    assert subset.name == "FD001"
    assert set(subset.train.columns) >= set(RAW_COLUMNS) | {"rul", "rul_true"}
    assert subset.train["rul"].max() <= RUL_CAP
    assert subset.test["unit"].nunique() == len(subset.test_rul)
    # last row per test unit is what gets scored
    assert len(subset.test_last) == len(subset.test_rul)


def test_load_subset_rejects_mismatched_rul_file(tmp_path):
    train = _frame({1: 40})
    test = _frame({1: 20, 2: 20})
    train.to_csv(tmp_path / "train_FD009.txt", sep=" ", header=False, index=False)
    test.to_csv(tmp_path / "test_FD009.txt", sep=" ", header=False, index=False)
    np.savetxt(tmp_path / "RUL_FD009.txt", [10], fmt="%d")  # only one value for two units
    with pytest.raises(ValueError, match="RUL file"):
        load_subset("FD009", tmp_path)
