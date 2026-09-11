"""End-to-end training + evaluation for one C-MAPSS sub-dataset.

Evaluation is done on the official test split: one prediction per test unit, at
the last observed cycle, against ``RUL_FD00x.txt``. We score against the *capped*
truth (``min(true, RUL_CAP)``) because that is what the models are trained to
predict and it is the convention in the C-MAPSS literature — but the uncapped
numbers are recorded too so the effect of the cap is visible rather than hidden.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .config import RUL_CAP, WINDOW
from .data import Subset
from .features import Preprocessor
from .metrics import summarise
from .models import Bundle, LinearRUL, LightGBMRUL, LSTMRUL

log = logging.getLogger(__name__)


def train_subset(
    subset: Subset,
    models: tuple[str, ...] = ("linear", "lgbm", "lstm"),
    cap: int = RUL_CAP,
    window: int = WINDOW,
    lstm_epochs: int = 40,
    lstm_kwargs: dict | None = None,
    device: str | None = None,
    n_regimes: int | None = None,
) -> Bundle:
    t0 = time.time()
    pre = Preprocessor(subset.name, window=window, n_regimes=n_regimes).fit(subset.train)
    log.info(
        "%s: %d regimes, kept %d sensors, dropped %s",
        subset.name, pre.n_regimes, len(pre.sensors), pre.dropped_sensors,
    )

    bundle = Bundle(dataset=subset.name, preprocessor=pre)
    metrics: dict = {
        "dataset": subset.name,
        "rul_cap": cap,
        "window": window,
        "n_regimes": pre.n_regimes,
        "sensors_kept": pre.sensors,
        "sensors_dropped": pre.dropped_sensors,
        "n_train_units": int(subset.train["unit"].nunique()),
        "n_test_units": int(subset.test["unit"].nunique()),
        "models": {},
    }

    y_true_uncapped = subset.test_rul.sort_index()
    y_true = y_true_uncapped.clip(upper=cap)

    # ---- tabular models -------------------------------------------------
    if any(m in models for m in ("linear", "lgbm")):
        feats = pre.engineer(subset.train)
        cols = pre.feature_columns()
        X_train = feats[cols]
        y_train = subset.train.sort_values(["unit", "cycle"])["rul"].to_numpy()

        test_feats = pre.engineer(subset.test)
        test_last = test_feats.groupby("unit").tail(1).set_index("unit").sort_index()
        X_test = test_last[cols]

        if "linear" in models:
            m = LinearRUL(cap=cap).fit(X_train, y_train)
            bundle.models["linear"] = m
            metrics["models"]["linear"] = _score(m.predict(X_test), y_true, y_true_uncapped)
            log.info("linear: %s", metrics["models"]["linear"]["test"])

        if "lgbm" in models:
            m = LightGBMRUL(cap=cap).fit(X_train, y_train)
            bundle.models["lgbm"] = m
            metrics["models"]["lgbm"] = _score(m.predict(X_test), y_true, y_true_uncapped)
            metrics["models"]["lgbm"]["top_features"] = (
                m.feature_importance().head(15).round(1).to_dict()
            )
            log.info("lgbm: %s", metrics["models"]["lgbm"]["test"])

    # ---- sequence model -------------------------------------------------
    if "lstm" in models:
        X_seq, meta = pre.windows(subset.train)
        labels = subset.train.set_index(["unit", "cycle"])["rul"]
        y_seq = labels.loc[list(zip(meta["unit"], meta["cycle"]))].to_numpy()
        kwargs = dict(epochs=lstm_epochs, device=device)
        kwargs.update(lstm_kwargs or {})
        m = LSTMRUL(n_features=len(pre.sensors), cap=cap, **kwargs)
        m.fit(X_seq, y_seq, groups=meta["unit"].to_numpy(), log=log.info)
        bundle.models["lstm"] = m

        X_test_seq, test_meta = pre.windows(subset.test, last_only=True)
        pred = pd.Series(m.predict(X_test_seq), index=test_meta["unit"]).sort_index()
        metrics["models"]["lstm"] = _score(pred.to_numpy(), y_true, y_true_uncapped)
        metrics["models"]["lstm"]["epochs_trained"] = len(m.history)
        metrics["models"]["lstm"]["best_val_rmse"] = round(
            min(h["val_rmse"] for h in m.history), 3
        )
        metrics["models"]["lstm"]["device"] = m.device
        log.info("lstm: %s", metrics["models"]["lstm"]["test"])

    metrics["train_seconds"] = round(time.time() - t0, 1)
    bundle.metrics = metrics
    return bundle


def _score(pred: np.ndarray, y_true: pd.Series, y_true_uncapped: pd.Series) -> dict:
    return {
        "test": summarise(y_true.to_numpy(), pred),
        "test_uncapped_truth": summarise(y_true_uncapped.to_numpy(), pred),
    }
