"""A simulated fleet of engines "in service", for the dashboard.

Each engine is a unit from the official *test* split of a sub-dataset. The
simulator reveals its trajectory one cycle per tick, re-predicts RUL from the
readings seen so far, and compares against the true RUL — which is known for test
units because ``RUL_FD00x.txt`` gives the life remaining at the truncation point,
so ``rul_true(cycle) = rul_at_truncation + (last_cycle - cycle)``.

Nothing here is real telemetry: it replays NASA's simulated trajectories.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SENSOR_INFO, WINDOW
from .data import Subset
from .metrics import urgency
from .models import Bundle


@dataclass
class Engine:
    unit: int
    frame: pd.DataFrame  # full truncated test trajectory for this unit
    rul_at_end: int
    cursor: int  # number of cycles revealed so far
    history: list[dict] = field(default_factory=list)
    shop_visits: int = 0  # times this engine hit end-of-data and was "returned to service"

    @property
    def n_cycles(self) -> int:
        return len(self.frame)

    @property
    def at_end(self) -> bool:
        return self.cursor >= self.n_cycles

    def visible(self) -> pd.DataFrame:
        return self.frame.iloc[: self.cursor]

    def rul_true_at(self, cycle: int) -> int:
        last_cycle = int(self.frame["cycle"].iloc[-1])
        return self.rul_at_end + (last_cycle - cycle)


def _stratified_units(test_rul: pd.Series, n: int, rng: np.random.Generator) -> list[int]:
    """Pick ``n`` test units spread across low / mid / high remaining life.

    Test trajectories are truncated *before* failure, so a purely random sample is
    dominated by engines that never leave the healthy band. Sampling a third from
    each tercile of true RUL-at-truncation keeps the fleet view representative of
    all three urgency states without hand-picking specific engines.
    """
    ordered = test_rul.sort_values().index.to_numpy()
    n = min(n, len(ordered))
    terciles = np.array_split(ordered, 3)
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    chosen: list[int] = []
    for group, k in zip(terciles, per):
        chosen += [int(u) for u in rng.choice(group, size=min(k, len(group)), replace=False)]
    return sorted(chosen)


class FleetSimulator:
    def __init__(
        self,
        bundle: Bundle,
        subset: Subset,
        model: str = "lgbm",
        n_engines: int = 12,
        start_cycles: int = WINDOW,
        seed: int = 0,
    ):
        self.bundle = bundle
        self.subset = subset
        self.model = model
        self.tick_count = 0
        self._lock = threading.Lock()
        rng = np.random.default_rng(seed)
        chosen = _stratified_units(subset.test_rul, n_engines, rng)
        self.engines: dict[int, Engine] = {}
        for u in chosen:
            frame = subset.test[subset.test["unit"] == u].sort_values("cycle").reset_index(drop=True)
            # Stagger starting points so the fleet isn't in lock-step.
            start = min(len(frame), start_cycles + int(rng.integers(0, max(1, len(frame) - start_cycles))))
            eng = Engine(unit=int(u), frame=frame, rul_at_end=int(subset.test_rul.loc[u]), cursor=start)
            self.engines[int(u)] = eng
        self._predict_all()

    # ------------------------------------------------------------------ state
    def set_model(self, model: str) -> None:
        if model not in self.bundle.models:
            raise KeyError(model)
        with self._lock:
            self.model = model
            for e in self.engines.values():
                e.history.clear()
            self._predict_all()

    def tick(self, steps: int = 1) -> None:
        with self._lock:
            for _ in range(steps):
                self.tick_count += 1
                for e in self.engines.values():
                    if e.at_end:
                        # End of the recorded trajectory: send the engine to the shop
                        # and bring it back with a fresh history so the fleet keeps
                        # moving instead of freezing.
                        e.cursor = min(e.n_cycles, WINDOW)
                        e.history.clear()
                        e.shop_visits += 1
                    else:
                        e.cursor += 1
                self._predict_all()

    def reset(self) -> None:
        with self._lock:
            self.tick_count = 0
            for e in self.engines.values():
                e.cursor = min(e.n_cycles, WINDOW)
                e.history.clear()
                e.shop_visits = 0
            self._predict_all()

    # ------------------------------------------------------------ prediction
    def _predict_all(self) -> None:
        frames = [e.visible() for e in self.engines.values()]
        df = pd.concat(frames, ignore_index=True)
        preds = self.bundle.predict(df, self.model)
        for e in self.engines.values():
            cycle = int(e.frame["cycle"].iloc[e.cursor - 1])
            if e.history and e.history[-1]["cycle"] == cycle:
                continue  # already at end of data; nothing new to record
            pred = float(preds.loc[e.unit])
            e.history.append(
                {
                    "cycle": cycle,
                    "rul_pred": round(pred, 2),
                    "rul_true": e.rul_true_at(cycle),
                    "urgency": urgency(pred),
                }
            )

    # ---------------------------------------------------------------- views
    def status(self) -> list[dict]:
        out = []
        pre = self.bundle.preprocessor
        for e in self.engines.values():
            last = e.history[-1]
            regime = int(pre.assign_regime(e.frame.iloc[[e.cursor - 1]])[0])
            out.append(
                {
                    "unit": e.unit,
                    "cycle": last["cycle"],
                    "cycles_available": e.n_cycles,
                    "rul_pred": last["rul_pred"],
                    "rul_true": last["rul_true"],
                    "urgency": last["urgency"],
                    "regime": regime,
                    "at_end": e.at_end,
                    "shop_visits": e.shop_visits,
                }
            )
        return sorted(out, key=lambda r: r["rul_pred"])

    def summary(self) -> dict[str, int]:
        counts = {"healthy": 0, "warning": 0, "critical": 0}
        for e in self.engines.values():
            counts[e.history[-1]["urgency"]] += 1
        return counts

    def engine_history(self, unit: int) -> dict:
        e = self.engines[unit]
        pre = self.bundle.preprocessor
        norm = pre.normalise(e.visible())
        trace = norm[["cycle", *pre.sensors]].round(3).to_dict(orient="records")
        return {
            "unit": unit,
            "dataset": self.bundle.dataset,
            "model": self.model,
            "points": list(e.history),
            "sensors": list(pre.sensors),
            "sensor_info": {k: SENSOR_INFO[k] for k in pre.sensors},
            "sensor_trace": trace,
        }
