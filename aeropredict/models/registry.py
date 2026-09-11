"""Saving and loading a trained bundle (preprocessor + models + metrics) per dataset.

Layout on disk::

    artifacts/FD001/
        preprocessor.joblib
        linear.joblib
        lgbm.joblib
        lstm.pt
        metrics.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import torch

from ..config import ARTIFACT_DIR
from ..features import Preprocessor
from .baseline import LinearRUL, LightGBMRUL
from .sequence import LSTMRUL


@dataclass
class Bundle:
    dataset: str
    preprocessor: Preprocessor
    models: dict[str, object] = field(default_factory=dict)  # name -> fitted model
    metrics: dict = field(default_factory=dict)

    def predict(self, df, model: str):
        """Predict RUL for the *last* cycle of every unit in ``df``.

        Returns a pandas Series indexed by unit.
        """
        import pandas as pd

        m = self.models[model]
        if isinstance(m, LSTMRUL):
            X, meta = self.preprocessor.windows(df, last_only=True)
            pred = m.predict(X)
            return pd.Series(pred, index=pd.Index(meta["unit"], name="unit"))
        feats = self.preprocessor.engineer(df)
        last = feats.groupby("unit").tail(1)
        pred = m.predict(last[self.preprocessor.feature_columns()])
        return pd.Series(pred, index=pd.Index(last["unit"].to_numpy(), name="unit"))


def save_bundle(bundle: Bundle, artifact_dir: Path = ARTIFACT_DIR) -> Path:
    out = artifact_dir / bundle.dataset
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.preprocessor, out / "preprocessor.joblib")
    for name, model in bundle.models.items():
        if isinstance(model, LSTMRUL):
            torch.save(model.state(), out / "lstm.pt")
        else:
            joblib.dump(model, out / f"{name}.joblib")
    (out / "metrics.json").write_text(json.dumps(bundle.metrics, indent=2))
    return out


def load_bundle(dataset: str, artifact_dir: Path = ARTIFACT_DIR, device: str | None = None) -> Bundle:
    src = artifact_dir / dataset.upper()
    if not (src / "preprocessor.joblib").exists():
        raise FileNotFoundError(
            f"No trained artifacts for {dataset} in {artifact_dir}. "
            f"Run: python scripts/train.py --dataset {dataset}"
        )
    pre: Preprocessor = joblib.load(src / "preprocessor.joblib")
    models: dict[str, object] = {}
    for name, cls in (("linear", LinearRUL), ("lgbm", LightGBMRUL)):
        p = src / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
    if (src / "lstm.pt").exists():
        state = torch.load(src / "lstm.pt", map_location="cpu", weights_only=False)
        models["lstm"] = LSTMRUL.from_state(state, device=device)
    metrics = {}
    if (src / "metrics.json").exists():
        metrics = json.loads((src / "metrics.json").read_text())
    return Bundle(dataset=dataset.upper(), preprocessor=pre, models=models, metrics=metrics)


def available_datasets(artifact_dir: Path = ARTIFACT_DIR) -> list[str]:
    if not artifact_dir.exists():
        return []
    return sorted(p.name for p in artifact_dir.iterdir() if (p / "preprocessor.joblib").exists())
