"""AeroPulse RUL service.

Loads every trained bundle found in ``ARTIFACT_DIR`` at startup and exposes:

* ``POST /predict``       — RUL + maintenance-urgency flag for one engine's readings
* ``GET  /datasets``      — what is trained, with test-set metrics per model
* ``GET  /fleet``         — simulated fleet view (replays NASA test trajectories)
* ``POST /fleet/tick``    — advance the simulation one (or more) cycles
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..config import ARTIFACT_DIR, DATA_DIR, RAW_COLUMNS, RUL_CAP, SENSOR_INFO, SETTING_INFO
from ..data import load_subset
from ..download import dataset_present, download_cmapss
from ..fleet import FleetSimulator
from ..metrics import urgency
from ..models import Bundle, available_datasets, load_bundle
from .schemas import (
    DatasetInfo,
    EngineHistory,
    FleetResponse,
    ModelName,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger("aeropredict.api")


class State:
    bundles: dict[str, Bundle] = {}
    fleets: dict[str, FleetSimulator] = {}


state = State()


def _load_all() -> None:
    state.bundles.clear()
    for name in available_datasets(ARTIFACT_DIR):
        try:
            state.bundles[name] = load_bundle(name, ARTIFACT_DIR)
            log.info("loaded bundle %s with models %s", name, list(state.bundles[name].models))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("failed to load %s: %s", name, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    if not state.bundles:
        log.warning("no trained artifacts in %s — run scripts/train.py", ARTIFACT_DIR)
    yield


app = FastAPI(
    title="AeroPulse",
    version=__version__,
    description="Remaining-useful-life prediction for turbofan engines (NASA C-MAPSS).",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
)


# --------------------------------------------------------------------- helpers
def get_bundle(dataset: str) -> Bundle:
    name = dataset.upper()
    if name not in state.bundles:
        raise HTTPException(
            404, f"No trained model for {name}. Available: {sorted(state.bundles)}"
        )
    return state.bundles[name]


def get_fleet(dataset: str, model: ModelName | None = None) -> FleetSimulator:
    bundle = get_bundle(dataset)
    if bundle.dataset not in state.fleets:
        if not dataset_present(DATA_DIR, (bundle.dataset,)):
            if os.getenv("AEROPREDICT_AUTO_DOWNLOAD", "1") == "1":
                download_cmapss(DATA_DIR)
            else:
                raise HTTPException(503, "C-MAPSS data not present; run scripts/download_data.py")
        subset = load_subset(bundle.dataset, DATA_DIR)
        default_model = "lgbm" if "lgbm" in bundle.models else next(iter(bundle.models))
        state.fleets[bundle.dataset] = FleetSimulator(bundle, subset, model=default_model)
    fleet = state.fleets[bundle.dataset]
    if model and model != fleet.model:
        if model not in bundle.models:
            raise HTTPException(404, f"model {model} not trained for {bundle.dataset}")
        fleet.set_model(model)
    return fleet


# --------------------------------------------------------------------- routes
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "datasets": sorted(state.bundles)}


@app.get("/sensors")
def sensors() -> dict:
    """What each C-MAPSS column physically measures (Saxena et al., PHM08)."""
    return {"sensors": SENSOR_INFO, "settings": SETTING_INFO}


@app.get("/datasets", response_model=list[DatasetInfo])
def datasets() -> list[DatasetInfo]:
    out = []
    for name, b in sorted(state.bundles.items()):
        m = b.metrics
        out.append(
            DatasetInfo(
                dataset=name,
                n_regimes=m.get("n_regimes", b.preprocessor.n_regimes or 1),
                sensors_kept=b.preprocessor.sensors,
                sensors_dropped=b.preprocessor.dropped_sensors,
                n_train_units=m.get("n_train_units", 0),
                n_test_units=m.get("n_test_units", 0),
                rul_cap=m.get("rul_cap", RUL_CAP),
                window=b.preprocessor.window,
                models=m.get("models", {}),
                sensor_info={k: SENSOR_INFO[k] for k in b.preprocessor.sensors + b.preprocessor.dropped_sensors},
            )
        )
    return out


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    bundle = get_bundle(req.dataset)
    if req.model not in bundle.models:
        raise HTTPException(404, f"model {req.model} not trained for {bundle.dataset}")

    rows = [[1, r.cycle, *r.settings, *r.sensors] for r in req.readings]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    pred = float(bundle.predict(df, req.model).iloc[0])

    window = bundle.preprocessor.window
    note = None
    if len(df) < window:
        note = (
            f"Only {len(df)} reading{'s' if len(df) != 1 else ''} supplied; the model was trained on windows of "
            f"{window} cycles, so the earliest readings were padded. Expect lower confidence."
        )
    return PredictResponse(
        dataset=bundle.dataset,
        model=req.model,
        rul=round(pred, 2),
        rul_cap=bundle.metrics.get("rul_cap", RUL_CAP),
        urgency=urgency(pred),
        n_readings=len(df),
        last_cycle=int(df["cycle"].iloc[-1]),
        window=window,
        note=note,
    )


@app.get("/fleet", response_model=FleetResponse)
def fleet(
    dataset: str = Query("FD001"),
    model: ModelName | None = Query(None),
) -> FleetResponse:
    f = get_fleet(dataset, model)
    return FleetResponse(
        dataset=f.bundle.dataset, model=f.model, tick=f.tick_count,
        engines=f.status(), summary=f.summary(),
    )


@app.post("/fleet/tick", response_model=FleetResponse)
def fleet_tick(
    dataset: str = Query("FD001"),
    steps: int = Query(1, ge=1, le=50),
    model: ModelName | None = Query(None),
) -> FleetResponse:
    f = get_fleet(dataset, model)
    f.tick(steps)
    return FleetResponse(
        dataset=f.bundle.dataset, model=f.model, tick=f.tick_count,
        engines=f.status(), summary=f.summary(),
    )


@app.post("/fleet/reset", response_model=FleetResponse)
def fleet_reset(dataset: str = Query("FD001")) -> FleetResponse:
    f = get_fleet(dataset)
    f.reset()
    return FleetResponse(
        dataset=f.bundle.dataset, model=f.model, tick=f.tick_count,
        engines=f.status(), summary=f.summary(),
    )


@app.get("/fleet/{unit}", response_model=EngineHistory)
def fleet_engine(unit: int, dataset: str = Query("FD001")) -> EngineHistory:
    f = get_fleet(dataset)
    if unit not in f.engines:
        raise HTTPException(404, f"unit {unit} is not in the simulated fleet")
    return EngineHistory(**f.engine_history(unit))
