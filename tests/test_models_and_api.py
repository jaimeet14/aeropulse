import numpy as np
import pytest
from fastapi.testclient import TestClient

from aeropredict.config import RUL_CAP
from aeropredict.models import load_bundle


# ----------------------------------------------------------------- training
def test_bundle_trains_all_three_models(bundle):
    assert set(bundle.models) == {"linear", "lgbm", "lstm"}
    m = bundle.metrics
    assert m["n_regimes"] == 2
    for name in ("linear", "lgbm", "lstm"):
        res = m["models"][name]["test"]
        assert res["n"] == m["n_test_units"]
        assert 0 <= res["late_fraction"] <= 1
        assert res["rmse"] >= 0 and res["nasa_score"] >= 0
    assert m["models"]["lstm"]["epochs_trained"] >= 1
    assert "top_features" in m["models"]["lgbm"]


def test_models_beat_a_constant_predictor(bundle, subset):
    """A tiny sanity floor: on the synthetic data every model must do better than
    always predicting the cap (the classic degenerate answer)."""
    y = subset.test_rul.clip(upper=RUL_CAP).to_numpy()
    naive_rmse = float(np.sqrt(np.mean((RUL_CAP - y) ** 2)))
    for name in ("linear", "lgbm"):
        assert bundle.metrics["models"][name]["test"]["rmse"] < naive_rmse


def test_predictions_are_clipped_to_cap(bundle, subset):
    for name in bundle.models:
        pred = bundle.predict(subset.test, name)
        assert len(pred) == subset.test["unit"].nunique()
        assert pred.between(0, RUL_CAP).all()


def test_save_and_load_round_trip(bundle, artifact_dir, subset):
    loaded = load_bundle("FD001", artifact_dir)
    assert set(loaded.models) == set(bundle.models)
    assert loaded.preprocessor.sensors == bundle.preprocessor.sensors
    assert loaded.metrics["models"]["lgbm"]["test"] == bundle.metrics["models"]["lgbm"]["test"]
    for name in bundle.models:
        a = bundle.predict(subset.test, name).to_numpy()
        b = loaded.predict(subset.test, name).to_numpy()
        assert np.allclose(a, b, atol=1e-4)


def test_load_missing_bundle_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/train.py"):
        load_bundle("FD004", tmp_path)


# ---------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(artifact_dir):
    from aeropredict.api.main import app

    with TestClient(app) as c:
        yield c


def _readings(subset, unit: int, n: int | None = None):
    df = subset.test[subset.test.unit == unit].sort_values("cycle")
    if n:
        df = df.head(n)
    return [
        {
            "cycle": int(r.cycle),
            "settings": [r.setting_1, r.setting_2, r.setting_3],
            "sensors": [getattr(r, f"s{i}") for i in range(1, 22)],
        }
        for r in df.itertuples()
    ]


def test_health_lists_trained_datasets(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["datasets"] == ["FD001"]


def test_datasets_endpoint_exposes_metrics_and_pruned_sensors(client):
    body = client.get("/datasets").json()
    assert len(body) == 1
    d = body[0]
    assert d["dataset"] == "FD001"
    assert "s1" in d["sensors_dropped"]
    assert set(d["models"]) == {"linear", "lgbm", "lstm"}
    assert "nasa_score" in d["models"]["lgbm"]["test"]


@pytest.mark.parametrize("model", ["linear", "lgbm", "lstm"])
def test_predict_returns_rul_and_urgency(client, subset, model):
    unit = int(subset.test["unit"].iloc[0])
    r = client.post("/predict", json={"dataset": "fd001", "model": model, "readings": _readings(subset, unit)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == model and body["dataset"] == "FD001"
    assert 0 <= body["rul"] <= body["rul_cap"]
    assert body["urgency"] in {"healthy", "warning", "critical"}
    assert body["note"] is None


def test_predict_short_history_is_padded_and_flagged(client, subset):
    unit = int(subset.test["unit"].iloc[0])
    r = client.post("/predict", json={"dataset": "FD001", "model": "lstm", "readings": _readings(subset, unit, n=3)})
    assert r.status_code == 200
    assert "padded" in r.json()["note"]


def test_predict_validation_errors(client, subset):
    unit = int(subset.test["unit"].iloc[0])
    good = _readings(subset, unit, n=5)
    # unknown dataset
    r = client.post("/predict", json={"dataset": "FD004", "model": "lgbm", "readings": good})
    assert r.status_code == 404
    # out-of-order cycles
    bad = list(reversed(good))
    r = client.post("/predict", json={"dataset": "FD001", "model": "lgbm", "readings": bad})
    assert r.status_code == 422
    # wrong sensor count
    bad = [{**good[0], "sensors": good[0]["sensors"][:20]}]
    r = client.post("/predict", json={"dataset": "FD001", "model": "lgbm", "readings": bad})
    assert r.status_code == 422
    # unknown model name
    r = client.post("/predict", json={"dataset": "FD001", "model": "xgboost", "readings": good})
    assert r.status_code == 422


def test_fleet_simulation_advances_and_tracks_truth(client):
    f = client.get("/fleet?dataset=FD001").json()
    assert f["tick"] == 0 and f["model"] == "lgbm"
    assert len(f["engines"]) == 8  # synthetic test set has 8 units
    assert sum(f["summary"].values()) == 8
    # sorted by predicted RUL ascending: most urgent first
    ruls = [e["rul_pred"] for e in f["engines"]]
    assert ruls == sorted(ruls)

    before = {e["unit"]: e["cycle"] for e in f["engines"]}
    f2 = client.post("/fleet/tick?dataset=FD001&steps=3").json()
    assert f2["tick"] == 3
    for e in f2["engines"]:
        # either advanced, or hit end-of-data and was returned to service
        assert e["cycle"] >= before[e["unit"]] or e["shop_visits"] >= 1
        assert e["rul_true"] is not None

    unit = next(e["unit"] for e in f2["engines"] if e["shop_visits"] == 0)
    h = client.get(f"/fleet/{unit}?dataset=FD001").json()
    assert h["unit"] == unit
    assert len(h["points"]) == 4  # initial + 3 ticks
    assert {"cycle", "rul_pred", "rul_true", "urgency"} <= set(h["points"][0])
    assert len(h["sensor_trace"]) == h["points"][-1]["cycle"]
    # truth counts down by exactly one per tick
    truths = [p["rul_true"] for p in h["points"]]
    assert truths == list(range(truths[0], truths[0] - 4, -1))


def test_fleet_switches_model_and_resets(client):
    f = client.get("/fleet?dataset=FD001&model=lstm").json()
    assert f["model"] == "lstm"
    f = client.post("/fleet/reset?dataset=FD001").json()
    assert f["tick"] == 0
    r = client.get("/fleet/999999?dataset=FD001")
    assert r.status_code == 404
    r = client.get("/fleet?dataset=FD003")
    assert r.status_code == 404


def test_engine_returns_to_service_after_end_of_data(client):
    """Ticking far past the shortest trajectory must loop the engine, not freeze it."""
    f = client.post("/fleet/reset?dataset=FD001").json()
    for _ in range(4):
        f = client.post("/fleet/tick?dataset=FD001&steps=50").json()
    assert any(e["shop_visits"] >= 1 for e in f["engines"])
    assert all(e["cycle"] <= e["cycles_available"] for e in f["engines"])


def test_sensor_names_are_exposed(client):
    body = client.get("/sensors").json()
    assert body["sensors"]["s14"]["symbol"] == "NRc"
    assert body["settings"]["setting_1"]["description"] == "Altitude"
    d = client.get("/datasets").json()[0]
    assert set(d["sensor_info"]) == set(d["sensors_kept"]) | set(d["sensors_dropped"])
    f = client.get("/fleet?dataset=FD001").json()
    h = client.get(f"/fleet/{f['engines'][0]['unit']}?dataset=FD001").json()
    assert set(h["sensor_info"]) == set(h["sensors"])
    assert "description" in h["sensor_info"][h["sensors"][0]]
