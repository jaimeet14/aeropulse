from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModelName = Literal["linear", "lgbm", "lstm"]
Urgency = Literal["healthy", "warning", "critical"]


class Reading(BaseModel):
    """One operating cycle of one engine: 3 operating settings + 21 sensor values,
    in the C-MAPSS column order."""

    cycle: int = Field(ge=1)
    settings: list[float] = Field(min_length=3, max_length=3)
    sensors: list[float] = Field(min_length=21, max_length=21)


class PredictRequest(BaseModel):
    dataset: str = Field(default="FD001", description="Which trained bundle to use")
    model: ModelName = "lgbm"
    readings: list[Reading] = Field(min_length=1, description="Chronological readings of ONE engine")

    @field_validator("dataset")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("readings")
    @classmethod
    def _monotonic(cls, v: list[Reading]) -> list[Reading]:
        cycles = [r.cycle for r in v]
        if cycles != sorted(cycles) or len(set(cycles)) != len(cycles):
            raise ValueError("readings must be in strictly increasing cycle order")
        return v


class PredictResponse(BaseModel):
    dataset: str
    model: ModelName
    rul: float = Field(description="Predicted remaining useful life in cycles (capped)")
    rul_cap: int
    urgency: Urgency
    n_readings: int
    last_cycle: int
    window: int
    note: str | None = None


class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    nasa_score: float
    late_fraction: float
    urgency_accuracy: float
    n: int


class DatasetInfo(BaseModel):
    dataset: str
    n_regimes: int
    sensors_kept: list[str]
    sensors_dropped: list[str]
    n_train_units: int
    n_test_units: int
    rul_cap: int
    window: int
    models: dict[str, dict]


class EngineStatus(BaseModel):
    unit: int
    cycle: int
    cycles_available: int
    rul_pred: float
    rul_true: int | None
    urgency: Urgency
    regime: int
    at_end: bool
    shop_visits: int = 0


class FleetResponse(BaseModel):
    dataset: str
    model: ModelName
    tick: int
    engines: list[EngineStatus]
    summary: dict[str, int]


class EngineHistory(BaseModel):
    unit: int
    dataset: str
    model: ModelName
    points: list[dict]  # {cycle, rul_pred, rul_true, urgency}
    sensors: list[str]
    sensor_trace: list[dict]  # {cycle, <sensor>: normalised value, ...}
