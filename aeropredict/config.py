"""Central configuration. Everything is overridable via environment variables so the
API, the training scripts and the tests can point at different data/artifact folders."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("AEROPREDICT_DATA_DIR", ROOT / "data" / "raw"))
ARTIFACT_DIR = Path(os.getenv("AEROPREDICT_ARTIFACT_DIR", ROOT / "artifacts"))

DATASETS = ("FD001", "FD002", "FD003", "FD004")

# Number of distinct operating regimes per sub-dataset (from the C-MAPSS readme).
# FD001/FD003 run at sea level only; FD002/FD004 cycle through six flight conditions.
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RAW_COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

# Piecewise-linear RUL target: an engine's true RUL early in life is unobservable from
# the sensors (nothing is degrading yet), so the standard trick in the C-MAPSS
# literature is to cap the label at a constant. 125 is the usual value.
RUL_CAP = int(os.getenv("AEROPREDICT_RUL_CAP", "125"))

# Sensors whose per-regime standard deviation falls below this are treated as flat
# and dropped. On FD001 this removes s1, s5, s6, s10, s16, s18, s19.
FLAT_SENSOR_STD_THRESHOLD = 1e-3

# Rolling-window length for engineered features and for the sequence model input.
WINDOW = int(os.getenv("AEROPREDICT_WINDOW", "30"))

# Maintenance-urgency buckets (in remaining cycles). Used by the classification
# variant of the problem and by the API's urgency flag.
URGENCY_THRESHOLDS = {"critical": 20, "warning": 50}  # else "healthy"

DOWNLOAD_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
