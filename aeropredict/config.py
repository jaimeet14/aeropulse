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

# What the 21 sensors physically are, from Saxena, Goebel, Simon & Eklund,
# "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"
# (PHM08), Table 1. Station numbers follow gas-turbine convention: 2 = fan inlet,
# 24 = LPC outlet, 30 = HPC outlet, 50 = LPT outlet. °R = degrees Rankine.
SENSOR_INFO: dict[str, dict[str, str]] = {
    "s1": {"symbol": "T2", "description": "Total temperature at fan inlet", "unit": "°R"},
    "s2": {"symbol": "T24", "description": "Total temperature at LPC outlet", "unit": "°R"},
    "s3": {"symbol": "T30", "description": "Total temperature at HPC outlet", "unit": "°R"},
    "s4": {"symbol": "T50", "description": "Total temperature at LPT outlet", "unit": "°R"},
    "s5": {"symbol": "P2", "description": "Pressure at fan inlet", "unit": "psia"},
    "s6": {"symbol": "P15", "description": "Total pressure in bypass duct", "unit": "psia"},
    "s7": {"symbol": "P30", "description": "Total pressure at HPC outlet", "unit": "psia"},
    "s8": {"symbol": "Nf", "description": "Physical fan speed", "unit": "rpm"},
    "s9": {"symbol": "Nc", "description": "Physical core speed", "unit": "rpm"},
    "s10": {"symbol": "epr", "description": "Engine pressure ratio (P50/P2)", "unit": "–"},
    "s11": {"symbol": "Ps30", "description": "Static pressure at HPC outlet", "unit": "psia"},
    "s12": {"symbol": "phi", "description": "Fuel flow / Ps30", "unit": "pps/psi"},
    "s13": {"symbol": "NRf", "description": "Corrected fan speed", "unit": "rpm"},
    "s14": {"symbol": "NRc", "description": "Corrected core speed", "unit": "rpm"},
    "s15": {"symbol": "BPR", "description": "Bypass ratio", "unit": "–"},
    "s16": {"symbol": "farB", "description": "Burner fuel–air ratio", "unit": "–"},
    "s17": {"symbol": "htBleed", "description": "Bleed enthalpy", "unit": "–"},
    "s18": {"symbol": "Nf_dmd", "description": "Demanded fan speed", "unit": "rpm"},
    "s19": {"symbol": "PCNfR_dmd", "description": "Demanded corrected fan speed", "unit": "rpm"},
    "s20": {"symbol": "W31", "description": "HPT coolant bleed", "unit": "lbm/s"},
    "s21": {"symbol": "W32", "description": "LPT coolant bleed", "unit": "lbm/s"},
}

SETTING_INFO: dict[str, dict[str, str]] = {
    "setting_1": {"symbol": "Alt", "description": "Altitude", "unit": "ft"},
    "setting_2": {"symbol": "Mach", "description": "Mach number", "unit": "–"},
    "setting_3": {"symbol": "TRA", "description": "Throttle resolver angle", "unit": "°"},
}

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
