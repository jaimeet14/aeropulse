# Architecture

## Training pipeline

```mermaid
flowchart TD
    A[train_FD00x.txt] --> B[add_rul: rul_true = max_cycle - cycle<br/>rul = min(rul_true, 125)]
    B --> C[Preprocessor.fit]
    C --> C1[k-means on setting_1..3<br/>k = 1 for FD001/FD003, 6 for FD002/FD004]
    C --> C2[per-regime mean / std for all 21 sensors]
    C --> C3[flat-sensor rule:<br/>std < 1e-3 OR nunique ≤ 2 in every regime]
    C1 & C2 & C3 --> D{fitted preprocessor}
    D -->|engineer| E[per kept sensor:<br/>value · rolling mean · rolling std · rolling slope<br/>+ cycle + regime]
    D -->|windows| F[(n, 30, n_sensors) tensors<br/>label = rul at window end]
    E --> G[Ridge] & H[LightGBM]
    F --> I[LSTM<br/>10% of units held out · early stopping]
    G & H & I --> J[evaluate on test_FD00x last cycle vs RUL_FD00x]
    J --> K[artifacts/FD00x/]
```

All rolling statistics are computed with `groupby("unit").rolling(window, min_periods=…)`, i.e. they only ever see *earlier* cycles of the *same* engine. `tests/test_features.py::test_engineered_features_do_not_look_forward` perturbs every cycle after *t* and asserts that features at or before *t* are unchanged.

## Request flow: `POST /predict`

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant P as Preprocessor
    participant M as Model

    C->>API: POST /predict {dataset, model, readings[]}
    API->>API: Pydantic: 3 settings, 21 sensors, strictly increasing cycles
    API->>API: readings -> DataFrame (unit = 1)
    alt model = lgbm / linear
        API->>P: engineer(df)  (rolling features)
        P-->>API: features for every cycle
        API->>M: predict(last row)
    else model = lstm
        API->>P: windows(df, last_only=True)  (pads if < 30 cycles)
        P-->>API: (1, 30, n_sensors)
        API->>M: predict(window)
    end
    M-->>API: RUL in [0, 125]
    API-->>C: {rul, urgency, n_readings, window, note?}
```

## Request flow: fleet simulation

```mermaid
sequenceDiagram
    participant FE as Dashboard
    participant API as FastAPI
    participant F as FleetSimulator

    FE->>API: GET /fleet?dataset=FD001&model=lgbm
    API->>API: first call: load test_FD001 + RUL_FD001 (download if missing)
    API->>F: FleetSimulator(bundle, subset)
    F->>F: pick 12 test units stratified by true RUL-at-truncation
    F->>F: reveal a staggered prefix of each trajectory, predict
    API-->>FE: engines[] sorted by predicted RUL, summary{critical,warning,healthy}

    loop every 1.2 s while "Run" is on
        FE->>API: POST /fleet/tick?steps=1
        API->>F: tick(): cursor += 1 per engine (or return from shop at end of data)
        F->>F: bundle.predict(all visible prefixes)
        API-->>FE: updated engines[] + summary
    end

    FE->>API: GET /fleet/{unit}
    API-->>FE: history[{cycle, rul_pred, rul_true, urgency}], normalised sensor trace
```

The true RUL at any revealed cycle is `rul_at_truncation + (last_recorded_cycle - cycle)`, which is why the dashboard can draw the grey truth line for test engines even though the test files are truncated.

## Deployment

```mermaid
flowchart LR
    subgraph docker compose
        API[api :8010<br/>python:3.12-slim + CPU torch<br/>artifacts/ baked in]
        FE[frontend :5173<br/>nginx serving Vite build]
    end
    FE -->|fetch| API
    API -->|first /fleet call| NASA[(phm-datasets S3 mirror)]
    API --- V[(./data volume)]
```

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AEROPREDICT_DATA_DIR` | `data/raw` | where the NASA text files live |
| `AEROPREDICT_ARTIFACT_DIR` | `artifacts` | trained bundles |
| `AEROPREDICT_AUTO_DOWNLOAD` | `1` | download C-MAPSS on first fleet request |
| `AEROPREDICT_RUL_CAP` | `125` | piecewise-linear cap (training only) |
| `AEROPREDICT_WINDOW` | `30` | rolling / sequence window (training only) |
| `CORS_ORIGINS` | `http://localhost:5173` | comma-separated allowed origins |
| `VITE_API_URL` (frontend) | `http://localhost:8010` | API base URL baked into the build |
