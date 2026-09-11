# Modelling notes

The decisions below are the ones an interviewer is likely to poke at. Each one has a
reason and, where possible, a test or a number behind it.

## 1. The dataset

NASA C-MAPSS (Saxena & Goebel, 2008) simulates a fleet of turbofan engines with a
thermodynamic model, injects a fault (HPC degradation, and for FD003/FD004 also fan
degradation), adds sensor noise, and runs every engine to failure.

| Subset | Train engines | Test engines | Operating regimes | Fault modes |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 249 | 248 | 6 | 2 |

Each row is one flight cycle: `unit, cycle, setting_1..3, s1..s21`. Training
trajectories run to failure; test trajectories are cut off at a random point and
`RUL_FD00x.txt` gives the remaining cycles at that point.

### What the columns physically are

From Table 1 of Saxena et al. (PHM08). Station numbers follow gas-turbine
convention: 2 = fan inlet, 24 = LPC outlet, 30 = HPC outlet, 50 = LPT outlet.
The mapping lives in `aeropredict/config.py` (`SENSOR_INFO`) and is served by
`GET /sensors`, so the dashboard can label every trace and feature.

| Column | Symbol | Measurement | Unit |
| --- | --- | --- | --- |
| `s1` | T2 | Total temperature at fan inlet | °R |
| `s2` | T24 | Total temperature at LPC outlet | °R |
| `s3` | T30 | Total temperature at HPC outlet | °R |
| `s4` | T50 | Total temperature at LPT outlet | °R |
| `s5` | P2 | Pressure at fan inlet | psia |
| `s6` | P15 | Total pressure in bypass duct | psia |
| `s7` | P30 | Total pressure at HPC outlet | psia |
| `s8` | Nf | Physical fan speed | rpm |
| `s9` | Nc | Physical core speed | rpm |
| `s10` | epr | Engine pressure ratio (P50/P2) | – |
| `s11` | Ps30 | Static pressure at HPC outlet | psia |
| `s12` | phi | Fuel flow / Ps30 | pps/psi |
| `s13` | NRf | Corrected fan speed | rpm |
| `s14` | NRc | Corrected core speed | rpm |
| `s15` | BPR | Bypass ratio | – |
| `s16` | farB | Burner fuel–air ratio | – |
| `s17` | htBleed | Bleed enthalpy | – |
| `s18` | Nf_dmd | Demanded fan speed | rpm |
| `s19` | PCNfR_dmd | Demanded corrected fan speed | rpm |
| `s20` | W31 | HPT coolant bleed | lbm/s |
| `s21` | W32 | LPT coolant bleed | lbm/s |

| Setting | Symbol | Meaning | Unit |
| --- | --- | --- | --- |
| `setting_1` | Alt | Altitude | ft |
| `setting_2` | Mach | Mach number | – |
| `setting_3` | TRA | Throttle resolver angle | ° |

The degradation-sensitive channels — the ones LightGBM ranks highest — are the
core-speed sensors (`s9` Nc, `s14` NRc), the HPC-outlet pressures (`s7` P30,
`s11` Ps30), and the LPT-outlet temperature (`s4` T50): exactly where you would
expect an HPC fault to show up.

## 2. Piecewise-linear RUL target (cap = 125)

The naive label is `RUL = cycles_to_failure`. But early in life nothing is degrading:
the sensors of an engine with 250 cycles left look identical to one with 150 left.
Asking a model to tell them apart teaches it noise. The standard fix — used by
essentially every C-MAPSS paper since Heimes (2008) — is `RUL = min(true, 125)`.

Consequences we keep visible:

- Predictions are clipped to `[0, 125]`; the API can never say "200 cycles left".
- Test truth is capped the same way for the headline metrics. `metrics.json` and the
  dashboard also carry scores against the *uncapped* truth so the flattering effect
  of the cap is measurable (FD001 LightGBM: RMSE 11.98 capped vs 13.26 uncapped).

## 3. Operating regimes

FD002/FD004 fly six distinct conditions. `setting_1..3` (altitude, Mach, throttle
resolver angle) fall into six tight clusters, so k-means with k = 6 recovers them
exactly. Every sensor is then standardised with the mean/std *of its regime*.

Why not just add the settings as features and let the model sort it out? Because the
regime offset on a sensor (tens to hundreds of units) is far larger than the
degradation trend (single units over a whole life). Trees can partition on the
setting, but the rolling features — mean, std, slope — would be computed across
regime switches and be dominated by them, and the LSTM would need to learn six
separate baselines. Normalising first makes "distance from healthy, in this regime"
the quantity every model sees.

`tests/test_features.py::test_normalisation_is_per_regime` asserts mean 0 / std 1
within every regime on the synthetic two-regime fixture.

## 4. Flat-sensor pruning

Known result: on FD001, `s1, s5, s6, s10, s16, s18, s19` are constant. Rather than
hard-code that, `Preprocessor.fit` drops a sensor if, in *every* regime, it has
`std < 1e-3` **or** at most two distinct values. The second clause exists because
`s6` is not strictly constant — it flips between 21.60 and 21.61 — but two-valued
noise is not a signal. The rule reproduces the canonical FD001 list exactly, keeps `s6` on FD002–FD004
and `s10` on FD003/FD004, where they genuinely vary, and drops `s10` on FD002.

## 5. Features for the tabular models

Per kept sensor, over a 30-cycle backward window within the same unit:

| Feature | Why |
| --- | --- |
| current normalised value | instantaneous health |
| rolling mean | denoised level |
| rolling std | noise / instability grows near failure on some channels |
| rolling slope (least squares vs cycle) | degradation *rate* — the feature that most directly encodes "how fast is this getting worse" |

plus `cycle` (age is legitimately known in service) and `regime`. The slope is
computed from rolling sums (`n·Σty − Σt·Σy` over `n·Σt² − (Σt)²`) so it costs O(n)
instead of a polyfit per row.

LightGBM's top split features on FD001 are `cycle`, then rolling means of `s14`,
`s9`, `s3`, `s7`, `s13`, `s4` and the std of `s9`/`s14` — the HPC/HPT-related
channels the C-MAPSS documentation identifies as the degradation-sensitive ones.

## 6. Models

| Model | Input | Role |
| --- | --- | --- |
| Ridge | engineered features | linear sanity floor |
| LightGBM (600 trees, lr 0.03, 31 leaves, subsampling) | engineered features | strong tabular baseline |
| LSTM (2 × 64, dropout 0.2, MLP head) | raw 30-cycle windows of normalised sensors | sequence model that has to learn the trend itself |

LSTM training details that matter for honesty:

- Target scaled to `[0, 1]`, MSE loss, Adam 1e-3, gradient clipping at 1.0,
  `ReduceLROnPlateau`, batch 256, up to 40 epochs.
- **Validation split is by engine**, not by row. A random row split puts cycle *t*
  of an engine in train and *t+1* in validation and makes the validation loss
  meaningless.
- Early stopping (patience 6) restores the best-validation weights.
- Seeded; CPU training takes 45–90 s per subset.

## 7. Metrics

**RMSE** is reported because everyone does. **The PHM08 score** is reported because it
is what the problem actually costs:

```
d = predicted − true
s = Σ (exp(−d/13) − 1)   for d < 0   (early: mild)
  + Σ (exp( d/10) − 1)   for d ≥ 0   (late:  harsh)
```

A prediction 20 cycles late costs `e² − 1 ≈ 6.4`; 20 cycles early costs
`e^(20/13) − 1 ≈ 3.7`; 40 cycles late costs 53.6. The score is a sum, so it also
scales with the number of test engines (259 for FD002 vs 100 for FD001) — compare
across subsets with that in mind.

Also reported:

- **Late fraction** — share of engines whose life we over-estimated. ~50% for every
  model, because MSE has no preference. A pinball/asymmetric loss is the obvious way
  to buy NASA score with RMSE; it is deliberately *not* done here so the comparison
  stays like-for-like.
- **Urgency accuracy** — the classification variant: `critical ≤ 20`, `warning ≤ 50`,
  else `healthy`. This is what a maintenance planner actually consumes.

## 8. What would move the numbers

In rough order of expected payoff:

1. **Asymmetric training loss** (e.g. weight positive residuals more) — directly
   targets the NASA score.
2. **Ensembling LightGBM + LSTM** — their errors are only partly correlated.
3. **Longer windows / multi-scale rolling features** (e.g. 10, 30, 60 cycles).
4. **A 1-D CNN or small Transformer** in place of the LSTM — the literature's best
   sequence results on FD002/FD004 come from attention models.
5. **Regime-aware sequence input** — feed the regime id (one-hot) alongside the
   normalised sensors so the LSTM can condition on flight phase transitions.

None of these were done, so the table in the README is the un-tuned first result.
