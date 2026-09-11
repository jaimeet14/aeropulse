import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { DatasetInfo, ModelName, MODEL_LABELS, featureLabel, sensorTitle } from "../api";

export function ModelsPanel({ datasets }: { datasets: DatasetInfo[] }) {
  const [uncapped, setUncapped] = useState(false);
  const key = uncapped ? "test_uncapped_truth" : "test";

  return (
    <div>
      <div className="card">
        <h3>Test-set results per sub-dataset</h3>
        <p className="muted small">
          One prediction per test engine at its last observed cycle, scored against <code>RUL_FD00x.txt</code>. RMSE is
          symmetric; the <strong>NASA score</strong> (PHM08) penalises late predictions exponentially harder than early
          ones and is the number that matters operationally. <strong>Late %</strong> is the share of engines we
          over-estimated. Lower is better for all three.
        </p>
        <label className="inline">
          <input type="checkbox" checked={uncapped} onChange={(e) => setUncapped(e.target.checked)} />
          score against <em>uncapped</em> truth (shows the cost of the piecewise-linear RUL target)
        </label>
        <div className="table-scroll">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Subset</th>
                <th>Regimes</th>
                <th>Engines (train / test)</th>
                <th>Sensors dropped</th>
                <th>Model</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>NASA score</th>
                <th>Late %</th>
                <th>Urgency acc.</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) =>
                (Object.keys(d.models) as ModelName[]).map((m, i) => {
                  const r = d.models[m][key];
                  const best = bestModel(d, key);
                  return (
                    <tr key={`${d.dataset}-${m}`} className={m === best ? "best" : ""}>
                      {i === 0 && (
                        <>
                          <td rowSpan={Object.keys(d.models).length}>
                            <strong>{d.dataset}</strong>
                          </td>
                          <td rowSpan={Object.keys(d.models).length}>{d.n_regimes}</td>
                          <td rowSpan={Object.keys(d.models).length}>
                            {d.n_train_units} / {d.n_test_units}
                          </td>
                          <td rowSpan={Object.keys(d.models).length} className="small">
                            {d.sensors_dropped.map((s, j) => (
                              <span key={s} title={sensorTitle(s, d.sensor_info)}>
                                {j > 0 && ", "}
                                {s} <span className="muted">{d.sensor_info[s]?.symbol}</span>
                              </span>
                            ))}
                          </td>
                        </>
                      )}
                      <td>{MODEL_LABELS[m]}</td>
                      <td>{r.rmse.toFixed(2)}</td>
                      <td>{r.mae.toFixed(2)}</td>
                      <td>{r.nasa_score.toFixed(0)}</td>
                      <td>{(100 * r.late_fraction).toFixed(0)}%</td>
                      <td>{(100 * r.urgency_accuracy).toFixed(0)}%</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="detail-grid">
        <div className="card">
          <h3>NASA score by model</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartRows(datasets, key, "nasa_score")}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="dataset" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="linear" name="Ridge" fill="#94a3b8" />
              <Bar dataKey="lgbm" name="LightGBM" fill="#4f46e5" />
              <Bar dataKey="lstm" name="LSTM" fill="#0891b2" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3>RMSE by model</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartRows(datasets, key, "rmse")}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="dataset" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="linear" name="Ridge" fill="#94a3b8" />
              <Bar dataKey="lgbm" name="LightGBM" fill="#4f46e5" />
              <Bar dataKey="lstm" name="LSTM" fill="#0891b2" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {datasets.map((d) =>
        d.models.lgbm?.top_features ? (
          <div className="card" key={d.dataset}>
            <h3>
              {d.dataset} · LightGBM top features <span className="muted small">(split importance)</span>
            </h3>
            <div className="chip-row">
              {Object.entries(d.models.lgbm.top_features).map(([f, v]) => (
                <span className="chip static" key={f} title={`${f}: ${describeFeature(f, d)}`}>
                  {featureLabel(f, d.sensor_info)} <span className="muted">{v}</span>
                </span>
              ))}
            </div>
            <p className="muted small">
              <em>mean</em> / <em>std</em> / <em>slope</em> are computed over the last {d.window} cycles;{" "}
              <em>cycle</em> is the engine's age. Hover a chip for the full sensor description.
            </p>
          </div>
        ) : null
      )}

      <SensorGlossary datasets={datasets} />
    </div>
  );
}

const STAT_WORDS: Record<string, string> = {
  mean: "rolling average of",
  std: "rolling variability of",
  slope: "rolling trend (rate of change) of",
};

function describeFeature(feature: string, d: DatasetInfo): string {
  if (feature === "cycle") return "engine age in cycles";
  if (feature === "regime") return "operating-condition cluster";
  const m = feature.match(/^(s\d+)(?:_(mean|std|slope))?$/);
  if (!m) return feature;
  const [, col, stat] = m;
  const i = d.sensor_info[col];
  const what = i ? `${i.description} (${i.symbol}, ${i.unit})` : col;
  return stat ? `${STAT_WORDS[stat]} ${what}` : what;
}

function SensorGlossary({ datasets }: { datasets: DatasetInfo[] }) {
  // Union of sensor info across datasets (they all describe the same 21 columns).
  const info = Object.assign({}, ...datasets.map((d) => d.sensor_info));
  const cols = Object.keys(info).sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
  return (
    <div className="card">
      <h3>
        Sensor glossary <span className="muted small">(Saxena et al., PHM08 — station 2 = fan inlet, 24 = LPC, 30 = HPC, 50 = LPT)</span>
      </h3>
      <div className="table-scroll">
        <table className="glossary-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Symbol</th>
              <th>Measurement</th>
              <th>Unit</th>
              {datasets.map((d) => (
                <th key={d.dataset}>{d.dataset}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cols.map((c) => (
              <tr key={c}>
                <td>
                  <strong>{c}</strong>
                </td>
                <td>
                  <code>{info[c].symbol}</code>
                </td>
                <td>{info[c].description}</td>
                <td className="muted">{info[c].unit}</td>
                {datasets.map((d) => (
                  <td key={d.dataset} className={d.sensors_dropped.includes(c) ? "dropped" : "kept"}>
                    {d.sensors_dropped.includes(c) ? "dropped" : "kept"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        "dropped" = the sensor is constant (or takes at most two values) in every operating regime of that subset,
        so it carries no information about wear and is excluded before training.
      </p>
    </div>
  );
}

function bestModel(d: DatasetInfo, key: "test" | "test_uncapped_truth"): ModelName {
  let best: ModelName = "lgbm";
  let score = Infinity;
  for (const m of Object.keys(d.models) as ModelName[]) {
    const s = d.models[m][key].nasa_score;
    if (s < score) {
      score = s;
      best = m;
    }
  }
  return best;
}

function chartRows(datasets: DatasetInfo[], key: "test" | "test_uncapped_truth", metric: "nasa_score" | "rmse") {
  return datasets.map((d) => {
    const row: Record<string, number | string> = { dataset: d.dataset };
    for (const m of Object.keys(d.models) as ModelName[]) row[m] = d.models[m][key][metric];
    return row;
  });
}
