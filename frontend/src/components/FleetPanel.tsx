import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  DatasetInfo,
  EngineStatus,
  FleetResponse,
  ModelName,
  MODEL_LABELS,
  URGENCY_COLORS,
} from "../api";
import { EngineDetail } from "./EngineDetail";

const TICK_MS = 1200;

export function FleetPanel({ datasets }: { datasets: DatasetInfo[] }) {
  const [dataset, setDataset] = useState(datasets[0].dataset);
  const [model, setModel] = useState<ModelName>("lgbm");
  const [fleet, setFleet] = useState<FleetResponse | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);

  const info = datasets.find((d) => d.dataset === dataset)!;
  const models = Object.keys(info.models) as ModelName[];

  const load = useCallback(async () => {
    try {
      setFleet(await api.fleet(dataset, model));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [dataset, model]);

  const step = useCallback(
    async (steps = 1) => {
      if (busy.current) return;
      busy.current = true;
      try {
        setFleet(await api.tick(dataset, steps, model));
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        busy.current = false;
      }
    },
    [dataset, model]
  );

  useEffect(() => {
    setSelected(null);
    load();
  }, [load]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => step(1), TICK_MS);
    return () => clearInterval(id);
  }, [running, step]);

  useEffect(() => {
    if (!models.includes(model)) setModel(models[0]);
  }, [dataset]); // eslint-disable-line react-hooks/exhaustive-deps

  const reset = async () => {
    setRunning(false);
    setFleet(await api.reset(dataset));
  };

  return (
    <div>
      <div className="card controls">
        <label>
          Sub-dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.dataset} value={d.dataset}>
                {d.dataset} · {d.n_regimes} regime{d.n_regimes > 1 ? "s" : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          Model
          <select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {MODEL_LABELS[m]}
              </option>
            ))}
          </select>
        </label>
        <div className="button-row">
          <button onClick={() => setRunning((r) => !r)}>{running ? "⏸ Pause" : "▶ Run"}</button>
          <button className="secondary" onClick={() => step(1)} disabled={running}>
            Step +1
          </button>
          <button className="secondary" onClick={() => step(10)} disabled={running}>
            Step +10
          </button>
          <button className="secondary" onClick={reset}>
            Reset
          </button>
        </div>
        <div className="muted small">
          {fleet ? `tick ${fleet.tick} · ${fleet.engines.length} engines in service` : ""}
        </div>
      </div>

      {error && <div className="card error">{error}</div>}

      {fleet && (
        <>
          <div className="tiles">
            {(["critical", "warning", "healthy"] as const).map((u) => (
              <div className="tile" key={u} style={{ borderTopColor: URGENCY_COLORS[u] }}>
                <div className="tile-value">{fleet.summary[u]}</div>
                <div className="tile-label">{u}</div>
                <div className="muted small">
                  {u === "critical" && "≤ 20 cycles left — ground for maintenance"}
                  {u === "warning" && "21–50 cycles left — schedule shop visit"}
                  {u === "healthy" && "> 50 cycles left"}
                </div>
              </div>
            ))}
          </div>

          <div className="card">
            <table className="fleet-table">
              <thead>
                <tr>
                  <th>Engine</th>
                  <th>Cycle</th>
                  <th>Regime</th>
                  <th>Predicted RUL</th>
                  <th>True RUL</th>
                  <th>Error</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {fleet.engines.map((e) => (
                  <EngineRow
                    key={e.unit}
                    e={e}
                    cap={info.rul_cap}
                    selected={selected === e.unit}
                    onSelect={() => setSelected(selected === e.unit ? null : e.unit)}
                  />
                ))}
              </tbody>
            </table>
          </div>

          {selected !== null && (
            <EngineDetail dataset={dataset} unit={selected} tick={fleet.tick} model={fleet.model} cap={info.rul_cap} />
          )}
        </>
      )}
    </div>
  );
}

function EngineRow({
  e,
  cap,
  selected,
  onSelect,
}: {
  e: EngineStatus;
  cap: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const truth = e.rul_true === null ? null : Math.min(e.rul_true, cap);
  const err = truth === null ? null : e.rul_pred - truth;
  return (
    <tr className={selected ? "selected" : ""} onClick={onSelect}>
      <td>
        <strong>#{e.unit}</strong>
        {e.shop_visits > 0 && <span className="muted small"> · {e.shop_visits}× shop</span>}
      </td>
      <td>
        {e.cycle}
        <span className="muted small"> / {e.cycles_available}</span>
      </td>
      <td>{e.regime}</td>
      <td>
        <div className="rul-bar">
          <div
            className="rul-fill"
            style={{ width: `${(100 * e.rul_pred) / cap}%`, background: URGENCY_COLORS[e.urgency] }}
          />
          <span>{e.rul_pred.toFixed(0)}</span>
        </div>
      </td>
      <td>{truth ?? "—"}</td>
      <td className={err !== null && err > 0 ? "late" : "early"}>
        {err === null ? "—" : `${err > 0 ? "+" : ""}${err.toFixed(0)}`}
        {err !== null && err > 0 && <span className="muted small"> late</span>}
      </td>
      <td>
        <span className="badge" style={{ background: URGENCY_COLORS[e.urgency] }}>
          {e.urgency}
        </span>
      </td>
    </tr>
  );
}
