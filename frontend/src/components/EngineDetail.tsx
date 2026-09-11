import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, EngineHistory, ModelName, MODEL_LABELS } from "../api";

const TRACE_COLORS = ["#4f46e5", "#0891b2", "#db2777", "#ea580c", "#65a30d", "#7c3aed"];

export function EngineDetail({
  dataset,
  unit,
  tick,
  model,
  cap,
}: {
  dataset: string;
  unit: number;
  tick: number;
  model: ModelName;
  cap: number;
}) {
  const [hist, setHist] = useState<EngineHistory | null>(null);
  const [sensor, setSensor] = useState<string[]>([]);

  useEffect(() => {
    api.engine(dataset, unit).then((h) => {
      setHist(h);
      if (sensor.length === 0) setSensor(h.sensors.slice(0, 4));
    });
  }, [dataset, unit, tick, model]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!hist) return <div className="card">Loading engine #{unit}…</div>;

  const points = hist.points.map((p) => ({ ...p, rul_true_capped: Math.min(p.rul_true, cap) }));

  return (
    <div className="detail-grid">
      <div className="card">
        <h3>
          Engine #{unit} · predicted vs true RUL <span className="muted small">({MODEL_LABELS[hist.model]})</span>
        </h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="cycle" label={{ value: "cycle", position: "insideBottomRight", offset: -4 }} />
            <YAxis domain={[0, cap + 10]} />
            <Tooltip />
            <Legend />
            <ReferenceLine y={20} stroke="#dc2626" strokeDasharray="4 4" label="critical" />
            <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="4 4" label="warning" />
            <Line type="monotone" dataKey="rul_true_capped" name="true RUL (capped)" stroke="#64748b" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="rul_pred" name="predicted RUL" stroke="#4f46e5" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
        <p className="muted small">
          Truth is known here because this is a NASA test trajectory with a published RUL at truncation. Points above
          the grey line are <em>late</em> predictions — the costly direction.
        </p>
      </div>

      <div className="card">
        <h3>
          Normalised sensor trace{" "}
          <span className="muted small">(per-regime standardised · {hist.sensors.length} kept sensors)</span>
        </h3>
        <div className="chip-row">
          {hist.sensors.map((s) => (
            <button
              key={s}
              className={sensor.includes(s) ? "chip active" : "chip"}
              onClick={() =>
                setSensor((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s].slice(-6)))
              }
            >
              {s}
            </button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={hist.sensor_trace} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="cycle" />
            <YAxis />
            <Tooltip />
            <Legend />
            {sensor.map((s, i) => (
              <Line key={s} type="monotone" dataKey={s} stroke={TRACE_COLORS[i % TRACE_COLORS.length]} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
