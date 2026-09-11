import { useEffect, useState } from "react";
import { api, DatasetInfo } from "./api";
import { FleetPanel } from "./components/FleetPanel";
import { ModelsPanel } from "./components/ModelsPanel";

type Tab = "fleet" | "models";

const TABS: { id: Tab; label: string }[] = [
  { id: "fleet", label: "Fleet Monitor" },
  { id: "models", label: "Model Evaluation" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("fleet");
  const [datasets, setDatasets] = useState<DatasetInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .datasets()
      .then(setDatasets)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>AeroPulse</h1>
          <span className="tagline">
            Turbofan remaining-useful-life prediction · NASA C-MAPSS
          </span>
        </div>
        <span className="muted small">
          {datasets ? `${datasets.length} trained subset${datasets.length === 1 ? "" : "s"}` : "…"}
        </span>
      </header>

      <nav className="tab-bar">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={t.id === tab ? "tab active" : "tab"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {error && (
          <div className="card error">
            Could not reach the API: {error}. Is the service running and are models trained?
          </div>
        )}
        {datasets && datasets.length === 0 && (
          <div className="card error">
            The API is up but has no trained models. Run <code>python scripts/train.py --dataset FD001</code>.
          </div>
        )}
        {datasets && datasets.length > 0 && (
          <>
            <div style={{ display: tab === "fleet" ? "block" : "none" }}>
              <FleetPanel datasets={datasets} />
            </div>
            <div style={{ display: tab === "models" ? "block" : "none" }}>
              <ModelsPanel datasets={datasets} />
            </div>
          </>
        )}
      </main>

      <footer className="app-footer">
        Replays NASA's <em>simulated</em> C-MAPSS engine trajectories — not live aircraft telemetry.
      </footer>
    </div>
  );
}
