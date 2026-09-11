export const API_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8010";

export type ModelName = "linear" | "lgbm" | "lstm";
export type Urgency = "healthy" | "warning" | "critical";

export interface Metrics {
  rmse: number;
  mae: number;
  nasa_score: number;
  late_fraction: number;
  urgency_accuracy: number;
  n: number;
}

export interface ModelResult {
  test: Metrics;
  test_uncapped_truth: Metrics;
  top_features?: Record<string, number>;
  epochs_trained?: number;
  best_val_rmse?: number;
  device?: string;
}

export interface DatasetInfo {
  dataset: string;
  n_regimes: number;
  sensors_kept: string[];
  sensors_dropped: string[];
  n_train_units: number;
  n_test_units: number;
  rul_cap: number;
  window: number;
  models: Record<string, ModelResult>;
}

export interface EngineStatus {
  unit: number;
  cycle: number;
  cycles_available: number;
  rul_pred: number;
  rul_true: number | null;
  urgency: Urgency;
  regime: number;
  at_end: boolean;
  shop_visits: number;
}

export interface FleetResponse {
  dataset: string;
  model: ModelName;
  tick: number;
  engines: EngineStatus[];
  summary: Record<Urgency, number>;
}

export interface HistoryPoint {
  cycle: number;
  rul_pred: number;
  rul_true: number;
  urgency: Urgency;
}

export interface EngineHistory {
  unit: number;
  dataset: string;
  model: ModelName;
  points: HistoryPoint[];
  sensors: string[];
  sensor_trace: Record<string, number>[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; version: string; datasets: string[] }>("/health"),
  datasets: () => request<DatasetInfo[]>("/datasets"),
  fleet: (dataset: string, model?: ModelName) =>
    request<FleetResponse>(`/fleet?dataset=${dataset}${model ? `&model=${model}` : ""}`),
  tick: (dataset: string, steps = 1, model?: ModelName) =>
    request<FleetResponse>(
      `/fleet/tick?dataset=${dataset}&steps=${steps}${model ? `&model=${model}` : ""}`,
      { method: "POST" }
    ),
  reset: (dataset: string) =>
    request<FleetResponse>(`/fleet/reset?dataset=${dataset}`, { method: "POST" }),
  engine: (dataset: string, unit: number) =>
    request<EngineHistory>(`/fleet/${unit}?dataset=${dataset}`),
};

export const MODEL_LABELS: Record<ModelName, string> = {
  linear: "Ridge (linear)",
  lgbm: "LightGBM",
  lstm: "LSTM",
};

export const URGENCY_COLORS: Record<Urgency, string> = {
  healthy: "#16a34a",
  warning: "#f59e0b",
  critical: "#dc2626",
};
