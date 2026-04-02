const BASE = "/api";

export type DatasetStatus = {
  data_dir: string;
  exists: boolean;
  train_txt: boolean;
  test_txt: boolean;
};

export type DatasetsOverview = {
  datasets: DatasetStatus[];
};

export type JobStep = {
  id: string;
  label: string;
  status: "pending" | "running" | "done" | "error";
};

export type JobActivity = {
  model?: string | null;
  split?: string | null;
  phase?: string | null;
  epoch?: number | null;
  batch?: number | null;
  total_batches?: number | null;
  loss?: number | null;
  p_quantum?: number | null;
};

export type JobEvent = {
  ts: string;
  kind: string;
  message: string;
};

export type JobPublic = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  phase: string;
  detail: string | null;
  progress_pct: number;
  steps: JobStep[];
  activity: JobActivity | null;
  events: JobEvent[];
  created_at: string;
  updated_at: string;
  error: string | null;
  result: Record<string, unknown> | null;
};

export type RunSummary = {
  run_id: string;
  path: string;
  has_metrics: boolean;
  has_summary: boolean;
  best_lightgcn_auc: number | null;
  best_hybrid_auc: number | null;
};

export type ExperimentPresets = {
  lightweight: Record<string, unknown>;
  notebook: Record<string, unknown>;
  custom: Record<string, unknown>;
  quick?: Record<string, unknown>;
  full?: Record<string, unknown>;
};

export type QuantumSimulatorStatus = { available: boolean; error?: string };

export type ExperimentDeviceOverview = {
  torch_version: string;
  cuda_available: boolean;
  cuda_device_count: number;
  cuda_version: string | null;
  devices: { id: string; kind: string; name: string }[];
  quantum_simulators: Record<string, QuantumSimulatorStatus>;
};

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json() as Promise<T>;
}

export function getHealth() {
  return fetchJson<{ status: string; service: string }>("/health");
}

export function getDatasetStatus() {
  return fetchJson<DatasetsOverview>("/datasets/status");
}

export function getExperimentPresets() {
  return fetchJson<ExperimentPresets>("/experiments/presets");
}

export function getExperimentDevice() {
  return fetchJson<ExperimentDeviceOverview>("/experiments/device");
}

/** Start a run: pass preset + any config fields (snake_case) to merge on the server. */
export function startExperiment(body: Record<string, unknown>) {
  return fetchJson<JobPublic>("/experiments/runs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function cancelExperimentJob(id: string) {
  return fetchJson<JobPublic>(`/experiments/jobs/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
    body: "{}",
  });
}

export function listJobs() {
  return fetchJson<JobPublic[]>("/experiments/jobs");
}

export function getJob(id: string) {
  return fetchJson<JobPublic>(`/experiments/jobs/${id}`);
}

export function listHistory() {
  return fetchJson<RunSummary[]>("/experiments/history");
}

export function getComparative(runId: string) {
  return fetchJson<Record<string, string | number | null>[]>(
    `/experiments/history/${encodeURIComponent(runId)}/comparative`
  );
}
