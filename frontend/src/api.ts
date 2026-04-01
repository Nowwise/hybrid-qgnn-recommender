const BASE = "/api";

export type DatasetStatus = {
  data_dir: string;
  exists: boolean;
  train_txt: boolean;
  test_txt: boolean;
};

export type JobStep = {
  id: string;
  label: string;
  status: "pending" | "running" | "done" | "error";
};

export type JobPublic = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  phase: string;
  detail: string | null;
  progress_pct: number;
  steps: JobStep[];
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
  quick: Record<string, unknown>;
  full: Record<string, unknown>;
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
  return fetchJson<DatasetStatus>("/datasets/status");
}

export function getExperimentPresets() {
  return fetchJson<ExperimentPresets>("/experiments/presets");
}

/** Start a run: pass preset + any config fields (snake_case) to merge on the server. */
export function startExperiment(body: Record<string, unknown>) {
  return fetchJson<JobPublic>("/experiments/runs", {
    method: "POST",
    body: JSON.stringify(body),
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
