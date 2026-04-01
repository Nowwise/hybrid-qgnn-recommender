const BASE = "/api";

export type DatasetStatus = {
  data_dir: string;
  exists: boolean;
  train_txt: boolean;
  test_txt: boolean;
};

export type JobPublic = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  phase: string;
  detail: string | null;
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

export function startRun(quickDemo: boolean) {
  return fetchJson<JobPublic>("/experiments/runs", {
    method: "POST",
    body: JSON.stringify({ quick_demo: quickDemo }),
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
