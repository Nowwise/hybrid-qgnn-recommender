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
  /** Project-relative run folder; set once training creates artifacts. */
  save_dir?: string | null;
  progress_pct: number;
  steps: JobStep[];
  activity: JobActivity | null;
  events: JobEvent[];
  created_at: string;
  updated_at: string;
  error: string | null;
  result: Record<string, unknown> | null;
};

export type LiveValSnapshot = {
  epoch: number;
  metrics: Record<string, number | null>;
};

export type LiveBestAuc = {
  epoch: number;
  auc: number;
};

export type LiveRankingRow = {
  model: string;
  metric: string;
  value: number | null;
};

export type LiveMetricsPayload = {
  ready: boolean;
  reason?: string;
  val_auc: Record<string, { epoch: number; value: number }[]>;
  val_rmse?: Record<string, { epoch: number; value: number }[]>;
  val_mae?: Record<string, { epoch: number; value: number }[]>;
  train_loss: Record<string, { epoch: number; value: number }[]>;
  p_quantum: { epoch: number; value: number }[];
  row_count: number;
  latest_val?: Record<string, LiveValSnapshot>;
  best_val_auc?: Record<string, LiveBestAuc>;
  ranking_val?: LiveRankingRow[];
  ranking_test?: LiveRankingRow[];
};

export type RunSummary = {
  run_id: string;
  path: string;
  has_metrics: boolean;
  has_summary: boolean;
  experiment_name?: string | null;
  best_lightgcn_auc: number | null;
  best_hybrid_auc: number | null;
  data_dir?: string | null;
  hybrid_backbone?: string | null;
  q?: number | null;
  L?: number | null;
  d?: number | null;
  K?: number | null;
  epochs_lg?: number | null;
  epochs_hyb?: number | null;
  has_hybrid_checkpoint?: boolean;
  n_baseline_checkpoints?: number;
  modified_at?: string | null;
  has_graph_context?: boolean;
};

export type ScorePairsResponse = {
  scores: number[];
  pairs: [number, number][];
  n_pairs: number;
  run_id: string;
  n_users: number;
  n_items: number;
  hybrid_backbone: string;
  graph_context: string;
};

export type ExperimentPresets = {
  lightweight: Record<string, unknown>;
  notebook: Record<string, unknown>;
  large?: Record<string, unknown>;
  extra_large?: Record<string, unknown>;
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
  return fetchJson<JobPublic>(`/experiments/jobs/${encodeURIComponent(id)}`);
}

export function getJobLiveMetrics(jobId: string) {
  return fetchJson<LiveMetricsPayload>(`/experiments/jobs/${encodeURIComponent(jobId)}/live-metrics`);
}

/** PNG saved under run plots/ — append cache buster for refresh while training. */
export function jobPlotUrl(jobId: string, filename: string, cacheBust?: number) {
  const qs = cacheBust != null ? `?t=${cacheBust}` : "";
  return `${BASE}/experiments/jobs/${encodeURIComponent(jobId)}/plots/${encodeURIComponent(filename)}${qs}`;
}

export function listHistory() {
  return fetchJson<RunSummary[]>("/experiments/history");
}

/** Full merged config written at train start (ExperimentConfig + device info). */
export function getRunConfig(runId: string) {
  return fetchJson<Record<string, unknown>>(
    `/experiments/history/${encodeURIComponent(runId)}/config`,
  );
}

export function scorePairsOnRun(
  runId: string,
  body: { pairs_text: string; micro_bs?: number },
) {
  return fetchJson<ScorePairsResponse>(`/experiments/history/${encodeURIComponent(runId)}/score-pairs`, {
    method: "POST",
    body: JSON.stringify({ pairs_text: body.pairs_text, micro_bs: body.micro_bs ?? 256 }),
  });
}

export function deleteHistoryRun(runId: string) {
  return fetchJson<{ ok: boolean; run_id: string }>(
    `/experiments/history/${encodeURIComponent(runId)}`,
    { method: "DELETE" },
  );
}

export type ClearHistoryResponse = {
  removed_run_children: number;
  removed_job_records: number;
};

/** Deletes all run folders under ``runs/`` and clears the API job queue (memory). Fails with 409 if a job is active. */
export function clearExperimentHistory() {
  return fetchJson<ClearHistoryResponse>("/experiments/history/clear", {
    method: "POST",
    body: "{}",
  });
}

export function getComparative(runId: string) {
  return fetchJson<Record<string, string | number | null>[]>(
    `/experiments/history/${encodeURIComponent(runId)}/comparative`
  );
}

export type RunSummaryTextPayload = {
  text: string | null;
  relative_path: string;
};

export function getRunSummaryText(runId: string) {
  return fetchJson<RunSummaryTextPayload>(`/experiments/history/${encodeURIComponent(runId)}/summary`);
}

export type RunPhaseTimingsPayload = {
  phases: unknown;
  relative_path: string;
};

export function getRunPhaseTimings(runId: string) {
  return fetchJson<RunPhaseTimingsPayload>(`/experiments/history/${encodeURIComponent(runId)}/phase-timings`);
}

/** PNG under runs/{runId}/plots/ (same allowlist as job plots). */
export function historyPlotUrl(runId: string, filename: string, cacheBust?: number) {
  const qs = cacheBust != null ? `?t=${cacheBust}` : "";
  return `${BASE}/experiments/history/${encodeURIComponent(runId)}/plots/${encodeURIComponent(filename)}${qs}`;
}

export type HistoryDownloadAsset = "metrics.csv" | "summary.txt" | "run_config.json" | "full_model_comparative.csv";

export function historyDownloadUrl(runId: string, asset: HistoryDownloadAsset) {
  return `${BASE}/experiments/history/${encodeURIComponent(runId)}/download/${asset}`;
}
