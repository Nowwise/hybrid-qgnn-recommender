import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  getExperimentDevice,
  type DatasetStatus,
  type ExperimentDeviceOverview,
  type ExperimentPresets,
  type JobActivity,
  type JobPublic,
} from "../api";
import { LiveTrainingMonitor } from "./LiveTrainingMonitor";

const INT_KEYS = new Set([
  "max_users",
  "max_pos_per_user",
  "neg_per_pos",
  "epochs_lg",
  "epochs_hyb",
  "batch_size",
  "micro_bs",
  "d",
  "K",
  "q",
  "L",
  "eval_every",
  "seed",
  "ranking_max_users",
  "ranking_negatives",
  "early_stopping_patience",
]);

const FLOAT_KEYS = new Set([
  "val_ratio",
  "lr",
  "lightgcn_lr",
  "hybrid_lr",
  "wd",
  "hybrid_lr_mult",
  "p_quantum_start",
  "p_quantum_end",
  "early_stopping_min_delta",
]);

const BOOL_KEYS = new Set([
  "eval_ranking",
  "eval_test_ranking",
  "eval_hybrid_ablation",
  "log_phase_timings",
  "live_plots",
  "early_stopping",
  "quantum_entangle",
  "train_baseline_lightgcn",
  "train_baseline_ultragcn",
  "train_baseline_sgl",
  "train_baseline_ncl",
  "train_baseline_xsimgcl",
]);

/** PennyLane device names commonly used with this project (server resolves / falls back at run time). */
const PENNYLANE_BACKENDS: string[] = [
  "lightning.qubit",
  "lightning.gpu",
  "default.qubit",
  "default.mixed",
];

type ComputeModeId = "auto" | "cpu" | "gpu";

const CUSTOM_DATA_DIR = "__custom__";

/** User-facing copy; second line is the API/config key for power users and docs. */
const FIELD_META: Record<string, { label: string; param: string }> = {
  experiment_name: {
    label: "Experiment display name",
    param: "experiment_name",
  },
  data_dir: {
    label: "Dataset folder",
    param: "data_dir",
  },
  save_dir: {
    label: "Output folder (relative to project)",
    param: "save_dir",
  },
  seed: {
    label: "Random seed",
    param: "seed",
  },
  max_users: {
    label: "Max users to include",
    param: "max_users",
  },
  max_pos_per_user: {
    label: "Max training positives per user",
    param: "max_pos_per_user",
  },
  neg_per_pos: {
    label: "Negative samples per positive (training pairs)",
    param: "neg_per_pos",
  },
  val_ratio: {
    label: "Share of data held out for validation",
    param: "val_ratio",
  },
  batch_size: {
    label: "Batch size (users × items samples per step)",
    param: "batch_size",
  },
  micro_bs: {
    label: "Micro-batch size (quantum forward, smaller = less memory)",
    param: "micro_bs",
  },
  lr: {
    label: "Base learning rate",
    param: "lr",
  },
  wd: {
    label: "Weight decay (L2 regularization)",
    param: "wd",
  },
  eval_every: {
    label: "Validate every N epochs",
    param: "eval_every",
  },
  d: {
    label: "Embedding size (vector length per user/item)",
    param: "d",
  },
  K: {
    label: "Graph conv layers (how many hops on the graph)",
    param: "K",
  },
  epochs_lg: {
    label: "Graph baseline epochs (per enabled model)",
    param: "epochs_lg",
  },
  lightgcn_lr: {
    label: "Graph baseline LR (optional, else base LR)",
    param: "lightgcn_lr",
  },
  q: {
    label: "Number of qubits (quantum width)",
    param: "q",
  },
  L: {
    label: "Quantum layer depth (stacked blocks)",
    param: "L",
  },
  backend: {
    label: "Quantum simulator device (PennyLane)",
    param: "backend",
  },
  epochs_hyb: {
    label: "Hybrid model epochs",
    param: "epochs_hyb",
  },
  hybrid_lr: {
    label: "Hybrid head learning rate (optional, else scaled base LR)",
    param: "hybrid_lr",
  },
  hybrid_lr_mult: {
    label: "Hybrid LR multiplier vs base (if hybrid LR empty)",
    param: "hybrid_lr_mult",
  },
  p_quantum_start: {
    label: "Quantum blend at start (0–1, how much quantum early on)",
    param: "p_quantum_start",
  },
  p_quantum_end: {
    label: "Quantum blend at end (usually 1 = full quantum)",
    param: "p_quantum_end",
  },
  eval_ranking: {
    label: "Rank top-K metrics after training",
    param: "eval_ranking",
  },
  eval_test_ranking: {
    label: "Also evaluate ranking on test split",
    param: "eval_test_ranking",
  },
  eval_hybrid_ablation: {
    label: "Run hybrid ablation (classical head only)",
    param: "eval_hybrid_ablation",
  },
  log_phase_timings: {
    label: "Save wall-clock time per phase",
    param: "log_phase_timings",
  },
  live_plots: {
    label: "Live plots (metrics + PNG each epoch)",
    param: "live_plots",
  },
  ranking_max_users: {
    label: "Users sampled for ranking evaluation",
    param: "ranking_max_users",
  },
  ranking_negatives: {
    label: "Random negatives per ranking query (must be ≥ largest K)",
    param: "ranking_negatives",
  },
  ranking_ks: {
    label: "Top-K cutoffs (comma-separated, e.g. 5, 10, 20, 50)",
    param: "ranking_ks",
  },
  training_loss: {
    label: "Training objective",
    param: "training_loss",
  },
  early_stopping: {
    label: "Early stopping",
    param: "early_stopping",
  },
  early_stopping_monitor: {
    label: "Early stopping metric",
    param: "early_stopping_monitor",
  },
  early_stopping_patience: {
    label: "Patience (epochs without improvement)",
    param: "early_stopping_patience",
  },
  early_stopping_min_delta: {
    label: "Minimum improvement to reset patience",
    param: "early_stopping_min_delta",
  },
  quantum_entangle: {
    label: "Quantum ring entanglement (CNOT between qubits)",
    param: "quantum_entangle",
  },
  train_baseline_lightgcn: {
    label: "Train LightGCN baseline",
    param: "train_baseline_lightgcn",
  },
  train_baseline_ultragcn: {
    label: "Train UltraGCN-style baseline",
    param: "train_baseline_ultragcn",
  },
  train_baseline_sgl: {
    label: "Train SGL-style baseline",
    param: "train_baseline_sgl",
  },
  train_baseline_ncl: {
    label: "Train NCL-style baseline",
    param: "train_baseline_ncl",
  },
  train_baseline_xsimgcl: {
    label: "Train XSimGCL-style baseline",
    param: "train_baseline_xsimgcl",
  },
  hybrid_backbone: {
    label: "Hybrid encoder backbone (checkpoint used to init hybrid graph side)",
    param: "hybrid_backbone",
  },
};

function fieldMeta(fieldKey: string): { label: string; param: string } {
  return FIELD_META[fieldKey] ?? { label: humanizeKey(fieldKey), param: fieldKey };
}

function humanizeKey(fieldKey: string): string {
  return fieldKey
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function FieldLabel({ fieldKey }: { fieldKey: string }) {
  const { label, param } = fieldMeta(fieldKey);
  return (
    <span className="exp-form__label-stack">
      <span className="exp-form__label">{label}</span>
      <span className="exp-form__param mono" title="Name in API requests and saved run_config.json">
        {param}
      </span>
    </span>
  );
}

const COMPUTE_MODE_META = {
  label: "Training hardware (CPU vs GPU and quantum simulator)",
  param: "compute_mode",
} as const;

function ComputeModeLabel() {
  return (
    <span className="exp-form__label-stack">
      <span className="exp-form__label">{COMPUTE_MODE_META.label}</span>
      <span className="exp-form__param mono" title="Name in API requests and saved run_config.json">
        {COMPUTE_MODE_META.param}
      </span>
    </span>
  );
}

const FORM_GROUPS: { title: string; hint?: string; keys: string[] }[] = [
  {
    title: "Data & I/O",
    hint: "Pick a dataset that already has train.txt and test.txt, or choose Custom and type the folder path. If you enter a display name and leave the output folder blank, a new runs/… folder is created automatically and the name appears in history.",
    keys: ["experiment_name", "data_dir", "save_dir", "seed"],
  },
  {
    title: "Sampling",
    hint: "Controls how much of the interaction data is used to build training pairs and validation.",
    keys: ["max_users", "max_pos_per_user", "neg_per_pos", "val_ratio"],
  },
  {
    title: "Training (shared)",
    hint: "Settings shared by both training phases. The base learning rate applies when you leave the LightGCN or hybrid rate empty.",
    keys: ["batch_size", "micro_bs", "lr", "wd", "eval_every"],
  },
  {
    title: "Loss & early stopping",
    hint: "BPR uses pairwise rankings (closer to LightGCN’s SIGIR setup than pointwise BCE). Early stopping applies per graph baseline and again for hybrid training (counter resets after warmup).",
    keys: [
      "training_loss",
      "early_stopping",
      "early_stopping_monitor",
      "early_stopping_patience",
      "early_stopping_min_delta",
      "quantum_entangle",
    ],
  },
  {
    title: "Graph baselines (enable models)",
    hint: "Turn baselines off to skip them in a run. The hybrid backbone is always trained (even if its toggle is off) so the hybrid model can load weights. Pick which encoder initializes HybridQGNN below.",
    keys: [
      "train_baseline_lightgcn",
      "train_baseline_ultragcn",
      "train_baseline_sgl",
      "train_baseline_ncl",
      "train_baseline_xsimgcl",
      "hybrid_backbone",
    ],
  },
  {
    title: "Graph baseline training (shared)",
    hint: "Embedding size, layers, epoch count, and LR apply to every enabled graph baseline (same schedule as before LightGCN-only runs).",
    keys: ["d", "K", "epochs_lg", "lightgcn_lr"],
  },
  {
    title: "Hybrid QGNN (quantum head)",
    hint: "Quantum-augmented phase. You can set a separate hybrid rate, or leave it empty to use base rate × multiplier. Quantum blend ramps from start to end during hybrid training.",
    keys: ["q", "L", "backend", "epochs_hyb", "hybrid_lr", "hybrid_lr_mult", "p_quantum_start", "p_quantum_end"],
  },
  {
    title: "Evaluation",
    hint: "After training: sampled ranking (Recall@K, NDCG, etc.) and optional ablation. List K values with commas; negatives per query must be at least your largest K.",
    keys: [
      "eval_ranking",
      "eval_test_ranking",
      "eval_hybrid_ablation",
      "log_phase_timings",
      "live_plots",
      "ranking_max_users",
      "ranking_negatives",
      "ranking_ks",
    ],
  },
];

function sectionSlug(title: string): string {
  const s = title
    .replace(/[^a-z0-9]+/gi, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
  return s || "section";
}

function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/** Short labels for the sticky jump bar (avoid wrapping). */
const SECTION_JUMP_LABEL: Record<string, string> = {
  "Data & I/O": "Data",
  Sampling: "Sampling",
  "Training (shared)": "Train",
  "Loss & early stopping": "Loss · ES",
  "Graph baselines (enable models)": "Baselines",
  "Graph baseline training (shared)": "GCN train",
  "Hybrid QGNN (quantum head)": "Hybrid",
  Evaluation: "Eval",
};

const GRAPH_BASELINE_IDS = ["lightgcn", "ultragcn", "sgl", "ncl", "xsimgcl"] as const;

const HYBRID_BACKBONE_LABELS: Record<string, string> = {
  lightgcn: "LightGCN",
  ultragcn: "UltraGCN (lite)",
  sgl: "SGL (lite)",
  ncl: "NCL (lite)",
  xsimgcl: "XSimGCL (lite)",
};

function recordToForm(r: Record<string, unknown>): Record<string, string> {
  const o: Record<string, string> = {};
  for (const [k, v] of Object.entries(r)) {
    if (v === null || v === undefined) continue;
    if (typeof v === "boolean") {
      o[k] = v ? "true" : "false";
    } else if (typeof v === "number") {
      o[k] = String(v);
    } else if (Array.isArray(v)) {
      o[k] = v.map((x) => String(x)).join(", ");
    } else {
      o[k] = String(v);
    }
  }
  return o;
}

function parseIntList(raw: string): number[] {
  return raw
    .split(/[,;\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => parseInt(s, 10))
    .filter((n) => !Number.isNaN(n));
}

function buildPayload(
  preset: "lightweight" | "notebook" | "large" | "extra_large" | "custom",
  form: Record<string, string>,
  computeMode: ComputeModeId
): Record<string, unknown> {
  const out: Record<string, unknown> = { preset, quick_demo: false, compute_mode: computeMode };
  for (const [k, raw] of Object.entries(form)) {
    const key = k.trim();
    const v = raw.trim();
    if (v === "") continue;
    if (INT_KEYS.has(key)) {
      const n = parseInt(v, 10);
      if (!Number.isNaN(n)) out[key] = n;
    } else if (FLOAT_KEYS.has(key)) {
      const n = parseFloat(v);
      if (!Number.isNaN(n)) out[key] = n;
    } else if (BOOL_KEYS.has(key)) {
      if (v === "true") out[key] = true;
      else if (v === "false") out[key] = false;
    } else if (key === "ranking_ks") {
      const nums = parseIntList(v);
      if (nums.length) out[key] = nums;
    } else if (key === "training_loss" || key === "early_stopping_monitor") {
      if (v) out[key] = v;
    } else {
      out[key] = v;
    }
  }
  return out;
}

function buildBatchPayload(
  preset: "lightweight" | "notebook" | "large" | "extra_large" | "custom",
  form: Record<string, string>,
  computeMode: ComputeModeId,
  batch: { seeds: string; sweepQ: string; sweepL: string; entanglementAblation: boolean },
): Record<string, unknown> {
  const out = buildPayload(preset, form, computeMode);
  const seeds = parseIntList(batch.seeds);
  if (seeds.length) out.seeds = seeds;
  const sq = parseIntList(batch.sweepQ);
  if (sq.length) out.sweep_q = sq;
  const sL = parseIntList(batch.sweepL);
  if (sL.length) out.sweep_L = sL;
  if (batch.entanglementAblation) out.sweep_entangle = [true, false];
  return out;
}

function jobStatusClass(status: string) {
  if (status === "completed") return "job-status--completed";
  if (status === "failed") return "job-status--failed";
  if (status === "cancelled") return "job-status--cancelled";
  if (status === "running") return "job-status--running";
  return "job-status--queued";
}

function formatActivity(a: JobActivity): string {
  const parts: string[] = [];
  if (a.model) parts.push(a.model);
  if (a.phase) parts.push(String(a.phase));
  if (a.split) parts.push(a.split);
  if (a.epoch != null && a.epoch !== undefined) parts.push(`ep ${a.epoch}`);
  if (a.batch != null && a.total_batches != null) parts.push(`batch ${a.batch}/${a.total_batches}`);
  if (a.loss != null && a.loss !== undefined) parts.push(`loss ${a.loss}`);
  if (a.p_quantum != null && a.p_quantum !== undefined) parts.push(`p_q ${a.p_quantum}`);
  return parts.join(" · ") || "…";
}

function JobProgressBar({ job }: { job: JobPublic }) {
  const pct = Math.min(100, Math.max(0, job.progress_pct ?? 0));
  const events = job.events ?? [];
  const recent = [...events].slice(-40).reverse();

  return (
    <div className="progress-block" aria-label={`Overall progress ${pct.toFixed(0)} percent`}>
      <div className="progress-block__head">
        <span className="progress-block__label">Overall</span>
        <span className="progress-block__pct mono">{pct.toFixed(1)}%</span>
      </div>
      <div className="progress-track" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      {job.activity && (
        <div className="activity-strip mono" aria-label="Current batch activity">
          <span className="activity-strip__label">Now</span>
          <span className="activity-strip__text">{formatActivity(job.activity)}</span>
        </div>
      )}
      {job.steps.length > 0 && (
        <ol className="step-list">
          {job.steps.map((s) => (
            <li key={s.id} className={`step-list__item step-list__item--${s.status}`}>
              <span className="step-list__dot" aria-hidden />
              <span className="step-list__label">{s.label}</span>
              <span className="step-list__state mono">{s.status}</span>
            </li>
          ))}
        </ol>
      )}
      {recent.length > 0 && (
        <details className="event-log">
          <summary className="event-log__summary mono">Event log ({events.length})</summary>
          <ul className="event-log__list">
            {recent.map((ev, i) => (
              <li key={`${ev.ts}-${i}`} className="event-log__item mono">
                <time className="event-log__time" dateTime={ev.ts}>
                  {new Date(ev.ts).toLocaleTimeString()}
                </time>
                <span className="event-log__msg">{ev.message}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

type PresetId = "lightweight" | "notebook" | "large" | "extra_large" | "custom";

export type CloneRequestPayload = { nonce: number; config: Record<string, unknown> };

type Props = {
  presets: ExperimentPresets | null;
  datasets: DatasetStatus[];
  starting: boolean;
  cancelling: boolean;
  activeJob: JobPublic | null;
  onRun: (body: Record<string, unknown>) => void;
  onCancel: (jobId: string) => void;
  onDismissJob?: () => void;
  /** When set, merge config into the form (from Saved models → Retrain). */
  cloneRequest?: CloneRequestPayload | null;
  onCloneConsumed?: () => void;
};

export function ExperimentPanel({
  presets,
  datasets,
  starting,
  cancelling,
  activeJob,
  onRun,
  onCancel,
  onDismissJob,
  cloneRequest = null,
  onCloneConsumed,
}: Props) {
  const [preset, setPreset] = useState<PresetId>("custom");
  const [form, setForm] = useState<Record<string, string>>({});
  const [computeMode, setComputeMode] = useState<ComputeModeId>("auto");
  const [deviceInfo, setDeviceInfo] = useState<ExperimentDeviceOverview | null>(null);
  const [mainTab, setMainTab] = useState<"single" | "batch">("single");
  const [batchSeeds, setBatchSeeds] = useState("41, 42, 43, 44, 45");
  const [batchSweepQ, setBatchSweepQ] = useState("");
  const [batchSweepL, setBatchSweepL] = useState("");
  const [batchEntAblation, setBatchEntAblation] = useState(false);
  const appliedCloneNonce = useRef<number | null>(null);

  useEffect(() => {
    void getExperimentDevice()
      .then(setDeviceInfo)
      .catch(() => setDeviceInfo(null));
  }, []);

  const template = useMemo(() => {
    if (!presets) return null;
    if (preset === "lightweight") return presets.lightweight ?? presets.quick;
    if (preset === "notebook") return presets.notebook;
    if (preset === "large") return presets.large ?? presets.custom ?? presets.full;
    if (preset === "extra_large") return presets.extra_large ?? presets.custom ?? presets.full;
    return presets.custom ?? presets.full;
  }, [preset, presets]);

  useEffect(() => {
    if (template) {
      const base = recordToForm(template as Record<string, unknown>);
      const merged: Record<string, string> = { ...base };
      for (const g of FORM_GROUPS) {
        for (const k of g.keys) {
          if (merged[k] === undefined) merged[k] = "";
        }
      }
      setForm(merged);
    }
  }, [template]);

  useEffect(() => {
    if (cloneRequest == null || !presets) return;
    if (appliedCloneNonce.current === cloneRequest.nonce) return;
    appliedCloneNonce.current = cloneRequest.nonce;
    setPreset("custom");
    const src = (presets.custom ?? presets.full ?? {}) as Record<string, unknown>;
    const defaults = recordToForm(src);
    const fromRun = recordToForm(cloneRequest.config);
    const merged: Record<string, string> = { ...defaults };
    for (const g of FORM_GROUPS) {
      for (const k of g.keys) {
        if (fromRun[k] !== undefined && fromRun[k] !== "") merged[k] = fromRun[k];
        else if (merged[k] === undefined) merged[k] = "";
      }
    }
    for (const [k, v] of Object.entries(fromRun)) {
      if (merged[k] === undefined && v !== "") merged[k] = v;
    }
    setForm(merged);
    requestAnimationFrame(() => {
      document.getElementById("card-run-heading")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    onCloneConsumed?.();
  }, [cloneRequest?.nonce, presets, onCloneConsumed]);

  const setField = (k: string, v: string) => setForm((f) => ({ ...f, [k]: v }));

  const applyPreset = (p: PresetId) => {
    setPreset(p);
    if (!presets) return;
    const src =
      p === "lightweight"
        ? presets.lightweight ?? presets.quick
        : p === "notebook"
          ? presets.notebook
          : p === "large"
            ? presets.large ?? presets.custom ?? presets.full
            : p === "extra_large"
              ? presets.extra_large ?? presets.custom ?? presets.full
              : presets.custom ?? presets.full;
    if (src) {
      const base = recordToForm(src as Record<string, unknown>);
      const merged: Record<string, string> = { ...base };
      for (const g of FORM_GROUPS) {
        for (const k of g.keys) {
          if (merged[k] === undefined) merged[k] = "";
        }
      }
      setForm(merged);
    }
  };

  const canCancel =
    activeJob && (activeJob.status === "running" || activeJob.status === "queued");

  const datasetPathOptions = useMemo(() => {
    const fromApi = datasets.map((d) => d.data_dir);
    return fromApi.length > 0 ? fromApi : ["dataset/amazon-book", "dataset/movielens-100k"];
  }, [datasets]);

  const dataDirTrim = (form.data_dir ?? "").trim();
  const dataDirIsCustom = !datasetPathOptions.includes(dataDirTrim);

  const backendOptions = useMemo(() => {
    const cur = (form.backend ?? "").trim();
    const base = [...PENNYLANE_BACKENDS];
    if (cur && !base.includes(cur)) base.push(cur);
    return base;
  }, [form.backend]);

  const mountRow = datasets.find((d) => d.data_dir === dataDirTrim);
  const datasetReady = dataDirIsCustom
    ? dataDirTrim.length > 0
    : !!(mountRow?.train_txt && mountRow?.test_txt);

  const batchVariantCount = useMemo(() => {
    const seeds = parseIntList(batchSeeds);
    const nS = seeds.length > 0 ? seeds.length : 1;
    const sq = parseIntList(batchSweepQ);
    const nQ = sq.length > 0 ? sq.length : 1;
    const sL = parseIntList(batchSweepL);
    const nL = sL.length > 0 ? sL.length : 1;
    const nE = batchEntAblation ? 2 : 1;
    return nS * nQ * nL * nE;
  }, [batchSeeds, batchSweepQ, batchSweepL, batchEntAblation]);

  function fieldControl(key: string): ReactNode {
    if (key === "data_dir") {
      return (
        <div className="exp-form__stack">
          <select
            className="exp-form__select"
            aria-label={fieldMeta("data_dir").label}
            value={dataDirIsCustom ? CUSTOM_DATA_DIR : dataDirTrim}
            onChange={(e) => {
              if (e.target.value === CUSTOM_DATA_DIR) setField("data_dir", "");
              else setField("data_dir", e.target.value);
            }}
          >
            {datasetPathOptions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
            <option value={CUSTOM_DATA_DIR}>Custom path…</option>
          </select>
          {dataDirIsCustom && (
            <input
              className="exp-form__input"
              value={form.data_dir ?? ""}
              onChange={(e) => setField("data_dir", e.target.value)}
              placeholder="e.g. dataset/my-benchmark"
              autoComplete="off"
              spellCheck={false}
              aria-label={`${fieldMeta("data_dir").label} — custom path`}
            />
          )}
        </div>
      );
    }

    if (key === "backend") {
      const cur = (form.backend ?? "").trim();
      const sel = backendOptions.includes(cur) ? cur : backendOptions[0];
      return (
        <select
          className="exp-form__select"
          aria-label={fieldMeta("backend").label}
          value={sel}
          onChange={(e) => setField("backend", e.target.value)}
        >
          {backendOptions.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
      );
    }

    if (key === "training_loss") {
      const v = (form[key] ?? "bce").toLowerCase();
      return (
        <select
          className="exp-form__select"
          aria-label={fieldMeta("training_loss").label}
          value={v === "bpr" ? "bpr" : "bce"}
          onChange={(e) => setField(key, e.target.value)}
        >
          <option value="bce">BCE — pointwise pair classification</option>
          <option value="bpr">BPR — pairwise ranking loss</option>
        </select>
      );
    }

    if (key === "early_stopping_monitor") {
      const v = (form[key] ?? "val_auc").toLowerCase();
      return (
        <select
          className="exp-form__select"
          aria-label={fieldMeta("early_stopping_monitor").label}
          value={v === "val_training_loss" ? "val_training_loss" : "val_auc"}
          onChange={(e) => setField(key, e.target.value)}
        >
          <option value="val_auc">Validation AUC (↑ better)</option>
          <option value="val_training_loss">Validation training loss (↓ better)</option>
        </select>
      );
    }

    if (key === "hybrid_backbone") {
      const cur = (form.hybrid_backbone ?? "lightgcn").trim().toLowerCase();
      const sel = (GRAPH_BASELINE_IDS as readonly string[]).includes(cur) ? cur : "lightgcn";
      return (
        <select
          className="exp-form__select"
          aria-label={fieldMeta("hybrid_backbone").label}
          value={sel}
          onChange={(e) => setField("hybrid_backbone", e.target.value)}
        >
          {GRAPH_BASELINE_IDS.map((id) => (
            <option key={id} value={id}>
              {HYBRID_BACKBONE_LABELS[id] ?? id}
            </option>
          ))}
        </select>
      );
    }

    if (key.startsWith("train_baseline_")) {
      const hb = (form.hybrid_backbone ?? "lightgcn").trim().toLowerCase();
      const required = key === `train_baseline_${hb}`;
      const bv = required ? "true" : form[key] === "false" ? "false" : "true";
      return (
        <select
          className="exp-form__select"
          aria-label={fieldMeta(key).label}
          value={bv}
          disabled={required}
          title={
            required
              ? "This model is always trained because it is the hybrid encoder backbone."
              : undefined
          }
          onChange={(e) => setField(key, e.target.value)}
        >
          <option value="true">On</option>
          <option value="false">Off</option>
        </select>
      );
    }

    if (BOOL_KEYS.has(key)) {
      const bv = form[key] === "false" ? "false" : "true";
      return (
        <select
          className="exp-form__select"
          aria-label={fieldMeta(key).label}
          value={bv}
          onChange={(e) => setField(key, e.target.value)}
        >
          <option value="true">On</option>
          <option value="false">Off</option>
        </select>
      );
    }

    const placeholder =
      key === "experiment_name"
        ? "e.g. Amazon-book q=4 baseline"
        : key.includes("lr") && key !== "lr"
          ? "optional"
          : undefined;

    return (
      <input
        className="exp-form__input"
        value={form[key] ?? ""}
        onChange={(e) => setField(key, e.target.value)}
        autoComplete="off"
        spellCheck={key !== "experiment_name"}
        placeholder={placeholder}
        aria-label={fieldMeta(key).label}
      />
    );
  }

  return (
    <section className="card experiment-card" aria-labelledby="card-run-heading">
      <div className="card__head">
        <h2 id="card-run-heading" className="card__title">
          Run experiment
        </h2>
        <div className="card__icon card__icon--violet" aria-hidden>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path
              d="M10 3h4v5.5l4.5 9.5a2 2 0 0 1-1.8 2.9H7.3a2 2 0 0 1-1.8-2.9L10 8.5V3z"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinejoin="round"
            />
            <path d="M9 14h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
      </div>

      <div className="experiment-card__runner">
        <nav className="exp-jump-nav" aria-label="Jump to form section">
        <span className="exp-jump-nav__label mono">Jump</span>
        <button type="button" className="exp-jump-nav__btn" onClick={() => scrollToSection("exp-sec-compute")}>
          Compute
        </button>
        <button type="button" className="exp-jump-nav__btn" onClick={() => scrollToSection("exp-sec-presets")}>
          Presets
        </button>
        <button type="button" className="exp-jump-nav__btn" onClick={() => scrollToSection("exp-sec-mode")}>
          Mode
        </button>
        {mainTab === "batch" && (
          <button type="button" className="exp-jump-nav__btn" onClick={() => scrollToSection("exp-sec-batch")}>
            Batch
          </button>
        )}
        {FORM_GROUPS.map((g) => (
          <button
            key={g.title}
            type="button"
            className="exp-jump-nav__btn"
            onClick={() => scrollToSection(`exp-sec-${sectionSlug(g.title)}`)}
          >
            {SECTION_JUMP_LABEL[g.title] ?? g.title.slice(0, 14)}
          </button>
        ))}
        {activeJob && (
          <button type="button" className="exp-jump-nav__btn exp-jump-nav__btn--live" onClick={() => scrollToSection("exp-sec-live")}>
            Live
          </button>
        )}
        <button type="button" className="exp-jump-nav__btn exp-jump-nav__btn--accent" onClick={() => scrollToSection("exp-sec-actions")}>
          Run ▼
        </button>
        </nav>

        <div className="experiment-card__main">
      <p className="card__body exp-anchor" id="exp-sec-intro" style={{ marginTop: 0 }}>
        The block below chooses whether training uses your <strong>graphics card (GPU)</strong> or <strong>CPU</strong>, and
        which <strong>quantum simulator</strong> runs the hybrid layer. GPU mode uses CUDA for the neural-network math and
        prefers a fast GPU-backed simulator when installed; otherwise the server falls back to a CPU quantum simulator.
      </p>
      <div className="exp-form__group compute-profile exp-anchor" id="exp-sec-compute" style={{ marginBottom: "1rem" }}>
        <div className="exp-form__legend">Compute profile</div>
        <div className="exp-form__grid" style={{ alignItems: "end" }}>
          <label className="exp-form__field">
            <ComputeModeLabel />
            <select
              className="exp-form__select"
              aria-label={COMPUTE_MODE_META.label}
              value={computeMode}
              onChange={(e) => setComputeMode(e.target.value as ComputeModeId)}
            >
              <option value="auto">Auto (CUDA if available, else CPU)</option>
              <option value="cpu">CPU only (PyTorch CPU + lightning.qubit)</option>
              <option value="gpu">GPU (CUDA + lightning.gpu when possible)</option>
            </select>
          </label>
        </div>
        {deviceInfo && (
          <p
            className="exp-form__hint mono"
            style={{ marginTop: "0.5rem", marginBottom: 0, fontSize: "0.78rem", lineHeight: 1.45 }}
          >
            PyTorch {deviceInfo.torch_version}
            {deviceInfo.cuda_available
              ? ` · CUDA ${deviceInfo.cuda_version ?? "?"} (${deviceInfo.cuda_device_count} GPU)`
              : " · CUDA not available"}
            {" · "}
            <span className={deviceInfo.quantum_simulators?.["lightning.gpu"]?.available ? "" : "path-display--warn"}>
              lightning.gpu {deviceInfo.quantum_simulators?.["lightning.gpu"]?.available ? "OK" : "unavailable"}
            </span>
            {deviceInfo.quantum_simulators?.["lightning.gpu"]?.error && !deviceInfo.quantum_simulators["lightning.gpu"].available
              ? ` — ${deviceInfo.quantum_simulators["lightning.gpu"].error!.slice(0, 120)}`
              : ""}
          </p>
        )}
        {!deviceInfo && <p className="exp-form__hint">Loading device probe…</p>}
      </div>

      <p className="card__body" style={{ marginTop: 0 }}>
        Choose <strong>lightweight</strong> for a short test, <strong>notebook</strong> for the medium thesis preset,{" "}
        <strong>large</strong> for BPR training, more users, early stopping, and stronger sampled ranking (199 negatives/query),{" "}
        <strong>extra large</strong> for the largest default user cap, longer budgets, Recall/NDCG through @100, and 499
        negatives/query (ranking evaluation is slower), or <strong>custom</strong> for full defaults. Advanced rates are optional: empty LightGCN or
        hybrid rate boxes use the shared
        base learning rate (hybrid also uses the multiplier field when its own rate is empty). To label a run in history, fill{" "}
        <strong>Experiment display name</strong> and clear <strong>Output folder</strong>; a new timestamped folder under{" "}
        <span className="code-inline">runs/</span> is created. Lightweight preset without a name: clear output folder for an
        auto <span className="code-inline">runs/quick_*</span> folder. For MovieLens-100K, set dataset to{" "}
        <span className="code-inline">dataset/movielens-100k</span> after running{" "}
        <span className="code-inline">python scripts/prepare_movielens100k.py</span> once from the repo root.
      </p>
      {!datasetReady && presets && (
        <p className="card__body" style={{ marginTop: "0.5rem", color: "var(--rose)" }}>
          {!dataDirTrim
            ? "Choose a dataset folder or enter a custom path."
            : `Selected ${dataDirTrim} is missing train.txt or test.txt under the project root.`}
        </p>
      )}

      <div className="preset-switch preset-switch--triple exp-anchor" id="exp-sec-presets" role="group" aria-label="Configuration preset">
        <button
          type="button"
          className={`preset-switch__btn${preset === "lightweight" ? " preset-switch__btn--active" : ""}`}
          onClick={() => applyPreset("lightweight")}
          disabled={!presets}
        >
          Lightweight
        </button>
        <button
          type="button"
          className={`preset-switch__btn${preset === "notebook" ? " preset-switch__btn--active" : ""}`}
          onClick={() => applyPreset("notebook")}
          disabled={!presets}
        >
          Notebook
        </button>
        <button
          type="button"
          className={`preset-switch__btn${preset === "large" ? " preset-switch__btn--active" : ""}`}
          onClick={() => applyPreset("large")}
          disabled={!presets}
        >
          Large
        </button>
        <button
          type="button"
          className={`preset-switch__btn${preset === "extra_large" ? " preset-switch__btn--active" : ""}`}
          onClick={() => applyPreset("extra_large")}
          disabled={!presets}
        >
          Extra large
        </button>
        <button
          type="button"
          className={`preset-switch__btn${preset === "custom" ? " preset-switch__btn--active" : ""}`}
          onClick={() => applyPreset("custom")}
          disabled={!presets}
        >
          Custom
        </button>
      </div>

      <div
        className="preset-switch preset-switch--triple exp-anchor"
        id="exp-sec-mode"
        style={{ marginTop: "0.75rem" }}
        role="tablist"
        aria-label="Run mode"
      >
        <button
          type="button"
          role="tab"
          aria-selected={mainTab === "single"}
          className={`preset-switch__btn${mainTab === "single" ? " preset-switch__btn--active" : ""}`}
          onClick={() => setMainTab("single")}
        >
          Single run
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mainTab === "batch"}
          className={`preset-switch__btn${mainTab === "batch" ? " preset-switch__btn--active" : ""}`}
          onClick={() => setMainTab("batch")}
        >
          Multi-seed &amp; ablations
        </button>
      </div>

      {mainTab === "batch" && (
        <div className="exp-form__group exp-anchor" id="exp-sec-batch" style={{ marginTop: "1rem" }}>
          <div className="exp-form__legend">Batch planner</div>
          <p className="exp-form__hint">
            Uses all fields below (data, loss, epochs, etc.) as the base config. Seeds and sweep lists multiply: leave a
            sweep box empty to use only the value from the form (e.g. q / L from the hybrid section). Output goes to{" "}
            <span className="mono">run_*</span> subfolders under your chosen output directory;{" "}
            <span className="mono">batch_summary.json</span> lists children when there are multiple jobs. Cancelling stops
            between sub-runs.
          </p>
          <div className="exp-form__grid">
            <label className="exp-form__field">
              <span className="exp-form__label-stack">
                <span className="exp-form__label">Seeds (comma-separated)</span>
                <span className="exp-form__param mono" title="API field">
                  seeds
                </span>
              </span>
              <input
                className="exp-form__input mono"
                value={batchSeeds}
                onChange={(e) => setBatchSeeds(e.target.value)}
                placeholder="e.g. 41, 42, 43 — empty = use Random seed from form only"
                autoComplete="off"
                spellCheck={false}
                aria-label="Batch seeds"
              />
            </label>
            <label className="exp-form__field">
              <span className="exp-form__label-stack">
                <span className="exp-form__label">Sweep qubits (optional)</span>
                <span className="exp-form__param mono">sweep_q</span>
              </span>
              <input
                className="exp-form__input mono"
                value={batchSweepQ}
                onChange={(e) => setBatchSweepQ(e.target.value)}
                placeholder="e.g. 4, 8, 12 — empty = form q only"
                autoComplete="off"
              />
            </label>
            <label className="exp-form__field">
              <span className="exp-form__label-stack">
                <span className="exp-form__label">Sweep quantum depths L (optional)</span>
                <span className="exp-form__param mono">sweep_L</span>
              </span>
              <input
                className="exp-form__input mono"
                value={batchSweepL}
                onChange={(e) => setBatchSweepL(e.target.value)}
                placeholder="e.g. 1, 3 — empty = form L only"
                autoComplete="off"
              />
            </label>
            <label className="exp-form__field">
              <span className="exp-form__label-stack">
                <span className="exp-form__label">Thesis entanglement ablation</span>
                <span className="exp-form__param mono">sweep_entangle</span>
              </span>
              <select
                className="exp-form__select"
                aria-label="Entanglement sweep"
                value={batchEntAblation ? "both" : "form"}
                onChange={(e) => setBatchEntAblation(e.target.value === "both")}
              >
                <option value="form">Use quantum circuit setting from form above</option>
                <option value="both">Run twice: with and without ring CNOT (true, false)</option>
              </select>
            </label>
            <div className="exp-form__field" style={{ gridColumn: "1 / -1" }}>
              <p className="exp-form__hint" style={{ margin: 0 }}>
                Planned training jobs: <strong>{batchVariantCount}</strong>
              </p>
            </div>
          </div>
        </div>
      )}

      {!presets ? (
        <p className="empty-state" style={{ padding: "1rem 0" }}>
          Loading presets…
        </p>
      ) : (
        <div className="exp-form">
          {FORM_GROUPS.map((g) => (
            <fieldset key={g.title} className="exp-form__group exp-anchor" id={`exp-sec-${sectionSlug(g.title)}`}>
              <legend className="exp-form__legend">{g.title}</legend>
              {g.hint && <p className="exp-form__hint">{g.hint}</p>}
              <div className="exp-form__grid">
                {g.keys.map((key) => (
                  <label key={key} className="exp-form__field">
                    <FieldLabel fieldKey={key} />
                    {fieldControl(key)}
                  </label>
                ))}
              </div>
            </fieldset>
          ))}
        </div>
      )}

      <div className="btn-row btn-row--split exp-anchor" id="exp-sec-actions">
        <button
          type="button"
          className="btn btn--primary"
          disabled={starting || !datasetReady || !presets}
          onClick={() =>
            mainTab === "batch"
              ? onRun(
                  buildBatchPayload(preset, form, computeMode, {
                    seeds: batchSeeds,
                    sweepQ: batchSweepQ,
                    sweepL: batchSweepL,
                    entanglementAblation: batchEntAblation,
                  }),
                )
              : onRun(buildPayload(preset, form, computeMode))
          }
        >
          {starting
            ? "Starting…"
            : mainTab === "batch"
              ? `Run batch (${batchVariantCount} jobs)`
              : "Run with settings above"}
        </button>
        {canCancel && (
          <button
            type="button"
            className="btn btn--danger"
            disabled={cancelling}
            onClick={() => onCancel(activeJob.id)}
          >
            {cancelling ? "Cancelling…" : "Cancel run"}
          </button>
        )}
      </div>

        </div>
      </div>

      {activeJob && (
        <div className="job-panel exp-anchor" id="exp-sec-live" role="status" aria-live="polite" style={{ marginTop: "1.25rem" }}>
          <div className="job-panel__header">
            <span className={`job-status ${jobStatusClass(activeJob.status)}`}>{activeJob.status}</span>
            <span className="job-phase">
              {activeJob.phase}
              {activeJob.detail ? ` · ${activeJob.detail}` : ""}
            </span>
            {onDismissJob && (
              <div className="job-panel__header-actions">
                <button type="button" className="btn btn--secondary btn--compact" onClick={() => onDismissJob()}>
                  Hide panel
                </button>
              </div>
            )}
          </div>
          {(activeJob.status === "running" ||
            activeJob.status === "queued" ||
            activeJob.steps.length > 0 ||
            (activeJob.events && activeJob.events.length > 0)) && <JobProgressBar job={activeJob} />}
          <LiveTrainingMonitor job={activeJob} />
          {activeJob.error && <p className="job-error">{activeJob.error}</p>}
          {activeJob.result && <pre className="job-pre">{JSON.stringify(activeJob.result, null, 2)}</pre>}
        </div>
      )}
    </section>
  );
}
