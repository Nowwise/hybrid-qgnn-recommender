import { useEffect, useMemo, useState } from "react";
import type { ExperimentPresets, JobPublic } from "../api";

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
]);

const FLOAT_KEYS = new Set([
  "val_ratio",
  "lr",
  "wd",
  "hybrid_lr_mult",
  "p_quantum_start",
  "p_quantum_end",
]);

function recordToForm(r: Record<string, unknown>): Record<string, string> {
  const o: Record<string, string> = {};
  for (const [k, v] of Object.entries(r)) {
    if (v === null || v === undefined) continue;
    o[k] = typeof v === "number" ? String(v) : String(v);
  }
  return o;
}

function buildPayload(preset: "quick" | "full", form: Record<string, string>): Record<string, unknown> {
  const out: Record<string, unknown> = { preset, quick_demo: false };
  for (const [k, raw] of Object.entries(form)) {
    const v = raw.trim();
    if (v === "") continue;
    if (INT_KEYS.has(k)) {
      const n = parseInt(v, 10);
      if (!Number.isNaN(n)) out[k] = n;
    } else if (FLOAT_KEYS.has(k)) {
      const n = parseFloat(v);
      if (!Number.isNaN(n)) out[k] = n;
    } else {
      out[k] = v;
    }
  }
  return out;
}

function jobStatusClass(status: string) {
  if (status === "completed") return "job-status--completed";
  if (status === "failed") return "job-status--failed";
  if (status === "running") return "job-status--running";
  return "job-status--queued";
}

function JobProgressBar({ job }: { job: JobPublic }) {
  const pct = Math.min(100, Math.max(0, job.progress_pct ?? 0));
  return (
    <div className="progress-block" aria-label={`Overall progress ${pct.toFixed(0)} percent`}>
      <div className="progress-block__head">
        <span className="progress-block__label">Overall</span>
        <span className="progress-block__pct mono">{pct.toFixed(1)}%</span>
      </div>
      <div className="progress-track" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
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
    </div>
  );
}

type Props = {
  presets: ExperimentPresets | null;
  datasetReady: boolean;
  starting: boolean;
  activeJob: JobPublic | null;
  onRun: (body: Record<string, unknown>) => void;
};

export function ExperimentPanel({ presets, datasetReady, starting, activeJob, onRun }: Props) {
  const [preset, setPreset] = useState<"quick" | "full">("full");
  const [form, setForm] = useState<Record<string, string>>({});

  const template = useMemo(() => (preset === "quick" ? presets?.quick : presets?.full), [preset, presets]);

  useEffect(() => {
    if (template) {
      setForm(recordToForm(template as Record<string, unknown>));
    }
  }, [template]);

  const setField = (k: string, v: string) => setForm((f) => ({ ...f, [k]: v }));

  const applyPreset = (p: "quick" | "full") => {
    setPreset(p);
    const src = p === "quick" ? presets?.quick : presets?.full;
    if (src) setForm(recordToForm(src as Record<string, unknown>));
  };

  const fields = useMemo(() => {
    const keys = new Set<string>([
      "data_dir",
      "save_dir",
      "max_users",
      "max_pos_per_user",
      "neg_per_pos",
      "val_ratio",
      "epochs_lg",
      "epochs_hyb",
      "batch_size",
      "micro_bs",
      "d",
      "K",
      "q",
      "L",
      "lr",
      "wd",
      "hybrid_lr_mult",
      "eval_every",
      "backend",
      "p_quantum_start",
      "p_quantum_end",
      "seed",
    ]);
    if (template) {
      for (const k of Object.keys(template as object)) keys.add(k);
    }
    return [...keys].sort();
  }, [template]);

  return (
    <section className="card" aria-labelledby="card-run-heading">
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

      <p className="card__body" style={{ marginTop: 0 }}>
        Pick a <strong>preset</strong>, adjust any field, then start. Progress updates live for data prep, LightGCN
        epochs, hybrid warmup, hybrid epochs, and export. For <strong>quick</strong> runs, clear{" "}
        <span className="code-inline">save_dir</span> to auto-create a timestamped{" "}
        <span className="code-inline">runs/quick_*</span> folder.
      </p>

      <div className="preset-switch" role="group" aria-label="Configuration preset">
        <button
          type="button"
          className={`preset-switch__btn${preset === "quick" ? " preset-switch__btn--active" : ""}`}
          onClick={() => applyPreset("quick")}
          disabled={!presets}
        >
          Quick demo
        </button>
        <button
          type="button"
          className={`preset-switch__btn${preset === "full" ? " preset-switch__btn--active" : ""}`}
          onClick={() => applyPreset("full")}
          disabled={!presets}
        >
          Full default
        </button>
      </div>

      {!presets ? (
        <p className="empty-state" style={{ padding: "1rem 0" }}>
          Loading presets…
        </p>
      ) : (
        <div className="exp-form">
          <div className="exp-form__grid">
            {fields.map((key) => (
              <label key={key} className="exp-form__field">
                <span className="exp-form__key mono">{key}</span>
                <input
                  className="exp-form__input"
                  value={form[key] ?? ""}
                  onChange={(e) => setField(key, e.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
              </label>
            ))}
          </div>
        </div>
      )}

      <div className="btn-row">
        <button
          type="button"
          className="btn btn--primary"
          disabled={starting || !datasetReady || !presets}
          onClick={() => onRun(buildPayload(preset, form))}
        >
          {starting ? "Starting…" : "Run with settings above"}
        </button>
      </div>

      {activeJob && (
        <div className="job-panel" role="status" aria-live="polite" style={{ marginTop: "1.25rem" }}>
          {(activeJob.status === "running" || activeJob.steps.length > 0) && (
            <JobProgressBar job={activeJob} />
          )}
          <div className="job-panel__header" style={{ marginTop: activeJob.steps.length ? "1rem" : 0 }}>
            <span className={`job-status ${jobStatusClass(activeJob.status)}`}>{activeJob.status}</span>
            <span className="job-phase">
              {activeJob.phase}
              {activeJob.detail ? ` · ${activeJob.detail}` : ""}
            </span>
          </div>
          {activeJob.error && <p className="job-error">{activeJob.error}</p>}
          {activeJob.result && <pre className="job-pre">{JSON.stringify(activeJob.result, null, 2)}</pre>}
        </div>
      )}
    </section>
  );
}
