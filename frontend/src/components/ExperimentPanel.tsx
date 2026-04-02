import { useEffect, useMemo, useState } from "react";
import type { DatasetStatus, ExperimentPresets, JobActivity, JobPublic } from "../api";

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
  "lightgcn_lr",
  "hybrid_lr",
  "wd",
  "hybrid_lr_mult",
  "p_quantum_start",
  "p_quantum_end",
]);

const FORM_GROUPS: { title: string; hint?: string; keys: string[] }[] = [
  {
    title: "Data & I/O",
    hint: "data_dir: dataset/amazon-book or dataset/movielens-100k (train.txt + test.txt in that folder).",
    keys: ["data_dir", "save_dir", "seed"],
  },
  { title: "Sampling", keys: ["max_users", "max_pos_per_user", "neg_per_pos", "val_ratio"] },
  {
    title: "Training (shared)",
    hint: "Base lr is used when LightGCN or hybrid learning rate is left empty.",
    keys: ["batch_size", "micro_bs", "lr", "wd", "eval_every"],
  },
  {
    title: "LightGCN",
    hint: "Optional lightgcn_lr overrides base lr for the encoder-only phase.",
    keys: ["d", "K", "epochs_lg", "lightgcn_lr"],
  },
  {
    title: "Hybrid QGNN",
    hint: "Optional hybrid_lr overrides lr × hybrid_lr_mult for the quantum head phase.",
    keys: ["q", "L", "backend", "epochs_hyb", "hybrid_lr", "hybrid_lr_mult", "p_quantum_start", "p_quantum_end"],
  },
];

function recordToForm(r: Record<string, unknown>): Record<string, string> {
  const o: Record<string, string> = {};
  for (const [k, v] of Object.entries(r)) {
    if (v === null || v === undefined) continue;
    o[k] = typeof v === "number" ? String(v) : String(v);
  }
  return o;
}

function buildPayload(
  preset: "lightweight" | "notebook" | "custom",
  form: Record<string, string>
): Record<string, unknown> {
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

type PresetId = "lightweight" | "notebook" | "custom";

type Props = {
  presets: ExperimentPresets | null;
  datasets: DatasetStatus[];
  starting: boolean;
  cancelling: boolean;
  activeJob: JobPublic | null;
  onRun: (body: Record<string, unknown>) => void;
  onCancel: (jobId: string) => void;
};

export function ExperimentPanel({
  presets,
  datasets,
  starting,
  cancelling,
  activeJob,
  onRun,
  onCancel,
}: Props) {
  const [preset, setPreset] = useState<PresetId>("custom");
  const [form, setForm] = useState<Record<string, string>>({});

  const template = useMemo(() => {
    if (!presets) return null;
    if (preset === "lightweight") return presets.lightweight ?? presets.quick;
    if (preset === "notebook") return presets.notebook;
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

  const setField = (k: string, v: string) => setForm((f) => ({ ...f, [k]: v }));

  const applyPreset = (p: PresetId) => {
    setPreset(p);
    if (!presets) return;
    const src =
      p === "lightweight"
        ? presets.lightweight ?? presets.quick
        : p === "notebook"
          ? presets.notebook
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

  const dataDirKey = (form.data_dir || "dataset/amazon-book").trim();
  const mountRow = datasets.find((d) => d.data_dir === dataDirKey);
  const datasetReady = !!(mountRow?.train_txt && mountRow.test_txt);

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
        Choose <strong>lightweight</strong> for a quick smoke test, <strong>notebook</strong> for the original notebook
        balanced profile, or <strong>custom</strong> starting from library defaults. LightGCN and hybrid phases can use
        separate learning rates; leave them empty to fall back to the shared base lr (hybrid also respects{" "}
        <span className="code-inline">hybrid_lr_mult</span>). For lightweight, leave{" "}
        <span className="code-inline">save_dir</span> empty to auto-create a timestamped{" "}
        <span className="code-inline">runs/quick_*</span> folder. Set{" "}
        <span className="code-inline">data_dir</span> to <span className="code-inline">dataset/movielens-100k</span> after
        running <span className="code-inline">python scripts/prepare_movielens100k.py</span> from the repo root.
      </p>
      {!datasetReady && presets && (
        <p className="card__body" style={{ marginTop: "0.5rem", color: "var(--rose)" }}>
          Selected <span className="mono">{dataDirKey}</span> is missing <span className="code-inline">train.txt</span>{" "}
          or <span className="code-inline">test.txt</span> — fix paths or choose another dataset.
        </p>
      )}

      <div className="preset-switch preset-switch--triple" role="group" aria-label="Configuration preset">
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
          className={`preset-switch__btn${preset === "custom" ? " preset-switch__btn--active" : ""}`}
          onClick={() => applyPreset("custom")}
          disabled={!presets}
        >
          Custom
        </button>
      </div>

      {!presets ? (
        <p className="empty-state" style={{ padding: "1rem 0" }}>
          Loading presets…
        </p>
      ) : (
        <div className="exp-form">
          {FORM_GROUPS.map((g) => (
            <fieldset key={g.title} className="exp-form__group">
              <legend className="exp-form__legend">{g.title}</legend>
              {g.hint && <p className="exp-form__hint">{g.hint}</p>}
              <div className="exp-form__grid">
                {g.keys.map((key) => (
                  <label key={key} className="exp-form__field">
                    <span className="exp-form__key mono">{key}</span>
                    <input
                      className="exp-form__input"
                      value={form[key] ?? ""}
                      onChange={(e) => setField(key, e.target.value)}
                      autoComplete="off"
                      spellCheck={false}
                      placeholder={key.includes("lr") && key !== "lr" ? "optional" : undefined}
                    />
                  </label>
                ))}
              </div>
            </fieldset>
          ))}
        </div>
      )}

      <div className="btn-row btn-row--split">
        <button
          type="button"
          className="btn btn--primary"
          disabled={starting || !datasetReady || !presets}
          onClick={() => onRun(buildPayload(preset, form))}
        >
          {starting ? "Starting…" : "Run with settings above"}
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

      {activeJob && (
        <div className="job-panel" role="status" aria-live="polite" style={{ marginTop: "1.25rem" }}>
          {(activeJob.status === "running" ||
            activeJob.status === "queued" ||
            activeJob.steps.length > 0 ||
            (activeJob.events && activeJob.events.length > 0)) && <JobProgressBar job={activeJob} />}
          <div
            className="job-panel__header"
            style={{ marginTop: activeJob.steps.length || activeJob.events?.length ? "1rem" : 0 }}
          >
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
