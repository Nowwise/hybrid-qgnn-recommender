import { useState } from "react";
import { getRunConfig, type RunSummary } from "../api";
import { prepareRunConfigForFormClone } from "../runClone";

type Props = {
  history: RunSummary[];
  deletingRunId: string | null;
  onBack: () => void;
  onCloneToLab: (config: Record<string, unknown>) => void;
  onDeleteRun: (runId: string) => void;
  onRefresh: () => void;
};

function fmtAuc(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(4);
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export function SavedModelsPage({
  history,
  deletingRunId,
  onBack,
  onCloneToLab,
  onDeleteRun,
  onRefresh,
}: Props) {
  const [openId, setOpenId] = useState<string | null>(null);
  const [loadingCloneId, setLoadingCloneId] = useState<string | null>(null);
  const [cloneErr, setCloneErr] = useState<string | null>(null);

  async function cloneRun(runId: string) {
    setCloneErr(null);
    setLoadingCloneId(runId);
    try {
      const raw = (await getRunConfig(runId)) as Record<string, unknown>;
      onCloneToLab(prepareRunConfigForFormClone(raw));
    } catch (e) {
      setCloneErr(e instanceof Error ? e.message : "Could not load run config");
    } finally {
      setLoadingCloneId(null);
    }
  }

  return (
    <div className="saved-models">
      <header className="saved-models__hero">
        <button type="button" className="btn btn--secondary saved-models__back" onClick={onBack}>
          ← Back to lab
        </button>
        <div>
          <h1 className="saved-models__title">Saved models</h1>
          <p className="saved-models__subtitle">
            Each card is a folder under <span className="code-inline">runs/</span> with{" "}
            <span className="code-inline">run_config.json</span> and checkpoints (
            <span className="code-inline">hyb_best.pt</span>, graph baselines). Use{" "}
            <strong>Retrain from settings</strong> to copy hyperparameters into Run experiment (new output folder). To
            point at another dataset, change <span className="code-inline">data_dir</span> on the lab page before
            starting — full retraining is required; checkpoints are tied to the user/item graph from the original run.
          </p>
        </div>
        <button type="button" className="btn btn--secondary" onClick={() => void onRefresh()}>
          Refresh list
        </button>
      </header>

      {cloneErr && (
        <div className="alert" role="alert">
          {cloneErr}
        </div>
      )}

      {history.length === 0 ? (
        <div className="empty-state saved-models__empty">
          No runs yet. Train once from the lab — artifacts will show up here.
        </div>
      ) : (
        <ul className="saved-models__grid" aria-label="Trained run folders">
          {history.map((h) => {
            const open = openId === h.run_id;
            return (
              <li key={h.run_id} className={`model-card${open ? " model-card--open" : ""}`}>
                <button
                  type="button"
                  className="model-card__main"
                  onClick={() => setOpenId(open ? null : h.run_id)}
                  aria-expanded={open}
                >
                  <span className="model-card__name">{h.experiment_name?.trim() || h.run_id}</span>
                  <span className="model-card__folder mono">{h.run_id}</span>
                  <div className="model-card__chips">
                    {h.data_dir && (
                      <span className="model-card__chip mono" title="Dataset">
                        {h.data_dir}
                      </span>
                    )}
                    {h.hybrid_backbone && (
                      <span className="model-card__chip">backbone: {h.hybrid_backbone}</span>
                    )}
                    {h.q != null && h.L != null && (
                      <span className="model-card__chip">
                        q={h.q} · L={h.L}
                      </span>
                    )}
                    {h.d != null && h.K != null && (
                      <span className="model-card__chip">
                        d={h.d} · K={h.K}
                      </span>
                    )}
                  </div>
                  <div className="model-card__metrics">
                    <span title="Best hybrid val AUC (from summary.txt)">Hybrid AUC {fmtAuc(h.best_hybrid_auc)}</span>
                    <span className="model-card__dot" aria-hidden />
                    <span title="LightGCN line in summary (if present)">LG AUC {fmtAuc(h.best_lightgcn_auc)}</span>
                  </div>
                  <div className="model-card__artifacts">
                    {h.has_hybrid_checkpoint ? (
                      <span className="yes-no yes-no--yes">hybrid ckpt</span>
                    ) : (
                      <span className="yes-no yes-no--no">no hybrid ckpt</span>
                    )}
                    <span className="model-card__dot" aria-hidden />
                    <span className="mono model-card__artifact-count">
                      {h.n_baseline_checkpoints ?? 0} graph ckpts
                    </span>
                    {h.has_graph_context ? (
                      <>
                        <span className="model-card__dot" aria-hidden />
                        <span className="yes-no yes-no--yes">score w/o train</span>
                      </>
                    ) : null}
                  </div>
                  <span className="model-card__hint mono">{fmtDate(h.modified_at ?? null)}</span>
                </button>

                {open && (
                  <div className="model-card__detail">
                    <dl className="model-card__dl">
                      <div>
                        <dt>Epochs (graph / hybrid)</dt>
                        <dd className="mono">
                          {h.epochs_lg ?? "—"} / {h.epochs_hyb ?? "—"}
                        </dd>
                      </div>
                      <div>
                        <dt>Metrics CSV</dt>
                        <dd>{h.has_metrics ? "Yes" : "No"}</dd>
                      </div>
                      <div>
                        <dt>Summary</dt>
                        <dd>{h.has_summary ? "Yes" : "No"}</dd>
                      </div>
                    </dl>
                    <div className="model-card__actions">
                      <button
                        type="button"
                        className="btn btn--primary"
                        disabled={loadingCloneId === h.run_id}
                        onClick={(e) => {
                          e.stopPropagation();
                          void cloneRun(h.run_id);
                        }}
                      >
                        {loadingCloneId === h.run_id ? "Loading…" : "Retrain from settings → Lab"}
                      </button>
                      <p className="model-card__action-hint">
                        Opens the lab with this run’s hyperparameters (new <span className="code-inline">runs/…</span>{" "}
                        name). Edit <span className="code-inline">data_dir</span> there if you want another benchmark.
                      </p>
                      <button
                        type="button"
                        className="btn btn--danger btn--compact"
                        disabled={deletingRunId === h.run_id}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!window.confirm(`Delete run folder "${h.run_id}" and all files?`)) return;
                          onDeleteRun(h.run_id);
                        }}
                      >
                        {deletingRunId === h.run_id ? "…" : "Delete folder"}
                      </button>
                    </div>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
