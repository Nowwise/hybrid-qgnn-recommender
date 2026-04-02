import { useCallback, useMemo, useState } from "react";
import { useDashboard } from "./hooks/useDashboard";
import { ExperimentPanel, type CloneRequestPayload } from "./components/ExperimentPanel";
import { SavedModelsPage } from "./components/SavedModelsPage";
import { ScorePairsPage } from "./components/ScorePairsPage";
import { IconAlert, IconServer, IconTable } from "./components/Icons";

function StatusBadge({ label, variant }: { label: string; variant: "ok" | "bad" | "pending" }) {
  return (
    <span className={`badge badge--${variant}`}>
      <span className="badge__dot" aria-hidden />
      {label}
    </span>
  );
}

type AppView = "lab" | "saved_models" | "score_pairs";

export function App() {
  const [view, setView] = useState<AppView>("lab");
  const [cloneRequest, setCloneRequest] = useState<CloneRequestPayload | null>(null);
  const onCloneConsumed = useCallback(() => setCloneRequest(null), []);

  const {
    apiOk,
    datasets,
    history,
    jobs,
    activeJob,
    comparative,
    selectedRun,
    err,
    starting,
    cancelling,
    anyDatasetReady,
    presets,
    refresh,
    runExperiment,
    cancelRun,
    onSelectRun,
    openJob,
    dismissActiveJob,
    wipeHistory,
    clearingHistory,
    removeHistoryRun,
    deletingRunId,
  } = useDashboard();

  const selectedRunMeta = useMemo(() => {
    if (!selectedRun) return { title: "", folder: "" as string | null };
    const row = history.find((h) => h.run_id === selectedRun);
    const name = row?.experiment_name?.trim();
    return {
      title: name || selectedRun,
      folder: name ? selectedRun : null,
    };
  }, [history, selectedRun]);

  const apiVariant = apiOk === null ? "pending" : apiOk ? "ok" : "bad";
  const apiLabel = apiOk === null ? "Checking API" : apiOk ? "API online" : "API offline";

  const handleCloneFromSaved = useCallback((cfg: Record<string, unknown>) => {
    setCloneRequest({ nonce: Date.now(), config: cfg });
    setView("lab");
  }, []);

  return (
    <div className="app">
      <div className="app__bg" aria-hidden>
        <div className="app__bg-mesh" />
        <div className="app__bg-grid" />
        <div className="app__bg-noise" />
      </div>

      <div className="app__shell">
        <header className="hero">
          <p className="hero__eyebrow">Master thesis · hybrid quantum GNN</p>
          <h1 className="hero__title">
            <span className="hero__title-accent">QGNN</span> lab
          </h1>
          <p className="hero__subtitle">
            Compare LightGCN against the hybrid quantum–classical recommender on Amazon-Book or MovieLens-100K
            (same on-disk layout), stream validation metrics, and audit every run from one control surface.
          </p>
          <nav className="app-view-nav" aria-label="Primary pages">
            <button
              type="button"
              className={`app-view-nav__btn${view === "lab" ? " app-view-nav__btn--active" : ""}`}
              onClick={() => setView("lab")}
            >
              Lab
            </button>
            <button
              type="button"
              className={`app-view-nav__btn${view === "saved_models" ? " app-view-nav__btn--active" : ""}`}
              onClick={() => setView("saved_models")}
            >
              Saved models
            </button>
            <button
              type="button"
              className={`app-view-nav__btn${view === "score_pairs" ? " app-view-nav__btn--active" : ""}`}
              onClick={() => setView("score_pairs")}
            >
              Score pairs
            </button>
          </nav>
        </header>

        {view === "saved_models" ? (
          <SavedModelsPage
            history={history}
            deletingRunId={deletingRunId}
            onBack={() => setView("lab")}
            onCloneToLab={handleCloneFromSaved}
            onDeleteRun={(id) => void removeHistoryRun(id)}
            onRefresh={() => void refresh()}
          />
        ) : null}

        {view === "score_pairs" ? <ScorePairsPage onBack={() => setView("lab")} /> : null}

        {view === "lab" ? (
          <>
        <div className="grid-cards">
          <section className="card" aria-labelledby="card-status-heading">
            <div className="card__head">
              <h2 id="card-status-heading" className="card__title">
                System &amp; dataset
              </h2>
              <div className="card__icon" aria-hidden>
                <IconServer />
              </div>
            </div>
            <div className="badge-row">
              <StatusBadge label={apiLabel} variant={apiVariant} />
              <StatusBadge
                label={anyDatasetReady ? "Benchmark folder(s) ready" : "No benchmark data"}
                variant={anyDatasetReady ? "ok" : "bad"}
              />
            </div>
            {datasets.length > 0 ? (
              <ul className="dataset-list" aria-label="Benchmark datasets">
                {datasets.map((d) => {
                  const ok = d.train_txt && d.test_txt;
                  return (
                    <li
                      key={d.data_dir}
                      className={`path-display path-display--compact${ok ? " path-display--row-ok" : " path-display--row-bad"}`}
                    >
                      <span className={`mono${ok ? "" : " path-display--warn"}`}>{d.data_dir}</span>
                      <span className="mono path-display__state">
                        {ok ? "train.txt ✓ · test.txt ✓" : !d.exists ? "missing" : "incomplete"}
                      </span>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <div className="path-display" aria-busy="true">
                <span className="skeleton skeleton-line" style={{ maxWidth: "14rem" }} />
              </div>
            )}
            <div className="btn-row">
              <button type="button" className="btn btn--secondary" onClick={() => void refresh()}>
                Sync status
              </button>
            </div>
          </section>
        </div>

        <ExperimentPanel
          presets={presets}
          datasets={datasets}
          starting={starting}
          cancelling={cancelling}
          activeJob={activeJob}
          onRun={(body) => void runExperiment(body)}
          onCancel={(id) => void cancelRun(id)}
          onDismissJob={dismissActiveJob}
          cloneRequest={cloneRequest}
          onCloneConsumed={onCloneConsumed}
        />

        {err && (
          <div className="alert" role="alert">
            <span className="alert__icon" aria-hidden>
              <IconAlert />
            </span>
            <div>{err}</div>
          </div>
        )}

        <section className="panel" aria-labelledby="jobs-heading">
          <div className="panel__bar" aria-hidden />
          <div className="panel__inner">
            <h2 id="jobs-heading" className="panel__title">
              Job queue
            </h2>
            {jobs.length === 0 ? (
              <div className="empty-state">
                No training jobs yet.
                <p className="empty-state__hint">Start a run above — completed jobs land here automatically.</p>
              </div>
            ) : (
              <div className="table-wrap" tabIndex={0} role="region" aria-label="Recent jobs">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Job ID</th>
                      <th scope="col">Status</th>
                      <th scope="col">Progress</th>
                      <th scope="col">Phase</th>
                      <th scope="col">Updated</th>
                      <th scope="col">View</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.slice(0, 12).map((j) => (
                      <tr key={j.id}>
                        <td className="mono">{j.id.slice(0, 8)}…</td>
                        <td>
                          <span className="metric-pill" style={{ textTransform: "capitalize" }}>
                            {j.status}
                          </span>
                        </td>
                        <td className="mono">{typeof j.progress_pct === "number" ? `${j.progress_pct.toFixed(0)}%` : "—"}</td>
                        <td className="mono">
                          {j.phase}
                          {j.detail ? ` · ${j.detail}` : ""}
                        </td>
                        <td className="mono">{new Date(j.updated_at).toLocaleString()}</td>
                        <td>
                          <button
                            type="button"
                            className="btn btn--secondary btn--compact"
                            onClick={() => void openJob(j.id)}
                          >
                            Open
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        <section className="panel" aria-labelledby="history-heading">
          <div className="panel__bar" aria-hidden />
          <div className="panel__inner">
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem", flexWrap: "wrap" }}>
              <h2 id="history-heading" className="panel__title" style={{ margin: 0, flex: "1 1 12rem" }}>
                Experiment history
              </h2>
              {(history.length > 0 || jobs.length > 0) && (
                <button
                  type="button"
                  className="btn btn--danger btn--compact"
                  disabled={clearingHistory}
                  onClick={() => {
                    if (
                      !window.confirm(
                        "Delete every run folder under runs/ and remove all jobs from the API queue. This cannot be undone. You must cancel any running or queued job first (the server will refuse otherwise).",
                      )
                    ) {
                      return;
                    }
                    void wipeHistory();
                  }}
                >
                  {clearingHistory ? "Clearing…" : "Clear all history"}
                </button>
              )}
              <div className="card__icon" style={{ width: 36, height: 36 }} aria-hidden>
                <IconTable />
              </div>
            </div>
            {history.length === 0 ? (
              <div className="empty-state">
                Nothing in <span className="code-inline">runs/</span> yet.
                <p className="empty-state__hint">Artifacts appear after your first completed training.</p>
              </div>
            ) : (
              <>
                <div className="table-wrap" tabIndex={0} role="region" aria-label="Run history, rows selectable">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th scope="col">Name</th>
                        <th scope="col">Folder</th>
                        <th scope="col">LightGCN val AUC</th>
                        <th scope="col">Hybrid val AUC</th>
                        <th scope="col">Metrics</th>
                        <th scope="col">Delete</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map((h) => (
                        <tr
                          key={h.run_id}
                          className={`data-table__clickable${selectedRun === h.run_id ? " data-table__selected" : ""}`}
                          onClick={() => void onSelectRun(h.run_id)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              void onSelectRun(h.run_id);
                            }
                          }}
                          tabIndex={0}
                          role="button"
                          aria-pressed={selectedRun === h.run_id}
                          aria-label={`Run ${h.experiment_name?.trim() || h.run_id}, folder ${h.run_id}, select for comparison`}
                        >
                          <td className="strong">{h.experiment_name?.trim() || "—"}</td>
                          <td className="mono">{h.run_id}</td>
                          <td className="mono">{h.best_lightgcn_auc?.toFixed(4) ?? "—"}</td>
                          <td className="mono">{h.best_hybrid_auc?.toFixed(4) ?? "—"}</td>
                          <td>
                            <span className={`yes-no ${h.has_metrics ? "yes-no--yes" : "yes-no--no"}`}>
                              {h.has_metrics ? "Available" : "—"}
                            </span>
                          </td>
                          <td onClick={(e) => e.stopPropagation()}>
                            <button
                              type="button"
                              className="btn btn--danger btn--compact"
                              disabled={deletingRunId === h.run_id}
                              aria-label={`Delete run folder ${h.run_id}`}
                              onClick={(e) => {
                                e.stopPropagation();
                                if (
                                  !window.confirm(
                                    `Delete the run folder "${h.run_id}" and all files inside? This cannot be undone.`,
                                  )
                                ) {
                                  return;
                                }
                                void removeHistoryRun(h.run_id);
                              }}
                            >
                              {deletingRunId === h.run_id ? "…" : "Delete"}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {selectedRun && comparative && comparative.length > 0 && (
                  <div className="subpanel">
                    <h3 className="subpanel__title">Model comparison · {selectedRunMeta.title}</h3>
                    {selectedRunMeta.folder && (
                      <p
                        className="subpanel__hint mono"
                        style={{ margin: "0 0 0.75rem", fontSize: "0.78rem", color: "var(--text-tertiary)" }}
                      >
                        {selectedRunMeta.folder}
                      </p>
                    )}
                    <p className="subpanel__hint" style={{ margin: "0 0 0.75rem", fontSize: "0.82rem", color: "var(--text-tertiary)" }}>
                      Val metrics at best-AUC epoch, sampled ranking (val / test), and hybrid minus LightGCN where both exist.
                      Scroll horizontally if the table is wide.
                    </p>
                    <div
                      className="table-wrap table-wrap--wide"
                      tabIndex={0}
                      role="region"
                      aria-label="Comparative metrics"
                      style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}
                    >
                      <table className="data-table data-table--sticky-first-col">
                        <thead>
                          <tr>
                            {Object.keys(comparative[0]).map((k, colIdx) => (
                              <th key={k} scope="col" className={colIdx === 0 ? "data-table__sticky-head" : undefined}>
                                {k}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {comparative.map((row, i) => (
                            <tr key={i}>
                              {Object.entries(row).map(([k, v], j) => (
                                <td
                                  key={k}
                                  className={
                                    j === 0 ? "data-table__sticky-cell mono data-table__cell-model" : "mono"
                                  }
                                >
                                  {v === null || v === undefined ? "—" : String(v)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {selectedRun && comparative === null && (
                  <div className="subpanel" aria-busy="true">
                    <div className="skeleton skeleton-line" style={{ maxWidth: "100%" }} />
                    <div className="skeleton skeleton-line" style={{ maxWidth: "80%", marginTop: "0.5rem" }} />
                  </div>
                )}

                {selectedRun && comparative && comparative.length === 0 && (
                  <p className="empty-state" style={{ padding: "1.25rem 0 0", textAlign: "left" }}>
                    No comparable metrics for this run (empty or timing-only{" "}
                    <span className="code-inline">metrics.csv</span>). Run a full experiment, or ensure the CSV contains
                    validation rows (<span className="code-inline">split=val</span>) and/or ranking rows (
                    <span className="code-inline">val_ranking</span> / <span className="code-inline">test_ranking</span>
                    ). The API builds <span className="code-inline">full_model_comparative.csv</span> automatically when
                    you open this panel.
                  </p>
                )}
              </>
            )}
          </div>
        </section>

        <footer className="app-footer">
          <span>Hybrid quantum–classical GNN · research dashboard</span>
          <a href="/docs" target="_blank" rel="noreferrer">
            OpenAPI docs
          </a>
        </footer>
          </>
        ) : null}
      </div>
    </div>
  );
}
