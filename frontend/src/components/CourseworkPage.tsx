import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getComparative,
  getRunConfig,
  getRunPhaseTimings,
  getRunSummaryText,
  historyDownloadUrl,
  historyPlotUrl,
  type RunSummary,
} from "../api";

const CONFIG_KEYS: readonly string[] = [
  "experiment_name",
  "data_dir",
  "seed",
  "training_loss",
  "hybrid_backbone",
  "epochs_lg",
  "epochs_hyb",
  "d",
  "K",
  "q",
  "L",
  "eval_ranking",
  "ranking_ks",
  "ranking_max_users",
  "ranking_negatives",
  "eval_test_ranking",
  "eval_hybrid_ablation",
  "early_stopping",
  "early_stopping_monitor",
  "early_stopping_patience",
  "early_stopping_min_delta",
  "p_quantum_start",
  "p_quantum_end",
  "backend",
  "batch_size",
  "micro_bs",
  "live_plots",
  "log_phase_timings",
  "val_ratio",
  "max_users",
  "max_pos_per_user",
  "neg_per_pos",
];

function fmtConfigValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

type PhaseRow = { phase?: string; seconds?: number };

function isPhaseRowArray(x: unknown): x is PhaseRow[] {
  return Array.isArray(x) && x.every((e) => e !== null && typeof e === "object");
}

export function CourseworkPage({
  history,
  onBack,
}: {
  history: RunSummary[];
  onBack: () => void;
}) {
  const [runId, setRunId] = useState<string | null>(null);
  const [cfg, setCfg] = useState<Record<string, unknown> | null>(null);
  const [summary, setSummary] = useState<{ text: string | null; path: string } | null>(null);
  const [phases, setPhases] = useState<unknown>(null);
  const [phasesPath, setPhasesPath] = useState<string>("");
  const [comparative, setComparative] = useState<Record<string, string | number | null>[] | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [plotBust, setPlotBust] = useState(0);

  useEffect(() => {
    if (!runId && history.length > 0) {
      setRunId(history[0].run_id);
    }
  }, [history, runId]);

  const loadRun = useCallback(async (id: string) => {
    setLoading(true);
    setLoadErr(null);
    setCfg(null);
    setSummary(null);
    setPhases(null);
    setPhasesPath("");
    setComparative(null);
    setPlotBust(Date.now());
    try {
      const settled = await Promise.allSettled([
        getRunConfig(id),
        getRunSummaryText(id),
        getRunPhaseTimings(id),
        getComparative(id),
      ]);
      const [cRes, sRes, pRes, compRes] = settled;
      if (cRes.status === "fulfilled") setCfg(cRes.value);
      else setCfg(null);
      if (sRes.status === "fulfilled") {
        setSummary({ text: sRes.value.text, path: sRes.value.relative_path });
      } else {
        setSummary(null);
      }
      if (pRes.status === "fulfilled") {
        setPhases(pRes.value.phases);
        setPhasesPath(pRes.value.relative_path);
      } else {
        setPhases(null);
        setPhasesPath("");
      }
      if (compRes.status === "fulfilled") setComparative(compRes.value);
      else setComparative([]);
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : "Failed to load run");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!runId) return;
    void loadRun(runId);
  }, [runId, loadRun]);

  const meta = useMemo(() => history.find((h) => h.run_id === runId), [history, runId]);

  const configRows = useMemo(() => {
    if (!cfg) return [];
    const rows: { key: string; value: string }[] = [];
    for (const k of CONFIG_KEYS) {
      if (k in cfg) rows.push({ key: k, value: fmtConfigValue(cfg[k]) });
    }
    if (cfg.device_info != null) {
      rows.push({ key: "device_info", value: fmtConfigValue(cfg.device_info) });
    }
    if (cfg.quantum_backend_info != null) {
      rows.push({ key: "quantum_backend_info", value: fmtConfigValue(cfg.quantum_backend_info) });
    }
    return rows;
  }, [cfg]);

  const phaseRows: PhaseRow[] = useMemo(() => {
    if (isPhaseRowArray(phases)) return phases;
    return [];
  }, [phases]);

  return (
    <div className="coursework-page">
      <div className="coursework-page__head">
        <button type="button" className="btn btn--secondary" onClick={onBack}>
          ← Back
        </button>
        <div>
          <h2 className="coursework-page__title">Coursework &amp; report pack</h2>
          <p className="coursework-page__lede">
            One place for the run you are submitting: configuration, downloadable artifacts, comparison table, training
            dashboard image, and what each metric means for your write-up.
          </p>
        </div>
      </div>

      <section className="card coursework-card" aria-labelledby="cw-select-heading">
        <h3 id="cw-select-heading" className="card__title">
          Select run
        </h3>
        {history.length === 0 ? (
          <p className="empty-state" style={{ margin: 0 }}>
            No runs under <span className="code-inline">runs/</span> yet. Complete an experiment on the Lab page first.
          </p>
        ) : (
          <div className="coursework-page__select-row">
            <label className="coursework-page__label" htmlFor="cw-run-select">
              Folder
            </label>
            <select
              id="cw-run-select"
              className="coursework-page__select"
              value={runId ?? ""}
              onChange={(e) => setRunId(e.target.value || null)}
            >
              {history.map((h) => (
                <option key={h.run_id} value={h.run_id}>
                  {(h.experiment_name?.trim() || h.run_id) + ` · ${h.run_id}`}
                </option>
              ))}
            </select>
          </div>
        )}
        {meta && (
          <ul className="coursework-page__meta" aria-label="Run snapshot">
            <li>
              <span className="coursework-page__meta-k">LightGCN val AUC (from summary scan)</span>
              <span className="mono">{meta.best_lightgcn_auc?.toFixed(4) ?? "—"}</span>
            </li>
            <li>
              <span className="coursework-page__meta-k">Hybrid val AUC</span>
              <span className="mono">{meta.best_hybrid_auc?.toFixed(4) ?? "—"}</span>
            </li>
            <li>
              <span className="coursework-page__meta-k">metrics.csv</span>
              <span className="mono">{meta.has_metrics ? "present" : "missing"}</span>
            </li>
            <li>
              <span className="coursework-page__meta-k">summary.txt</span>
              <span className="mono">{meta.has_summary ? "present" : "missing"}</span>
            </li>
          </ul>
        )}
      </section>

      {loadErr && (
        <div className="alert" role="alert">
          {loadErr}
        </div>
      )}

      {runId && (
        <>
          <section className="card coursework-card" aria-labelledby="cw-guide-heading">
            <h3 id="cw-guide-heading" className="card__title">
              What to include in your report
            </h3>
            <ol className="coursework-page__guide">
              <li>
                <strong>Setup:</strong> dataset path, train/val split (implicit feedback), baselines trained, hybrid backbone
                and quantum settings — all listed under <em>Run configuration</em> below.
              </li>
              <li>
                <strong>Pairwise validation:</strong> ROC-AUC (and optional RMSE/MAE on probabilities) from{" "}
                <span className="code-inline">split=val</span> in <span className="code-inline">metrics.csv</span>. This
                scores labeled user–item pairs, not full top-<em>K</em> retrieval over the catalog.
              </li>
              <li>
                <strong>Ranking check:</strong> sampled <span className="code-inline">Recall@K</span> /{" "}
                <span className="code-inline">NDCG@K</span> from <span className="code-inline">val_ranking</span> (and
                optional <span className="code-inline">test_ranking</span>): one held-out positive per user-query plus random
                negatives. Report <em>K</em> and the number of sampled users.
              </li>
              <li>
                <strong>Figures:</strong> use <span className="code-inline">plots/training_dashboard.png</span> for curves and
                ranking bar panels (after <span className="code-inline">eval_ranking</span>).
              </li>
              <li>
                <strong>Reproducibility:</strong> cite <span className="code-inline">seed</span> from{" "}
                <span className="code-inline">run_config.json</span>; zip the downloadable files for an appendix if required.
              </li>
            </ol>
          </section>

          <section className="card coursework-card" aria-labelledby="cw-artifacts-heading">
            <h3 id="cw-artifacts-heading" className="card__title">
              Downloadable artifacts
            </h3>
            <p className="coursework-page__hint">
              Paths are relative to the project root on disk: <span className="mono">runs/{runId}/…</span>
            </p>
            <div className="coursework-page__downloads">
              <a className="btn btn--secondary" href={historyDownloadUrl(runId, "metrics.csv")} download>
                metrics.csv
              </a>
              <a className="btn btn--secondary" href={historyDownloadUrl(runId, "full_model_comparative.csv")} download>
                full_model_comparative.csv
              </a>
              <a className="btn btn--secondary" href={historyDownloadUrl(runId, "run_config.json")} download>
                run_config.json
              </a>
              <a className="btn btn--secondary" href={historyDownloadUrl(runId, "summary.txt")} download>
                summary.txt
              </a>
            </div>
            <p className="coursework-page__hint">
              If a download 404s, that file was not produced for this run (e.g. ranking disabled or run incomplete).
            </p>
          </section>

          <section className="card coursework-card" aria-labelledby="cw-cfg-heading">
            <h3 id="cw-cfg-heading" className="card__title">
              Run configuration
            </h3>
            {loading && !cfg ? (
              <div className="skeleton skeleton-line" style={{ maxWidth: "100%" }} />
            ) : cfg ? (
              <div className="table-wrap" tabIndex={0}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Field</th>
                      <th scope="col">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {configRows.map((r) => (
                      <tr key={r.key}>
                        <td className="mono">{r.key}</td>
                        <td className="mono coursework-page__cfg-val">{r.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="empty-state">No configuration loaded.</p>
            )}
          </section>

          {summary?.text != null && (
            <section className="card coursework-card" aria-labelledby="cw-sum-heading">
              <h3 id="cw-sum-heading" className="card__title">
                Text summary · <span className="mono">{summary.path}</span>
              </h3>
              <pre className="coursework-page__pre">{summary.text}</pre>
            </section>
          )}

          {summary?.text == null && summary && !loading && (
            <section className="card coursework-card" aria-labelledby="cw-sum-missing-heading">
              <h3 id="cw-sum-missing-heading" className="card__title">
                Text summary
              </h3>
              <p className="coursework-page__hint">
                <span className="mono">{summary.path}</span> was not found for this run.
              </p>
            </section>
          )}

          {phaseRows.length > 0 && (
            <section className="card coursework-card" aria-labelledby="cw-phase-heading">
              <h3 id="cw-phase-heading" className="card__title">
                Phase timings · <span className="mono">{phasesPath}</span>
              </h3>
              <div className="table-wrap" tabIndex={0}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Phase</th>
                      <th scope="col">Seconds</th>
                    </tr>
                  </thead>
                  <tbody>
                    {phaseRows.map((row, i) => (
                      <tr key={`${row.phase ?? i}-${i}`}>
                        <td className="mono">{row.phase ?? "—"}</td>
                        <td className="mono">{row.seconds != null ? row.seconds.toFixed(2) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <section className="card coursework-card" aria-labelledby="cw-plot-heading">
            <h3 id="cw-plot-heading" className="card__title">
              Training dashboard
            </h3>
            <p className="coursework-page__hint">
              <span className="mono">plots/training_dashboard.png</span> — validation AUC, training loss, sampled Recall@K /
              NDCG@K when ranking eval ran, hybrid <span className="mono">p_quantum</span>, and a short text summary.
            </p>
            <div className="coursework-page__png-frame">
              <img
                className="coursework-page__png"
                src={historyPlotUrl(runId, "training_dashboard.png", plotBust)}
                alt="Training dashboard for selected run"
                decoding="async"
              />
            </div>
          </section>

          <section className="panel coursework-panel-wide" aria-labelledby="cw-comp-heading">
            <div className="panel__bar" aria-hidden />
            <div className="panel__inner">
              <h3 id="cw-comp-heading" className="panel__title">
                Full model comparison table
              </h3>
              <p className="coursework-page__hint">
                Same wide table as on the Lab page: validation metrics at best-AUC epoch, ranking columns, and deltas where
                applicable.
              </p>
              {loading && comparative === null ? (
                <div className="skeleton skeleton-line" style={{ maxWidth: "100%" }} />
              ) : comparative && comparative.length > 0 ? (
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
                              className={j === 0 ? "data-table__sticky-cell mono data-table__cell-model" : "mono"}
                            >
                              {v === null || v === undefined ? "—" : String(v)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="empty-state" style={{ textAlign: "left", margin: 0 }}>
                  No comparative table for this run. Ensure <span className="code-inline">metrics.csv</span> contains
                  validation and/or ranking rows, then re-open this page or re-run training.
                </p>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
