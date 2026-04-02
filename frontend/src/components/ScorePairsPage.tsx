import { useEffect, useMemo, useState } from "react";
import { listHistory, scorePairsOnRun, type RunSummary, type ScorePairsResponse } from "../api";

type Props = {
  onBack: () => void;
};

export function ScorePairsPage({ onBack }: Props) {
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [runId, setRunId] = useState("");
  const [pairsText, setPairsText] = useState("0 1\n0 2");
  const [microBs, setMicroBs] = useState("256");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<ScorePairsResponse | null>(null);

  const hybridRuns = useMemo(() => history.filter((h) => h.has_hybrid_checkpoint), [history]);

  async function loadHistory() {
    setLoadingList(true);
    setErr(null);
    try {
      const rows = await listHistory();
      setHistory(rows);
      if (!runId && rows.length) {
        const first = rows.find((h) => h.has_hybrid_checkpoint && h.has_graph_context);
        if (first) setRunId(first.run_id);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load runs");
    } finally {
      setLoadingList(false);
    }
  }

  useEffect(() => {
    void loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount only
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setResult(null);
    if (!runId.trim()) {
      setErr("Select a trained run.");
      return;
    }
    const selected = history.find((h) => h.run_id === runId);
    if (!selected?.has_graph_context) {
      setErr(
        "This run has no graph_context.npz. Train again with the current code so the training graph is saved for inference.",
      );
      return;
    }
    if (!selected?.has_hybrid_checkpoint) {
      setErr("This run has no hyb_best.pt.");
      return;
    }
    const mb = parseInt(microBs, 10);
    if (Number.isNaN(mb) || mb < 1) {
      setErr("micro_bs must be a positive integer.");
      return;
    }
    setBusy(true);
    try {
      const out = await scorePairsOnRun(runId.trim(), { pairs_text: pairsText, micro_bs: mb });
      setResult(out);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Scoring failed");
    } finally {
      setBusy(false);
    }
  }

  const lines = pairsText
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));

  return (
    <div className="score-pairs">
      <header className="score-pairs__hero">
        <button type="button" className="btn btn--secondary" onClick={onBack}>
          ← Back to lab
        </button>
        <div>
          <h1 className="score-pairs__title">Score new pairs</h1>
          <p className="score-pairs__subtitle">
            Run the saved <span className="code-inline">hyb_best.pt</span> forward on (user, item) lines without
            training. IDs must be <strong>0-based</strong> like your <span className="code-inline">train.txt</span>.
            The graph is the exact one from training (saved as{" "}
            <span className="code-inline">graph_context.npz</span> on new runs). Pairs do not need to appear in the
            training graph; they only must be valid indices.
          </p>
        </div>
        <button type="button" className="btn btn--secondary" onClick={() => void loadHistory()} disabled={loadingList}>
          {loadingList ? "…" : "Refresh runs"}
        </button>
      </header>

      {err && (
        <div className="alert" role="alert">
          {err}
        </div>
      )}

      <form className="score-pairs__form card" onSubmit={onSubmit}>
        <div className="card__head">
          <h2 className="card__title">Inference</h2>
        </div>
        <div className="score-pairs__fields">
          <label className="score-pairs__label">
            <span>Trained run</span>
            <select
              className="exp-form__select"
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              disabled={loadingList || hybridRuns.length === 0}
            >
              {hybridRuns.length === 0 ? (
                <option value="">No runs with hyb_best.pt</option>
              ) : (
                hybridRuns.map((h) => (
                  <option key={h.run_id} value={h.run_id} disabled={!h.has_graph_context}>
                    {(h.experiment_name?.trim() || h.run_id) + ` · ${h.run_id}`}
                    {!h.has_graph_context ? " (retrain for graph_context.npz)" : ""}
                  </option>
                ))
              )}
            </select>
          </label>
          <label className="score-pairs__label">
            <span>
              Pairs (one <span className="mono">user item</span> per line)
            </span>
            <textarea
              className="score-pairs__textarea mono"
              value={pairsText}
              onChange={(e) => setPairsText(e.target.value)}
              rows={10}
              spellCheck={false}
              placeholder={"0 42\n3 7"}
            />
            <span className="score-pairs__hint mono">
              {lines.length} non-empty line{lines.length === 1 ? "" : "s"} (max 50k per request)
            </span>
          </label>
          <label className="score-pairs__label">
            <span>Quantum micro-batch</span>
            <input
              className="exp-form__input mono"
              value={microBs}
              onChange={(e) => setMicroBs(e.target.value)}
              inputMode="numeric"
            />
          </label>
        </div>
        <div className="btn-row">
          <button
            type="submit"
            className="btn btn--primary"
            disabled={
              busy ||
              hybridRuns.length === 0 ||
              !history.find((h) => h.run_id === runId && h.has_graph_context)
            }
          >
            {busy ? "Scoring…" : "Compute scores"}
          </button>
        </div>
      </form>

      {result && (
        <section className="score-pairs__result card" aria-label="Scores">
          <div className="card__head">
            <h2 className="card__title">Results</h2>
          </div>
          <p className="score-pairs__meta mono">
            run={result.run_id} · users={result.n_users} · items={result.n_items} · backbone={result.hybrid_backbone} ·
            graph={result.graph_context}
          </p>
          <div className="table-wrap" tabIndex={0}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>user</th>
                  <th>item</th>
                  <th>logit</th>
                </tr>
              </thead>
              <tbody>
                {result.pairs.map(([u, i], idx) => {
                  const s = result.scores[idx];
                  return (
                    <tr key={idx}>
                      <td className="mono">{idx + 1}</td>
                      <td className="mono">{u}</td>
                      <td className="mono">{i}</td>
                      <td className="mono">{typeof s === "number" ? s.toFixed(6) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
