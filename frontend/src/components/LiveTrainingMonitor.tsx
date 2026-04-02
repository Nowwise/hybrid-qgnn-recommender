import { useEffect, useMemo, useRef, useState } from "react";
import {
  type LiveMetricsPayload,
  type LiveRankingRow,
  type JobPublic,
  getJobLiveMetrics,
  jobPlotUrl,
} from "../api";

const COLORS: Record<string, string> = {
  LightGCN: "#e8a838",
  HybridQGNN: "#8b7cf8",
};

/** Table order for validation classification metrics (matches metrics.csv). */
const VAL_METRIC_ORDER = ["AUC", "RMSE", "MAE", "MSE", "MAPE", "WMAPE"] as const;

function fmtAuc(n: number): string {
  return Number.isFinite(n) ? n.toFixed(4) : "—";
}

function fmtTrainLoss(n: number): string {
  return Number.isFinite(n) ? n.toFixed(5) : "—";
}

function fmtErrorMetric(n: number): string {
  if (!Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a >= 1e-3 && a < 1e4) return n.toFixed(5);
  return n.toExponential(3);
}

function fmtPercentish(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(3);
}

function fmtPQuantum(n: number): string {
  return Number.isFinite(n) ? n.toFixed(4) : "—";
}

function fmtRankingValue(n: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a >= 0.0001 && a < 1000) return n.toPrecision(5);
  return n.toExponential(3);
}

function lastFinitePoint(series: Record<string, { epoch: number; value: number }[]>, model: string) {
  const pts = series[model];
  if (!pts?.length) return null;
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    if (typeof p.value === "number" && Number.isFinite(p.value)) return p;
  }
  return null;
}

function useResolvedSaveDir(job: JobPublic | null): string | null {
  return useMemo(() => {
    if (!job) return null;
    const r = job.result;
    const fromResult =
      r && typeof r === "object" && r !== null && typeof (r as { save_dir?: unknown }).save_dir === "string"
        ? String((r as { save_dir: string }).save_dir)
        : null;
    if (job.save_dir && String(job.save_dir).trim()) return String(job.save_dir);
    if (fromResult) return fromResult;
    const batch = r as { results?: { save_dir?: string }[] } | null;
    if (batch && Array.isArray(batch.results) && batch.results[0]?.save_dir) {
      return batch.results[0].save_dir!;
    }
    return null;
  }, [job]);
}

function SparkLines({
  title,
  series,
  yLabel,
  formatLegend,
}: {
  title: string;
  series: Record<string, { epoch: number; value: number }[]>;
  yLabel: string;
  formatLegend?: (epoch: number, value: number) => string;
}) {
  const models = Object.keys(series).filter((k) => (series[k] ?? []).some((p) => Number.isFinite(p.value)));
  if (models.length === 0) {
    return (
      <div className="live-monitor__spark">
        <div className="live-monitor__spark-title">{title}</div>
        <p className="live-monitor__spark-empty">Waiting for data…</p>
      </div>
    );
  }

  const allPts = models.flatMap((m) => series[m].filter((p) => Number.isFinite(p.value)));
  const epochs = allPts.map((p) => p.epoch);
  const emin = Math.min(...epochs);
  const emax = Math.max(...epochs);
  const vals = allPts.map((p) => p.value);
  let vmin = Math.min(...vals);
  let vmax = Math.max(...vals);
  const pad = (vmax - vmin) * 0.06 || 0.02;
  vmin -= pad;
  vmax += pad;

  const W = 360;
  const H = 100;
  const mx = 8;
  const my = 6;
  const sx = (e: number) => mx + ((e - emin) / (emax - emin || 1)) * (W - 2 * mx);
  const sy = (v: number) => H - my - ((v - vmin) / (vmax - vmin || 1)) * (H - 2 * my);

  return (
    <div className="live-monitor__spark">
      <div className="live-monitor__spark-head">
        <span className="live-monitor__spark-title">{title}</span>
        <span className="live-monitor__spark-axis mono">{yLabel}</span>
      </div>
      <svg className="live-monitor__svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
        <rect width={W} height={H} fill="transparent" />
        {models.map((m) => {
          const pts = series[m].filter((p) => Number.isFinite(p.value));
          if (!pts.length) return null;
          const d = pts
            .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.epoch).toFixed(1)},${sy(p.value).toFixed(1)}`)
            .join(" ");
          return (
            <path
              key={m}
              d={d}
              fill="none"
              stroke={COLORS[m] ?? "#94a3b8"}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          );
        })}
      </svg>
      <div className="live-monitor__legend">
        {models.map((m) => {
          const lp = lastFinitePoint(series, m);
          const detail =
            lp && formatLegend ? (
              <span className="live-monitor__legend-detail mono">
                {" "}
                ep {lp.epoch} → {formatLegend(lp.epoch, lp.value)}
              </span>
            ) : null;
          return (
            <span key={m} className="live-monitor__legend-item mono">
              <span className="live-monitor__swatch" style={{ background: COLORS[m] ?? "#888" }} />
              {m}
              {detail}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function PQuantumChart({ pts }: { pts: { epoch: number; value: number }[] }) {
  const clean = pts.filter((p) => Number.isFinite(p.value));
  if (!clean.length) {
    return (
      <div className="live-monitor__spark">
        <div className="live-monitor__spark-title">Hybrid · p_quantum</div>
        <p className="live-monitor__spark-empty">During full hybrid epochs…</p>
      </div>
    );
  }
  const last = clean[clean.length - 1];
  const epochs = clean.map((p) => p.epoch);
  const emin = Math.min(...epochs);
  const emax = Math.max(...epochs);
  const W = 360;
  const H = 72;
  const mx = 8;
  const sx = (e: number) => mx + ((e - emin) / (emax - emin || 1)) * (W - 2 * mx);
  const sy = (v: number) => 6 + (1 - v) * 56;
  const d = clean.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.epoch).toFixed(1)},${sy(p.value).toFixed(1)}`).join(" ");
  return (
    <div className="live-monitor__spark">
      <div className="live-monitor__spark-head">
        <span className="live-monitor__spark-title">Hybrid · p_quantum</span>
        <span className="live-monitor__spark-axis mono">
          now {fmtPQuantum(last.value)} @ ep {last.epoch}
        </span>
      </div>
      <svg className="live-monitor__svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
        <path d={d} fill="none" stroke="#34d399" strokeWidth={2} strokeLinecap="round" />
      </svg>
    </div>
  );
}

function LiveMetricSnapshot({ live }: { live: LiveMetricsPayload | null }) {
  const latest = live?.latest_val ?? {};
  const best = live?.best_val_auc ?? {};
  const models = Array.from(new Set([...Object.keys(latest), ...Object.keys(best)])).sort();
  if (!models.length) return null;

  return (
    <div className="live-monitor__snapshot live-monitor__span-full">
      <div className="live-monitor__snapshot-head">
        <span className="live-monitor__spark-title">Validation numbers</span>
        <span className="live-monitor__snapshot-meta mono">
          metrics rows: {live?.row_count ?? 0}
        </span>
      </div>
      <p className="live-monitor__snapshot-hint">
        Latest row uses the highest logged validation epoch per model. Best AUC is the max over all epochs so far.
      </p>
      <div className="live-monitor__cards">
        {models.map((model) => {
          const snap = latest[model];
          const b = best[model];
          const metrics = snap?.metrics ?? {};
          return (
            <div key={model} className="live-monitor__card">
              <div className="live-monitor__card-title">{model}</div>
              {b && (
                <div className="live-monitor__card-best mono">
                  best val AUC <span className="live-monitor__card-em">{fmtAuc(b.auc)}</span>
                  <span className="live-monitor__card-sub"> @ epoch {b.epoch}</span>
                </div>
              )}
              {snap && (
                <div className="live-monitor__card-latest mono">
                  latest epoch <span className="live-monitor__card-em">{snap.epoch}</span>
                </div>
              )}
              {snap && (
                <dl className="live-monitor__metric-dl">
                  {VAL_METRIC_ORDER.map((key) => {
                    const raw = metrics[key];
                    if (raw === undefined || raw === null) return null;
                    const v = typeof raw === "number" ? raw : Number(raw);
                    if (!Number.isFinite(v)) return null;
                    let formatted: string;
                    if (key === "AUC") formatted = fmtAuc(v);
                    else if (key === "MAPE" || key === "WMAPE") formatted = fmtPercentish(v);
                    else formatted = fmtErrorMetric(v);
                    return (
                      <div key={key} className="live-monitor__metric-row">
                        <dt>{key}</dt>
                        <dd title={String(v)}>{formatted}</dd>
                      </div>
                    );
                  })}
                </dl>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RankingTable({ title, rows }: { title: string; rows: LiveRankingRow[] }) {
  if (!rows.length) return null;
  return (
    <details className="live-monitor__ranking" open={rows.length <= 16}>
      <summary className="live-monitor__ranking-summary">
        {title}
        <span className="live-monitor__ranking-count mono">{rows.length} metrics</span>
      </summary>
      <div className="live-monitor__ranking-scroll">
        <table className="live-monitor__ranking-table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Metric</th>
              <th scope="col">Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.model}-${r.metric}-${i}`}>
                <td className="mono">{r.model}</td>
                <td className="mono">{r.metric}</td>
                <td className="mono live-monitor__ranking-val" title={r.value == null ? "" : String(r.value)}>
                  {fmtRankingValue(r.value)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export function LiveTrainingMonitor({ job }: { job: JobPublic }) {
  const saveDir = useResolvedSaveDir(job);
  const [live, setLive] = useState<LiveMetricsPayload | null>(null);
  const [plotBust, setPlotBust] = useState(0);
  const [plotOk, setPlotOk] = useState(false);
  const lastRowCountRef = useRef(0);
  const pngBumpTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pngEverLoadedRef = useRef(false);

  const active = job.status === "running" || job.status === "queued";

  useEffect(() => {
    if (!saveDir || !job.id) {
      setLive(null);
      lastRowCountRef.current = 0;
      return;
    }
    lastRowCountRef.current = 0;
    pngEverLoadedRef.current = false;
    setPlotBust(0);

    let cancelled = false;
    const clearPngTimer = () => {
      if (pngBumpTimerRef.current) {
        clearTimeout(pngBumpTimerRef.current);
        pngBumpTimerRef.current = null;
      }
    };

    const tick = async () => {
      try {
        const m = await getJobLiveMetrics(job.id);
        if (cancelled) return;
        setLive((prev) => {
          if (!m.ready && prev && prev.row_count > 0) return prev;
          if (prev && m.row_count > 0 && m.row_count < prev.row_count) return prev;
          return m;
        });
        if (m.ready && m.row_count > lastRowCountRef.current) {
          lastRowCountRef.current = m.row_count;
          clearPngTimer();
          pngBumpTimerRef.current = setTimeout(() => {
            pngBumpTimerRef.current = null;
            if (!cancelled) setPlotBust((b) => b + 1);
          }, 500);
        }
      } catch {
        /* keep last good chart data; avoid empty-state flicker */
      }
    };
    void tick();
    if (!active) {
      return () => {
        cancelled = true;
        clearPngTimer();
      };
    }
    const iv = setInterval(() => void tick(), 1500);
    return () => {
      cancelled = true;
      clearPngTimer();
      clearInterval(iv);
    };
  }, [job.id, saveDir, active]);

  useEffect(() => {
    if (active) setPlotOk(false);
  }, [active, job.id]);

  if (!saveDir) return null;

  const pngSrc = jobPlotUrl(job.id, "training_dashboard.png", plotBust + 1);
  const valRmse = live?.val_rmse ?? {};
  const valMae = live?.val_mae ?? {};

  return (
    <div className="live-monitor" aria-label="Live training metrics">
      <div className="live-monitor__head">
        <h3 className="live-monitor__title">Training curves</h3>
        <span className="live-monitor__path mono" title="Run output folder">
          {saveDir}
        </span>
      </div>
      <p className="live-monitor__hint">
        Streamed from <span className="mono">metrics.csv</span> each epoch. High-res figure:{" "}
        <span className="mono">plots/training_dashboard.png</span> (2×3: AUC, loss, sampled Recall@K / NDCG@K,{" "}
        <span className="mono">p_quantum</span>, summary). Ranking panels use <span className="mono">val_ranking</span>{" "}
        after training when <span className="mono">eval_ranking</span> is on. Charts omit non-finite points; tables use
        fixed precision for readability.
      </p>

      <div className="live-monitor__grid">
        <LiveMetricSnapshot live={live} />

        <SparkLines
          title="Validation AUC"
          series={live?.val_auc ?? {}}
          yLabel="AUC"
          formatLegend={(_ep, v) => fmtAuc(v)}
        />
        <SparkLines
          title="Training loss"
          series={live?.train_loss ?? {}}
          yLabel="loss"
          formatLegend={(_ep, v) => fmtTrainLoss(v)}
        />
        <SparkLines
          title="Validation RMSE"
          series={valRmse}
          yLabel="RMSE"
          formatLegend={(_ep, v) => fmtErrorMetric(v)}
        />
        <SparkLines
          title="Validation MAE"
          series={valMae}
          yLabel="MAE"
          formatLegend={(_ep, v) => fmtErrorMetric(v)}
        />
        <PQuantumChart pts={live?.p_quantum ?? []} />

        <div className="live-monitor__png-wrap live-monitor__span-full">
          <div className="live-monitor__spark-title">Saved dashboard (PNG)</div>
          <div className="live-monitor__png-frame">
            <img
              className="live-monitor__png"
              src={pngSrc}
              alt="Training dashboard"
              decoding="async"
              onLoad={() => {
                pngEverLoadedRef.current = true;
                setPlotOk(true);
              }}
              onError={() => {
                if (!pngEverLoadedRef.current) setPlotOk(false);
              }}
            />
          </div>
          {!plotOk && active && (
            <p className="live-monitor__spark-empty">First PNG appears after the first epoch writes metrics.</p>
          )}
        </div>

        <div className="live-monitor__ranking-wrap live-monitor__span-full">
          <RankingTable title="Ranking · validation sample" rows={live?.ranking_val ?? []} />
          <RankingTable title="Ranking · test sample" rows={live?.ranking_test ?? []} />
        </div>
      </div>
    </div>
  );
}
