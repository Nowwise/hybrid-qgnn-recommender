import { useCallback, useEffect, useState } from "react";
import {
  cancelExperimentJob,
  clearExperimentHistory,
  deleteHistoryRun,
  getComparative,
  getDatasetStatus,
  getExperimentPresets,
  getHealth,
  getJob,
  listHistory,
  listJobs,
  startExperiment,
  type DatasetStatus,
  type ExperimentPresets,
  type JobPublic,
  type RunSummary,
} from "../api";

export function useDashboard() {
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [datasets, setDatasets] = useState<DatasetStatus[]>([]);
  const [presets, setPresets] = useState<ExperimentPresets | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [jobs, setJobs] = useState<JobPublic[]>([]);
  const [activeJob, setActiveJob] = useState<JobPublic | null>(null);
  const [comparative, setComparative] = useState<Record<string, string | number | null>[] | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [clearingHistory, setClearingHistory] = useState(false);
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setErr(null);
    try {
      await getHealth();
      setApiOk(true);
    } catch {
      setApiOk(false);
    }
    try {
      const ds = await getDatasetStatus();
      setDatasets(ds.datasets);
      setPresets(await getExperimentPresets());
      setHistory(await listHistory());
      const jList = await listJobs();
      setJobs(jList);
      setActiveJob((prev) => {
        if (prev) {
          const updated = jList.find((j) => j.id === prev.id);
          if (updated) return updated;
          return null;
        }
        const alive = jList.find((j) => j.status === "running" || j.status === "queued");
        return alive ?? null;
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    const id = activeJob?.id;
    if (!id || activeJob.status === "completed" || activeJob.status === "failed" || activeJob.status === "cancelled")
      return;
    const t = setInterval(async () => {
      try {
        const j = await getJob(id);
        setActiveJob(j);
        if (j.status === "completed" || j.status === "failed" || j.status === "cancelled") {
          void refresh();
        }
      } catch {
        /* ignore poll errors */
      }
    }, 800);
    return () => clearInterval(t);
  }, [activeJob?.id, activeJob?.status, refresh]);

  const anyDatasetReady = datasets.some((d) => d.train_txt && d.test_txt);

  const runExperiment = async (body: Record<string, unknown>) => {
    setErr(null);
    setStarting(true);
    try {
      const j = await startExperiment(body);
      setActiveJob(j);
      setJobs(await listJobs());
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Start failed");
    } finally {
      setStarting(false);
    }
  };

  const cancelRun = async (jobId: string) => {
    setErr(null);
    setCancelling(true);
    try {
      const j = await cancelExperimentJob(jobId);
      setActiveJob(j);
      void refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Cancel failed");
    } finally {
      setCancelling(false);
    }
  };

  const onSelectRun = async (runId: string) => {
    setSelectedRun(runId);
    setComparative(null);
    try {
      setComparative(await getComparative(runId));
    } catch {
      setComparative([]);
    }
  };

  const openJob = useCallback(async (jobId: string) => {
    setErr(null);
    try {
      const j = await getJob(jobId);
      setActiveJob(j);
      requestAnimationFrame(() => {
        document.getElementById("exp-sec-live")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not load job");
    }
  }, []);

  const dismissActiveJob = useCallback(() => {
    setActiveJob(null);
  }, []);

  const wipeHistory = useCallback(async () => {
    setErr(null);
    setClearingHistory(true);
    try {
      await clearExperimentHistory();
      setActiveJob(null);
      setSelectedRun(null);
      setComparative(null);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Clear history failed");
    } finally {
      setClearingHistory(false);
    }
  }, [refresh]);

  const removeHistoryRun = useCallback(
    async (runId: string) => {
      setErr(null);
      setDeletingRunId(runId);
      try {
        await deleteHistoryRun(runId);
        setSelectedRun((prev) => (prev === runId ? null : prev));
        if (selectedRun === runId) {
          setComparative(null);
        }
        setActiveJob((prev) => {
          if (!prev?.save_dir) return prev;
          const norm = prev.save_dir.trim().replace(/\\/g, "/").replace(/^\.\//, "");
          const prefix = `runs/${runId}`;
          if (norm === prefix || norm.startsWith(`${prefix}/`)) return null;
          return prev;
        });
        await refresh();
      } catch (e) {
        setErr(e instanceof Error ? e.message : "Delete run failed");
      } finally {
        setDeletingRunId(null);
      }
    },
    [refresh, selectedRun],
  );

  return {
    apiOk,
    datasets,
    presets,
    history,
    jobs,
    activeJob,
    comparative,
    selectedRun,
    err,
    starting,
    cancelling,
    clearingHistory,
    deletingRunId,
    anyDatasetReady,
    refresh,
    runExperiment,
    cancelRun,
    onSelectRun,
    openJob,
    dismissActiveJob,
    wipeHistory,
    removeHistoryRun,
  };
}
