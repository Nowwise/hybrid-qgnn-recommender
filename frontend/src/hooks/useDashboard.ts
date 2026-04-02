import { useCallback, useEffect, useState } from "react";
import {
  cancelExperimentJob,
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
      setJobs(await listJobs());
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
    anyDatasetReady,
    refresh,
    runExperiment,
    cancelRun,
    onSelectRun,
  };
}
