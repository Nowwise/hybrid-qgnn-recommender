import { useCallback, useEffect, useState } from "react";
import {
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
  const [dataset, setDataset] = useState<DatasetStatus | null>(null);
  const [presets, setPresets] = useState<ExperimentPresets | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [jobs, setJobs] = useState<JobPublic[]>([]);
  const [activeJob, setActiveJob] = useState<JobPublic | null>(null);
  const [comparative, setComparative] = useState<Record<string, string | number | null>[] | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async () => {
    setErr(null);
    try {
      await getHealth();
      setApiOk(true);
    } catch {
      setApiOk(false);
    }
    try {
      setDataset(await getDatasetStatus());
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
    if (!id || activeJob.status === "completed" || activeJob.status === "failed") return;
    const t = setInterval(async () => {
      try {
        const j = await getJob(id);
        setActiveJob(j);
        if (j.status === "completed" || j.status === "failed") {
          void refresh();
        }
      } catch {
        /* ignore poll errors */
      }
    }, 800);
    return () => clearInterval(t);
  }, [activeJob?.id, activeJob?.status, refresh]);

  const datasetReady = !!(dataset?.train_txt && dataset?.test_txt);

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
    dataset,
    presets,
    history,
    jobs,
    activeJob,
    comparative,
    selectedRun,
    err,
    starting,
    datasetReady,
    refresh,
    runExperiment,
    onSelectRun,
  };
}
