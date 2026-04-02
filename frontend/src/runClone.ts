/** Strip run artifacts from run_config.json before cloning into the Run experiment form. */

const SKIP_KEYS = new Set(["device_info", "quantum_backend_info"]);

function isPlainCloneableValue(v: unknown): boolean {
  if (v === null || typeof v === "string" || typeof v === "number" || typeof v === "boolean") return true;
  if (Array.isArray(v)) return v.every((x) => typeof x === "number" || typeof x === "string" || typeof x === "boolean");
  return false;
}

export function prepareRunConfigForFormClone(raw: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (SKIP_KEYS.has(k)) continue;
    if (!isPlainCloneableValue(v)) continue;
    out[k] = v;
  }
  out.save_dir = "";
  const en = out.experiment_name;
  if (typeof en === "string" && en.trim()) {
    out.experiment_name = `${en.trim()} (retrain)`;
  }
  return out;
}
