from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data.dataset import dynamic_forcings
from .data.store import Era5Store
from .ensemble import EnsembleConfig, run_ensemble
from .events import Event, EventScores, Region
from .metrics import acc, bootstrap_means, bootstrap_weights, crps_ensemble, ensemble_variance, mse, percentile_interval, rank_counts, significant, skill_horizon
from .models.forecaster import Forecaster


def init_indices(store: Era5Store, start: str, end: str, steps: int, every: int) -> np.ndarray:
    starts = store.valid_starts(store.time_range(start, end), steps)
    return starts[::every]


PER_INIT = ("det_mse", "mean_mse", "persist_mse", "clim_mse", "variance", "crps", "det_acc", "mean_acc", "persist_acc")
ROOTS = {"det_rmse": "det_mse", "mean_rmse": "mean_mse", "persist_rmse": "persist_mse", "clim_rmse": "clim_mse", "spread": "variance"}
MEANS = ("crps", "det_acc", "mean_acc", "persist_acc")
ACC_THRESHOLD = 0.6


def evaluate(
    models: list[Forecaster],
    store: Era5Store,
    start: str,
    end: str,
    steps: int = 40,
    every: int = 8,
    ens: EnsembleConfig = EnsembleConfig(),
    max_inits: int | None = None,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    verbose: bool = True,
    events: list[Event] | None = None,
    regions: list[Region] | None = None,
    event_metrics: bool = True,
    bootstrap: int = 1000,
) -> dict:
    inits = init_indices(store, start, end, steps, every)
    if max_inits:
        inits = inits[:max_inits]
    w = store.latitude_weights()
    C = len(store.names)
    per_init = {k: np.zeros((len(inits), steps + 1, C)) for k in PER_INIT}
    ranks = np.zeros((steps + 1, C, ens.members + 1))
    scores = EventScores(store, ens.members, steps, events, regions, verbose=verbose) if event_metrics else None
    z = store.names.index("z500") if "z500" in store.names else 0
    for n, t in enumerate(inits):
        t = int(t)
        truth = np.array(store.fields[t : t + steps + 1], dtype=np.float32)
        x0 = torch.from_numpy(truth[0]).to(device)
        x0 = models[0].normalize(x0)
        forc = torch.from_numpy(dynamic_forcings(store, np.arange(t, t + steps))).to(device)
        traj = run_ensemble(models, x0, forc, ens, seed=seed + n)
        phys = models[0].denormalize(traj.reshape(-1, *traj.shape[2:])).reshape(traj.shape).cpu().numpy()
        clim = np.stack([store.climatology_at(t + k) for k in range(steps + 1)])
        det = phys[0]
        mean = phys.mean(axis=0)
        persist = np.broadcast_to(truth[0], truth.shape)
        per_init["det_mse"][n] = mse(det, truth, w)
        per_init["mean_mse"][n] = mse(mean, truth, w)
        per_init["persist_mse"][n] = mse(persist, truth, w)
        per_init["clim_mse"][n] = mse(clim, truth, w)
        per_init["variance"][n] = ensemble_variance(phys, w)
        per_init["crps"][n] = crps_ensemble(phys, truth, w)
        per_init["det_acc"][n] = acc(det, truth, clim, w)
        per_init["mean_acc"][n] = acc(mean, truth, clim, w)
        per_init["persist_acc"][n] = acc(persist, truth, clim, w)
        ranks += rank_counts(phys, truth)
        if scores is not None:
            scores.update(t, phys, truth)
        if verbose and (n % 10 == 0 or n == len(inits) - 1):
            done = slice(0, n + 1)
            det_now = np.sqrt(per_init["det_mse"][done, -1, z].mean())
            mean_now = np.sqrt(per_init["mean_mse"][done, -1, z].mean())
            print(f"init {n + 1}/{len(inits)} | {store.names[z]} +{steps * 6 // 24}d rmse control {det_now:.1f} media {mean_now:.1f}", flush=True)
    lead_hours = [6 * i for i in range(steps + 1)]
    block = max(1, math.ceil(steps / max(every, 1)))
    result = {
        "inits": int(len(inits)),
        "range": [start, end],
        "lead_hours": lead_hours,
        "channels": store.names,
        "ensemble": asdict(ens),
        "aggregation": "rmse y dispersion: raiz del promedio sobre inicializaciones (WeatherBench 2); *_por_inicio: promedio de las raices (WeatherBench 1)",
        "rank_histogram_last_lead": {name: (ranks[-1, c] / max(ranks[-1, c].sum(), 1)).round(4).tolist() for c, name in enumerate(store.names)},
        "rank_histogram": (ranks / np.maximum(ranks.sum(-1, keepdims=True), 1)).round(5).tolist(),
    }
    result.update({k: significant(v).tolist() for k, v in aggregate(per_init).items()})
    result["horizon_days"] = horizons(result, lead_hours)
    if bootstrap and len(inits) > 1:
        result["bootstrap"] = {"reps": int(bootstrap), "block": int(block), "level": 0.95}
        result["intervals"] = intervals(per_init, lead_hours, block, bootstrap, seed)
    if scores is not None:
        result["regions"] = {r.name: {"label": r.label, "lat": list(r.lat), "lon": list(r.lon)} for r in scores.regions}
        result["events"] = jsonable(scores.summary())
    result["per_init"] = {"times": store.times[inits].astype("datetime64[s]").astype(np.int64), **per_init}
    return result


def aggregate(per_init: dict, weights: np.ndarray | None = None) -> dict:
    def average(values: np.ndarray) -> np.ndarray:
        if weights is None:
            return values.mean(axis=0) if len(values) else np.zeros(values.shape[1:])
        return bootstrap_means(values, weights)

    out = {name: np.sqrt(average(per_init[source])) for name, source in ROOTS.items()}
    out.update({name: average(per_init[name]) for name in MEANS})
    out["spread_skill"] = out["spread"] / np.maximum(out["mean_rmse"], 1e-9)
    out["mean_gain_pct"] = np.where(out["det_rmse"] > 0, 100.0 * (1.0 - out["mean_rmse"] / np.maximum(out["det_rmse"], 1e-9)), 0.0)
    if weights is None:
        out["det_rmse_por_inicio"] = average(np.sqrt(per_init["det_mse"]))
        out["mean_rmse_por_inicio"] = average(np.sqrt(per_init["mean_mse"]))
    return out


def horizons(curves: dict, lead_hours: list[int], threshold: float = ACC_THRESHOLD) -> dict:
    out = {"threshold": threshold}
    for key in ("det_acc", "mean_acc"):
        table = np.asarray(curves[key], dtype=np.float64)
        out[key] = [skill_horizon(lead_hours, table[:, c], threshold) for c in range(table.shape[1])]
    return out


def intervals(per_init: dict, lead_hours: list[int], block: int, reps: int, seed: int = 0, level: float = 0.95) -> dict:
    n = len(per_init["crps"])
    samples = aggregate(per_init, bootstrap_weights(n, block, reps, seed))
    out = {}
    for key, values in samples.items():
        lo, hi = percentile_interval(values, level)
        out[key] = {"lo": significant(lo).tolist(), "hi": significant(hi).tolist()}
    out["horizon_days"] = {}
    for key in ("det_acc", "mean_acc"):
        days = np.array([[skill_horizon(lead_hours, rep[:, c], ACC_THRESHOLD) for c in range(rep.shape[1])] for rep in samples[key]])
        days = np.minimum(days, lead_hours[-1] / 24.0)
        lo, hi = percentile_interval(days, level)
        out["horizon_days"][key] = {"lo": lo.tolist(), "hi": hi.tolist()}
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return float(f"{v:.6g}") if math.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


EVENT_COLUMNS = ("base_rate", "brier", "brier_fair", "brier_clim", "bss", "bss_fair", "reliability", "resolution", "uncertainty", "auc")


def _cell(x) -> str:
    return "" if x is None else str(x)


def event_table(result: dict) -> list[str]:
    lines = ["evento,region,lead_h," + ",".join(EVENT_COLUMNS)]
    for name, event in result.get("events", {}).items():
        for region, scores in event["regions"].items():
            for i, lead in enumerate(result["lead_hours"]):
                if lead == 0:
                    continue
                lines.append(",".join([name, region, str(lead)] + [_cell(scores[k][i]) for k in EVENT_COLUMNS]))
    return lines


def to_json(x, level: int = 0) -> str:
    if isinstance(x, dict) and x:
        pad = "  " * (level + 1)
        items = [f"{pad}{json.dumps(str(k))}: {to_json(v, level + 1)}" for k, v in x.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    return json.dumps(x, separators=(", ", ": "))


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.write_text(to_json(jsonable(obj)) + "\n", encoding="utf-8")
    return path


def save_report(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    per_init = result.pop("per_init", None)
    if per_init is not None:
        np.savez_compressed(out / "por_inicio.npz", **per_init)
    write_json(out / "metrics.json", result)
    lines = ["variable,lead_h,det_rmse,mean_rmse,persist_rmse,clim_rmse,crps,spread,spread_skill,det_acc,mean_acc"]
    for c, name in enumerate(result["channels"]):
        for i, lead in enumerate(result["lead_hours"]):
            if lead == 0:
                continue
            lines.append(
                ",".join(
                    str(x)
                    for x in (
                        name,
                        lead,
                        result["det_rmse"][i][c],
                        result["mean_rmse"][i][c],
                        result["persist_rmse"][i][c],
                        result["clim_rmse"][i][c],
                        result["crps"][i][c],
                        result["spread"][i][c],
                        result["spread_skill"][i][c],
                        result["det_acc"][i][c],
                        result["mean_acc"][i][c] if "mean_acc" in result else "",
                    )
                )
            )
    (out / "metrics.csv").write_text("\n".join(lines))
    if "events" in result:
        (out / "eventos.csv").write_text("\n".join(event_table(result)) + "\n", encoding="utf-8")
    return out
