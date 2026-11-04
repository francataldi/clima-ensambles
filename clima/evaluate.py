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
from .metrics import acc, crps_ensemble, ensemble_spread, rank_counts, rmse
from .models.forecaster import Forecaster


def init_indices(store: Era5Store, start: str, end: str, steps: int, every: int) -> np.ndarray:
    starts = store.valid_starts(store.time_range(start, end), steps)
    return starts[::every]


def evaluate(
    models: list[Forecaster],
    store: Era5Store,
    start: str,
    end: str,
    steps: int = 20,
    every: int = 8,
    ens: EnsembleConfig = EnsembleConfig(),
    max_inits: int | None = None,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    verbose: bool = True,
    events: list[Event] | None = None,
    regions: list[Region] | None = None,
    event_metrics: bool = True,
) -> dict:
    inits = init_indices(store, start, end, steps, every)
    if max_inits:
        inits = inits[:max_inits]
    w = store.latitude_weights()
    C = len(store.names)
    acc_keys = ("det_rmse", "mean_rmse", "persist_rmse", "clim_rmse", "crps", "spread", "det_acc", "mean_acc", "persist_acc")
    sums = {k: np.zeros((steps + 1, C)) for k in acc_keys}
    ranks = np.zeros((steps + 1, C, ens.members + 1))
    scores = EventScores(store, ens.members, steps, events, regions, verbose=verbose) if event_metrics else None
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
        sums["det_rmse"] += rmse(det, truth, w)
        sums["mean_rmse"] += rmse(mean, truth, w)
        sums["persist_rmse"] += rmse(persist, truth, w)
        sums["clim_rmse"] += rmse(clim, truth, w)
        sums["crps"] += crps_ensemble(phys, truth, w)
        sums["spread"] += ensemble_spread(phys, w)
        sums["det_acc"] += acc(det, truth, clim, w)
        sums["mean_acc"] += acc(mean, truth, clim, w)
        sums["persist_acc"] += acc(persist, truth, clim, w)
        ranks += rank_counts(phys, truth)
        if scores is not None:
            scores.update(t, phys, truth)
        if verbose and (n % 10 == 0 or n == len(inits) - 1):
            z = store.names.index("z500") if "z500" in store.names else 0
            print(f"init {n + 1}/{len(inits)} | z500 +5d rmse det {sums['det_rmse'][-1, z] / (n + 1):.1f} media {sums['mean_rmse'][-1, z] / (n + 1):.1f}", flush=True)
    k = max(len(inits), 1)
    result = {
        "inits": int(len(inits)),
        "range": [start, end],
        "lead_hours": [6 * i for i in range(steps + 1)],
        "channels": store.names,
        "ensemble": asdict(ens),
        "rank_histogram_last_lead": {name: (ranks[-1, c] / max(ranks[-1, c].sum(), 1)).round(4).tolist() for c, name in enumerate(store.names)},
        "rank_histogram": (ranks / np.maximum(ranks.sum(-1, keepdims=True), 1)).round(5).tolist(),
    }
    for key, value in sums.items():
        result[key] = (value / k).round(5).tolist()
    result["spread_skill"] = (np.asarray(result["spread"]) / np.maximum(np.asarray(result["mean_rmse"]), 1e-9)).round(4).tolist()
    if scores is not None:
        result["regions"] = {r.name: {"label": r.label, "lat": list(r.lat), "lon": list(r.lon)} for r in scores.regions}
        result["events"] = jsonable(scores.summary())
    return result


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
    write_json(out / "metrics.json", result)
    lines = ["variable,lead_h,det_rmse,mean_rmse,persist_rmse,clim_rmse,crps,spread,spread_skill,det_acc"]
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
                    )
                )
            )
    (out / "metrics.csv").write_text("\n".join(lines))
    if "events" in result:
        (out / "eventos.csv").write_text("\n".join(event_table(result)) + "\n", encoding="utf-8")
    return out
