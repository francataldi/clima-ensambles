from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data.dataset import dynamic_forcings
from .data.store import Era5Store
from .ensemble import EnsembleConfig, run_ensemble
from .metrics import acc, crps_ensemble, ensemble_spread, rank_histogram, rmse
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
) -> dict:
    inits = init_indices(store, start, end, steps, every)
    if max_inits:
        inits = inits[:max_inits]
    w = store.latitude_weights()
    C = len(store.names)
    acc_keys = ("det_rmse", "mean_rmse", "persist_rmse", "clim_rmse", "crps", "spread", "det_acc", "persist_acc")
    sums = {k: np.zeros((steps + 1, C)) for k in acc_keys}
    ranks = np.zeros((len(store.names), ens.members + 1))
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
        sums["persist_acc"] += acc(persist, truth, clim, w)
        for c in range(C):
            ranks[c] += rank_histogram(phys[:, -1, c], truth[-1, c], ens.members + 1)
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
        "rank_histogram_last_lead": {name: (ranks[c] / max(ranks[c].sum(), 1)).round(4).tolist() for c, name in enumerate(store.names)},
    }
    for key, value in sums.items():
        result[key] = (value / k).round(5).tolist()
    result["spread_skill"] = (np.asarray(result["spread"]) / np.maximum(np.asarray(result["mean_rmse"]), 1e-9)).round(4).tolist()
    return result


def save_report(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
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
    return out
