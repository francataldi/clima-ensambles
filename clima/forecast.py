from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data.dataset import forcings_for_times
from .data.store import Era5Store
from .ensemble import EnsembleConfig, run_ensemble
from .metrics import bilinear
from .models.forecaster import Forecaster
from .plots import animate_field, plot_point_plume

BUENOS_AIRES = (-34.6, -58.4)
KELVIN = 273.15


def forecast(
    models: list[Forecaster],
    store: Era5Store,
    init: str,
    steps: int,
    ens: EnsembleConfig,
    device: torch.device,
    seed: int = 0,
) -> tuple[int, np.ndarray]:
    t0 = np.datetime64(init, "s")
    hits = np.flatnonzero(store.times == t0)
    if len(hits) == 0:
        raise ValueError(f"{init} no está en los datos")
    t = int(hits[0])
    x0 = models[0].normalize(torch.from_numpy(np.array(store.fields[t], dtype=np.float32)).to(device))
    times = store.times[t] + np.arange(steps) * np.timedelta64(6, "h")
    forc = torch.from_numpy(forcings_for_times(times, store.lat, store.lon)).to(device)
    traj = run_ensemble(models, x0, forc, ens, seed=seed)
    phys = models[0].denormalize(traj.reshape(-1, *traj.shape[2:])).reshape(traj.shape).cpu().numpy()
    return t, phys


def buenos_aires_report(
    store: Era5Store,
    t: int,
    phys: np.ndarray,
    out_dir: str | Path,
    threshold_c: float = 30.0,
    point: tuple[float, float] = BUENOS_AIRES,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = store.index("t2m")
    steps = phys.shape[1]
    times = [str(store.times[t] + np.timedelta64(6 * k, "h"))[:13].replace("T", " ") + "h" for k in range(steps)]
    series = bilinear(phys[:, :, c], store.lat, store.lon, *point) - KELVIN
    truth = None
    if t + steps <= len(store.times):
        truth = bilinear(np.asarray(store.fields[t : t + steps, c]), store.lat, store.lon, *point) - KELVIN
    plot_point_plume(series, times, threshold_c, f"Temperatura a 2 m en Buenos Aires · {phys.shape[0]} miembros", out / "buenos_aires_t2m.png", truth)
    animate_field(phys[:, :, c], store.lat, store.lon, times, "t2m (°C)", out / "t2m_ensamble.gif", offset=-KELVIN)
    if "z500" in store.names:
        animate_field(phys[:, :, store.index("z500")] / 9.80665, store.lat, store.lon, times, "altura de 500 hPa (m)", out / "z500_ensamble.gif")
    report = {
        "init": times[0],
        "point": point,
        "threshold_c": threshold_c,
        "members": int(phys.shape[0]),
        "lead_hours": [6 * k for k in range(steps)],
        "mean_c": series.mean(axis=0).round(2).tolist(),
        "p10_c": np.percentile(series, 10, axis=0).round(2).tolist(),
        "p90_c": np.percentile(series, 90, axis=0).round(2).tolist(),
        "prob_above": (series > threshold_c).mean(axis=0).round(3).tolist(),
        "truth_c": None if truth is None else truth.round(2).tolist(),
    }
    (out / "buenos_aires.json").write_text(json.dumps(report, indent=2))
    return report
