from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .data.store import Era5Store
from .ensemble import STRATEGIES, STRATEGY_LABELS, EnsembleConfig, strategy_available, strategy_config
from .evaluate import evaluate, write_json
from .metrics import bootstrap_means, bootstrap_weights, percentile_interval, rank_outlier_ratio, rank_reliability_index, significant
from .models.forecaster import Forecaster

COLUMNS = ("estrategia", "variable", "lead_h", "crps", "crpss_vs_ci", "crpss_lo", "crpss_hi", "spread", "rmse_media", "rmse_control", "spread_skill", "rank_outliers", "rank_ri")
REFERENCE = "ci"


def compare_strategies(
    models: list[Forecaster],
    store: Era5Store,
    start: str,
    end: str,
    strategies: tuple[str, ...] = tuple(STRATEGIES),
    base: EnsembleConfig = EnsembleConfig(),
    steps: int = 20,
    every: int = 8,
    max_inits: int | None = None,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    results = {}
    skipped = {}
    for name in strategies:
        reason = strategy_available(name, models)
        if reason:
            skipped[name] = reason
            if verbose:
                print(f"se omite {name}: {reason}", flush=True)
            continue
        cfg = strategy_config(name, base)
        if verbose:
            print(f"estrategia {name} ({STRATEGY_LABELS[name]}): ic_scale={cfg.ic_scale} ruido={cfg.model_noise} dropout={cfg.mc_dropout}", flush=True)
        results[name] = evaluate(models, store, start, end, steps, every, cfg, max_inits, device, seed, verbose, event_metrics=False)
    paired_skill(results, REFERENCE, seed)
    return {"strategies": results, "skipped": skipped, "labels": {k: STRATEGY_LABELS[k] for k in results}, "reference": REFERENCE}


def sweep_ic_scale(
    models: list[Forecaster],
    store: Era5Store,
    start: str,
    end: str,
    scales: tuple[float, ...],
    base: EnsembleConfig = EnsembleConfig(),
    steps: int = 20,
    every: int = 8,
    max_inits: int | None = None,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    results = {}
    labels = {}
    for scale in scales:
        name = f"ci={scale:g}"
        labels[name] = f"perturbación inicial × {scale:g}"
        if verbose:
            print(f"escala de condiciones iniciales {scale:g}", flush=True)
        results[name] = evaluate(models, store, start, end, steps, every, replace(base, ic_scale=float(scale)), max_inits, device, seed, verbose, event_metrics=False)
    reference = f"ci={base.ic_scale:g}" if f"ci={base.ic_scale:g}" in results else next(iter(results), None)
    paired_skill(results, reference, seed)
    return {"strategies": results, "skipped": {}, "labels": labels, "reference": reference}


def paired_skill(results: dict, reference: str | None, seed: int = 0) -> None:
    base = results.get(reference, {}).get("per_init")
    for name, r in results.items():
        own = r.pop("per_init", None) if name != reference else r.get("per_init")
        if base is None or own is None or name == reference or own["crps"].shape != base["crps"].shape or "bootstrap" not in r:
            continue
        weights = bootstrap_weights(len(own["crps"]), r["bootstrap"]["block"], r["bootstrap"]["reps"], seed)
        skill = 1.0 - bootstrap_means(own["crps"], weights) / np.maximum(bootstrap_means(base["crps"], weights), 1e-12)
        lo, hi = percentile_interval(skill)
        r["intervals"]["crpss"] = {"lo": significant(lo).tolist(), "hi": significant(hi).tolist()}
    if reference in results:
        results[reference].pop("per_init", None)


def comparison_rows(comparison: dict) -> list[dict]:
    results = comparison["strategies"]
    name = comparison.get("reference", REFERENCE)
    reference = np.asarray(results[name]["crps"]) if name in results else None
    rows = []
    for name, r in results.items():
        crps = np.asarray(r["crps"])
        crpss = 1.0 - crps / np.maximum(reference, 1e-12) if reference is not None and reference.shape == crps.shape else np.full(crps.shape, np.nan)
        band = r.get("intervals", {}).get("crpss")
        lo, hi = (np.asarray(band["lo"]), np.asarray(band["hi"])) if band else (np.full(crps.shape, np.nan), np.full(crps.shape, np.nan))
        ranks = np.asarray(r["rank_histogram"])
        outliers = rank_outlier_ratio(ranks)
        ri = rank_reliability_index(ranks)
        for c, var in enumerate(r["channels"]):
            for i, lead in enumerate(r["lead_hours"]):
                if lead == 0:
                    continue
                values = (crps[i, c], crpss[i, c], lo[i, c], hi[i, c], r["spread"][i][c], r["mean_rmse"][i][c], r["det_rmse"][i][c], r["spread_skill"][i][c], outliers[i, c], ri[i, c])
                rows.append({"estrategia": name, "variable": var, "lead_h": lead, **{k: _fmt(v) for k, v in zip(COLUMNS[3:], values, strict=True)}})
    return rows


def _fmt(value) -> str:
    value = float(value)
    return "" if not np.isfinite(value) else f"{value:.6g}"


def save_comparison(comparison: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "comparacion.json", comparison)
    with open(out / "comparacion_ensambles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(comparison_rows(comparison))
    return out


def summary_lines(comparison: dict, variable: str = "z500", leads: tuple[int, ...] = (24, 72, 120)) -> list[str]:
    lines = []
    for name, r in comparison["strategies"].items():
        if variable not in r["channels"]:
            continue
        c = r["channels"].index(variable)
        parts = [f"{h}h crps {r['crps'][i][c]:.4g} sprd/rmse {r['spread_skill'][i][c]:.2f}" for i, h in enumerate(r["lead_hours"]) if h in leads]
        lines.append(f"{name:>10} | {variable} | " + " | ".join(parts))
    return lines
