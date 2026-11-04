from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .data.store import Era5Store
from .ensemble import STRATEGIES, STRATEGY_LABELS, EnsembleConfig, strategy_available, strategy_config
from .evaluate import evaluate, write_json
from .metrics import rank_outlier_ratio, rank_reliability_index
from .models.forecaster import Forecaster

COLUMNS = ("estrategia", "variable", "lead_h", "crps", "crpss_vs_ci", "spread", "rmse_media", "rmse_control", "spread_skill", "rank_outliers", "rank_ri")


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
    return {"strategies": results, "skipped": skipped, "labels": {k: STRATEGY_LABELS[k] for k in results}}


def comparison_rows(comparison: dict) -> list[dict]:
    results = comparison["strategies"]
    reference = np.asarray(results["ci"]["crps"]) if "ci" in results else None
    rows = []
    for name, r in results.items():
        crps = np.asarray(r["crps"])
        crpss = 1.0 - crps / np.maximum(reference, 1e-12) if reference is not None and reference.shape == crps.shape else np.full(crps.shape, np.nan)
        ranks = np.asarray(r["rank_histogram"])
        outliers = rank_outlier_ratio(ranks)
        ri = rank_reliability_index(ranks)
        for c, var in enumerate(r["channels"]):
            for i, lead in enumerate(r["lead_hours"]):
                if lead == 0:
                    continue
                values = (crps[i, c], crpss[i, c], r["spread"][i][c], r["mean_rmse"][i][c], r["det_rmse"][i][c], r["spread_skill"][i][c], outliers[i, c], ri[i, c])
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
