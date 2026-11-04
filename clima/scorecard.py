from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PANELS = (
    ("skill_persistencia", "Control vs persistencia", "skill RMSE = 1 − RMSE / RMSE persistencia", False),
    ("skill_climatologia", "Control vs climatología", "skill RMSE = 1 − RMSE / RMSE climatología", False),
    ("cambio_rmse_pct", "Media del ensamble vs control: RMSE", "cambio de RMSE en % (negativo = la media gana)", True),
    ("cambio_acc", "Media del ensamble vs control: ACC", "ACC media − ACC control (positivo = la media gana)", False),
)

COLUMNS = (
    "variable",
    "lead_h",
    "rmse_control",
    "rmse_media",
    "rmse_persistencia",
    "rmse_climatologia",
    "skill_persistencia",
    "skill_climatologia",
    "cambio_rmse_pct",
    "acc_control",
    "acc_media",
    "cambio_acc",
)


def scorecard_leads(lead_hours: list[int], max_columns: int = 10) -> list[int]:
    positive = [h for h in lead_hours if h > 0]
    daily = [h for h in positive if h % 24 == 0]
    chosen = positive if len(positive) <= max_columns or len(daily) < 3 else daily
    if len(chosen) > max_columns:
        chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, max_columns).round().astype(int)]
    return chosen


def _ratio_skill(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return 1.0 - out


def scorecard(result: dict, variables: list[str] | None = None, leads: list[int] | None = None) -> dict:
    names = result["channels"]
    variables = [v for v in (variables or names) if v in names]
    leads = leads or scorecard_leads(result["lead_hours"])
    li = [result["lead_hours"].index(h) for h in leads]
    ci = [names.index(v) for v in variables]

    def pick(key: str) -> np.ndarray:
        if key not in result:
            return np.full((len(variables), len(leads)), np.nan)
        return np.asarray(result[key], dtype=np.float64)[np.ix_(li, ci)].T

    det, mean, persist, clim = pick("det_rmse"), pick("mean_rmse"), pick("persist_rmse"), pick("clim_rmse")
    det_acc, mean_acc = pick("det_acc"), pick("mean_acc")
    return {
        "variables": variables,
        "leads": leads,
        "inits": result.get("inits"),
        "members": result.get("ensemble", {}).get("members"),
        "rmse_control": det,
        "rmse_media": mean,
        "rmse_persistencia": persist,
        "rmse_climatologia": clim,
        "skill_persistencia": _ratio_skill(det, persist),
        "skill_climatologia": _ratio_skill(det, clim),
        "cambio_rmse_pct": -100.0 * _ratio_skill(mean, det),
        "acc_control": det_acc,
        "acc_media": mean_acc,
        "cambio_acc": mean_acc - det_acc,
    }


def save_scorecard(card: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scorecard.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for i, var in enumerate(card["variables"]):
            for j, lead in enumerate(card["leads"]):
                values = [card[k][i, j] for k in COLUMNS[2:]]
                writer.writerow([var, lead] + ["" if not np.isfinite(v) else f"{v:.6g}" for v in values])
    return path
