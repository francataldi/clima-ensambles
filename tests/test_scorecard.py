import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from clima.ensemble import EnsembleConfig
from clima.evaluate import evaluate, save_report
from clima.models.afno import AFNOConfig
from clima.models.forecaster import Forecaster
from clima.plots import plot_scorecard
from clima.scorecard import COLUMNS, save_scorecard, scorecard, scorecard_leads

ROOT = Path(__file__).resolve().parents[1]


def fake_result(steps=8, channels=("z500", "t2m", "u10")):
    rng = np.random.default_rng(0)
    L, C = steps + 1, len(channels)
    persist = rng.uniform(1.0, 2.0, (L, C))
    det = 0.5 * persist
    det[:, 2] = 0.0
    det_acc = rng.uniform(0.5, 0.9, (L, C))
    return {
        "inits": 3,
        "ensemble": {"members": 5},
        "channels": list(channels),
        "lead_hours": [6 * k for k in range(L)],
        "det_rmse": det.tolist(),
        "mean_rmse": (0.9 * det).tolist(),
        "persist_rmse": persist.tolist(),
        "clim_rmse": det.tolist(),
        "det_acc": det_acc.tolist(),
        "mean_acc": (det_acc + 0.05).tolist(),
    }


def test_known_skills():
    card = scorecard(fake_result())
    assert card["leads"] == [6, 12, 18, 24, 30, 36, 42, 48]
    assert np.allclose(card["skill_persistencia"][:2], 0.5)
    assert np.allclose(card["skill_persistencia"][2], 1.0)
    assert np.allclose(card["skill_climatologia"][:2], 0.0)
    assert np.isnan(card["skill_climatologia"][2]).all()
    assert np.allclose(card["cambio_rmse_pct"][:2], -10.0)
    assert np.allclose(card["cambio_acc"], 0.05)


def test_persistence_as_control_has_zero_skill():
    result = fake_result()
    result["det_rmse"] = result["persist_rmse"]
    card = scorecard(result, variables=["t2m"], leads=[12, 48])
    assert card["variables"] == ["t2m"] and card["leads"] == [12, 48]
    assert np.allclose(card["skill_persistencia"], 0.0)


def test_scorecard_leads_selection():
    assert scorecard_leads([6 * k for k in range(41)]) == [24 * d for d in range(1, 11)]
    assert scorecard_leads([0, 6, 12, 18, 24]) == [6, 12, 18, 24]
    hourly = scorecard_leads([6 * k for k in range(81)], max_columns=10)
    assert len(hourly) == 10 and hourly[0] == 24 and hourly[-1] == 480


def test_scorecard_files_and_missing_acc(tmp_path):
    result = fake_result()
    del result["mean_acc"]
    card = scorecard(result)
    assert np.isnan(card["cambio_acc"]).all()
    path = save_scorecard(card, tmp_path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == COLUMNS
    assert len(rows) == 3 * 8
    assert rows[0]["acc_media"] == ""
    assert plot_scorecard(card, tmp_path).stat().st_size > 10_000


def test_scorecard_from_evaluation_and_script(store, tmp_path):
    torch.manual_seed(0)
    model = Forecaster.from_store(store, AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2)).eval()
    result = evaluate([model], store, "2018-03-01", "2018-03-20", steps=4, every=4, ens=EnsembleConfig(members=3, batch=3), max_inits=2, verbose=False, event_metrics=False)
    card = scorecard(result)
    assert card["variables"] == store.names
    assert np.isfinite(card["cambio_acc"]).all()
    assert np.allclose(card["skill_persistencia"], 0.0, atol=1e-6)
    out = save_report(result, tmp_path / "eval")
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["mean_acc"]
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "scorecard.py"), "--metrics", str(out / "metrics.json"), "--variables", "z500", "t2m"], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert (out / "scorecard.png").exists() and (out / "scorecard.csv").exists()


@pytest.mark.parametrize("value,expected", [(0.004, "0"), (-12.4, "−12"), (-3.25, "−3.2"), (0.6213, "0.62")])
def test_cell_labels(value, expected):
    from clima.plots import _cell_text

    key = "cambio_rmse_pct" if abs(value) > 1 else "skill_persistencia"
    assert _cell_text(key, value) == expected
