import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from clima.compare import comparison_rows, sweep_ic_scale
from clima.ensemble import EnsembleConfig
from clima.evaluate import ACC_THRESHOLD, PER_INIT, aggregate, evaluate, save_report
from clima.forecast import buenos_aires_report, forecast
from clima.metrics import bootstrap_means, bootstrap_weights, percentile_interval, significant, skill_horizon
from clima.models.afno import AFNOConfig
from clima.models.forecaster import Forecaster, save_forecaster
from clima.plots import plot_acc, plot_skill, plot_strategy_scores

ROOT = Path(__file__).resolve().parents[1]


def make_model(store, seed=0):
    torch.manual_seed(seed)
    model = Forecaster.from_store(store, AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2, noise_channels=2)).eval()
    with torch.no_grad():
        for p in model.net.head.parameters():
            p.normal_(0, 0.05)
    return model


@pytest.fixture(scope="module")
def result(store):
    return evaluate([make_model(store)], store, "2018-03-01", "2018-03-25", steps=4, every=4, ens=EnsembleConfig(members=4, batch=4), verbose=False, event_metrics=False, bootstrap=200)


def test_bootstrap_weights_are_resamples():
    w = bootstrap_weights(12, block=3, reps=50, seed=1)
    assert w.shape == (50, 12)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w * 12, np.round(w * 12))
    assert np.array_equal(w, bootstrap_weights(12, block=3, reps=50, seed=1))
    assert not np.array_equal(w, bootstrap_weights(12, block=3, reps=50, seed=2))
    whole = bootstrap_weights(5, block=99, reps=4, seed=0)
    assert np.allclose(whole, 0.2)


def test_bootstrap_interval_covers_the_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(3.0, 1.0, size=(400, 2, 3))
    samples = bootstrap_means(values, bootstrap_weights(400, block=1, reps=500, seed=0))
    lo, hi = percentile_interval(samples)
    assert samples.shape == (500, 2, 3)
    assert np.all(lo < values.mean(axis=0)) and np.all(values.mean(axis=0) < hi)
    assert np.all(hi - lo < 0.3) and np.all(hi - lo > 0.1)


def test_block_bootstrap_widens_interval_for_correlated_series():
    rng = np.random.default_rng(1)
    series = np.repeat(rng.normal(size=60), 10)[:, None]
    width = []
    for block in (1, 10):
        lo, hi = percentile_interval(bootstrap_means(series, bootstrap_weights(len(series), block, 800, seed=0)))
        width.append(float(hi[0] - lo[0]))
    assert width[1] > 2.0 * width[0]


def test_skill_horizon_known_answers():
    hours = [0, 24, 48, 72, 96]
    assert skill_horizon(hours, [1.0, 0.9, 0.7, 0.5, 0.3]) == pytest.approx(2.5)
    assert skill_horizon(hours, [1.0, 0.9, 0.8, 0.7, 0.65]) == float("inf")
    assert skill_horizon(hours, [1.0, 0.5, 0.4, 0.3, 0.2]) == 0.0
    assert skill_horizon(hours, [1.0, 0.9, 0.6, 0.3, 0.2], threshold=0.6) == pytest.approx(2.0)


def test_significant_keeps_small_magnitudes():
    out = significant([123456.789, 0.000023456789, 0.0, -0.5, float("nan")], digits=4)
    assert out[0] == 123500.0 and out[1] == pytest.approx(2.346e-5) and out[2] == 0.0 and out[3] == -0.5 and np.isnan(out[4])


def test_climatology_is_interpolated_between_months(store):
    clim = np.asarray(store.climatology, dtype=np.float64)
    middle = int(np.flatnonzero(store.times == np.datetime64("2018-01-16T12", "s"))[0])
    assert np.allclose(store.climatology_at(middle), clim[0, 2], atol=1e-3)
    edge = int(np.flatnonzero(store.times == np.datetime64("2018-02-01T00", "s"))[0])
    assert np.allclose(store.climatology_at(edge), 0.5 * (clim[0, 0] + clim[1, 0]), atol=1e-3)
    late = int(np.flatnonzero(store.times == np.datetime64("2018-01-31T18", "s"))[0])
    weight = (30.75 / 31.0) - 0.5
    assert np.allclose(store.climatology_at(late), (1 - weight) * clim[0, 3] + weight * clim[1, 3], atol=1e-3)


def test_evaluate_keeps_per_init_and_aggregates_like_weatherbench_2(store, result):
    per_init = result["per_init"]
    n = result["inits"]
    assert n > 3
    assert set(PER_INIT) <= set(per_init)
    assert per_init["det_mse"].shape == (n, 5, len(store.names))
    assert per_init["times"].shape == (n,)
    assert np.allclose(result["det_rmse"], np.sqrt(per_init["det_mse"].mean(axis=0)), atol=1e-4)
    assert np.allclose(result["spread"], np.sqrt(per_init["variance"].mean(axis=0)), atol=1e-4)
    assert np.allclose(result["det_rmse_por_inicio"], np.sqrt(per_init["det_mse"]).mean(axis=0), atol=1e-4)
    assert np.all(np.asarray(result["det_rmse_por_inicio"]) <= np.asarray(result["det_rmse"]) + 1e-4)
    assert np.allclose(np.asarray(result["det_rmse"])[0], 0.0, atol=1e-5)
    again = aggregate(per_init)
    assert np.allclose(again["crps"], result["crps"], atol=1e-4)


def test_intervals_bracket_the_estimate(store, result):
    assert result["bootstrap"] == {"reps": 200, "block": 1, "level": 0.95}
    for key in ("det_rmse", "mean_rmse", "crps", "spread_skill", "det_acc", "mean_acc", "mean_gain_pct"):
        lo = np.asarray(result["intervals"][key]["lo"])[1:]
        hi = np.asarray(result["intervals"][key]["hi"])[1:]
        value = np.asarray(result[key])[1:]
        assert lo.shape == value.shape
        assert np.all(lo <= value + 1e-4) and np.all(value <= hi + 1e-4), key
    horizon = result["horizon_days"]
    assert horizon["threshold"] == ACC_THRESHOLD
    assert len(horizon["det_acc"]) == len(store.names)
    band = result["intervals"]["horizon_days"]["det_acc"]
    assert np.all(np.asarray(band["hi"]) <= 1.0 + 1e-9)
    single = evaluate([make_model(store)], store, "2018-03-01", "2018-03-25", steps=2, every=4, ens=EnsembleConfig(members=3, batch=3), max_inits=1, verbose=False, event_metrics=False)
    assert "intervals" not in single


def test_report_files_and_figures(store, result, tmp_path):
    out = save_report(result, tmp_path / "eval")
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "per_init" not in saved and "intervals" in saved and "horizon_days" in saved
    assert "per_init" in result
    with np.load(out / "por_inicio.npz") as z:
        assert set(PER_INIT) | {"times"} == set(z.files)
        assert np.allclose(z["crps"], result["per_init"]["crps"])
    assert plot_skill(saved, out).stat().st_size > 10_000
    assert plot_acc(saved, out).name == "horizonte.png"
    assert (out / "horizonte.png").stat().st_size > 10_000


def test_ic_scale_sweep_orders_spread_and_has_paired_interval(store, tmp_path):
    base = EnsembleConfig(members=4, batch=4, ic_scale=0.5, model_noise=False)
    sweep = sweep_ic_scale([make_model(store)], store, "2018-03-01", "2018-03-25", (0.1, 0.5), base, steps=2, every=4, verbose=False)
    assert list(sweep["strategies"]) == ["ci=0.1", "ci=0.5"]
    assert sweep["reference"] == "ci=0.5"
    small, large = (np.asarray(sweep["strategies"][k]["spread"])[1] for k in ("ci=0.1", "ci=0.5"))
    assert np.all(small < large)
    assert "crpss" in sweep["strategies"]["ci=0.1"]["intervals"]
    assert "crpss" not in sweep["strategies"]["ci=0.5"]["intervals"]
    assert all("per_init" not in r for r in sweep["strategies"].values())
    rows = [r for r in comparison_rows(sweep) if r["estrategia"] == "ci=0.1"]
    assert all(float(r["crpss_lo"]) <= float(r["crpss_vs_ci"]) + 1e-6 <= float(r["crpss_hi"]) + 2e-6 for r in rows)
    q500 = [float(r["crps"]) for r in rows if r["variable"] == "q500"]
    assert all(v > 0 and len(f"{v:.10g}".replace(".", "").lstrip("0").split("e")[0]) > 2 for v in q500)
    path = plot_strategy_scores(sweep, tmp_path, title="Cuánto perturbar el arranque", filename="calibracion.png")
    assert path.name == "calibracion.png" and path.stat().st_size > 10_000


def test_scripts_evaluate_and_calibrate_end_to_end(store, tmp_path):
    ckpt = tmp_path / "modelo.pt"
    save_forecaster(ckpt, make_model(store))
    common = ["--checkpoints", str(ckpt), "--data", str(store.root), "--start", "2018-03-01", "--end", "2018-03-25", "--device", "cpu"]
    out = tmp_path / "evaluacion"
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), *common, "--steps", "4", "--every", "8", "--members", "4", "--batch", "4", "--bootstrap", "50", "--no-events", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    for name in ("metrics.json", "metrics.csv", "por_inicio.npz", "skill.png", "horizonte.png", "scorecard.csv"):
        assert (out / name).exists(), name
    out = tmp_path / "calibracion"
    sizes = ["--scales", "0.2", "0.5", "--steps", "2", "--every", "8", "--members", "4", "--batch", "4"]
    cmd = [sys.executable, str(ROOT / "scripts" / "calibrate_ensemble.py"), *common, *sizes, "--variables", "z500", "t2m", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    for name in ("comparacion.json", "comparacion_ensambles.csv", "calibracion.png"):
        assert (out / name).exists(), name


def test_point_report_uses_common_style(store, tmp_path):
    model = make_model(store)
    t, phys = forecast([model], store, "2018-03-02T00", 8, EnsembleConfig(members=5, batch=5), torch.device("cpu"))
    report = buenos_aires_report(store, t, phys, tmp_path, threshold_c=20.0)
    assert report["members"] == 5 and len(report["prob_above"]) == 9
    assert (tmp_path / "buenos_aires_t2m.png").stat().st_size > 10_000
