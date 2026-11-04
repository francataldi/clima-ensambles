import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from clima.compare import COLUMNS, compare_strategies, comparison_rows, save_comparison
from clima.data.dataset import dynamic_forcings
from clima.ensemble import STRATEGIES, EnsembleConfig, run_ensemble, strategy_available, strategy_config
from clima.metrics import rank_counts, rank_histogram, rank_outlier_ratio, rank_reliability_index
from clima.models.afno import AFNOConfig
from clima.models.forecaster import Forecaster, save_forecaster
from clima.plots import plot_strategy_rank_histograms, plot_strategy_scores

ROOT = Path(__file__).resolve().parents[1]


def make_model(store, noise=2, dropout=0.2, seed=0):
    torch.manual_seed(seed)
    model = Forecaster.from_store(store, AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2, dropout=dropout, noise_channels=noise)).eval()
    with torch.no_grad():
        for p in model.net.head.parameters():
            p.normal_(0, 0.05)
    return model


@pytest.fixture(scope="module")
def setup(store):
    x0 = torch.randn(len(store.names), 32, 64, generator=torch.Generator().manual_seed(0))
    forc = torch.from_numpy(dynamic_forcings(store, np.arange(0, 3)))
    return make_model(store), x0, forc


def test_strategy_presets():
    base = EnsembleConfig(members=8, ic_scale=0.7)
    assert strategy_config("ci", base) == EnsembleConfig(members=8, ic_scale=0.7, model_noise=False, mc_dropout=False)
    assert strategy_config("ruido", base).ic_scale == 0.0 and strategy_config("ruido", base).model_noise
    both = strategy_config("ci+ruido", base)
    assert both.ic_scale == 0.7 and both.model_noise and not both.mc_dropout
    drop = strategy_config("mc_dropout", base)
    assert drop.ic_scale == 0.0 and drop.mc_dropout and not drop.model_noise
    with pytest.raises(ValueError):
        strategy_config("otra", base)


def test_strategy_availability(store):
    plain = make_model(store, noise=0, dropout=0.0)
    assert strategy_available("ci", [plain]) is None
    assert strategy_available("ruido", [plain]) is not None
    assert strategy_available("ci+ruido", [plain]) is not None
    assert strategy_available("mc_dropout", [plain]) is not None
    full = make_model(store)
    assert all(strategy_available(name, [full]) is None for name in STRATEGIES)


def test_without_any_perturbation_members_are_identical(setup):
    model, x0, forc = setup
    cfg = EnsembleConfig(members=4, ic_scale=0.0, model_noise=False, batch=4)
    traj = run_ensemble([model], x0, forc, cfg, seed=0)
    assert torch.allclose(traj, traj[:1].expand_as(traj))


def test_each_strategy_spreads_and_keeps_control(setup):
    model, x0, forc = setup
    det = model.rollout(x0.unsqueeze(0), forc.unsqueeze(0))[0]
    for name in STRATEGIES:
        traj = run_ensemble([model], x0, forc, strategy_config(name, EnsembleConfig(members=5, batch=3)), seed=3)
        assert torch.allclose(traj[0], det, atol=1e-5), name
        assert traj[1:, -1].std(0).mean() > 0, name
    assert not any(m.training for m in model.modules())


def test_mc_dropout_is_reproducible_and_leaves_global_rng(setup):
    model, x0, forc = setup
    cfg = strategy_config("mc_dropout", EnsembleConfig(members=4, batch=4))
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    a = run_ensemble([model], x0, forc, cfg, seed=7)
    assert torch.equal(torch.rand(3), expected)
    b = run_ensemble([model], x0, forc, cfg, seed=7)
    c = run_ensemble([model], x0, forc, cfg, seed=8)
    assert torch.equal(a, b)
    assert not torch.allclose(a[1:], c[1:])


def test_multi_model_ensemble_is_deterministic(store, setup):
    _, x0, forc = setup
    models = [make_model(store, seed=1), make_model(store, seed=2)]
    cfg = EnsembleConfig(members=6, batch=4)
    assert torch.equal(run_ensemble(models, x0, forc, cfg, seed=5), run_ensemble(models, x0, forc, cfg, seed=5))


def test_rank_metrics_known_answers():
    rng = np.random.default_rng(0)
    members = rng.normal(size=(9, 2, 3, 4, 5))
    truth = rng.normal(size=(2, 3, 4, 5))
    counts = rank_counts(members, truth)
    assert counts.shape == (2, 3, 10)
    assert all((counts[i, j] == rank_histogram(members[:, i, j], truth[i, j], 10)).all() for i in range(2) for j in range(3))
    flat = np.full(10, 7.0)
    assert rank_outlier_ratio(flat) == pytest.approx(1.0)
    assert rank_reliability_index(flat) == pytest.approx(0.0)
    u_shaped = np.zeros(10)
    u_shaped[[0, -1]] = 1.0
    assert rank_outlier_ratio(u_shaped) == pytest.approx(5.0)
    assert rank_reliability_index(u_shaped) == pytest.approx(1.6)


def test_compare_strategies_table_and_plots(store, setup, tmp_path):
    model = setup[0]
    base = EnsembleConfig(members=4, batch=4)
    comparison = compare_strategies([model], store, "2018-03-01", "2018-03-20", base=base, steps=4, every=8, max_inits=2, verbose=False)
    assert list(comparison["strategies"]) == list(STRATEGIES)
    assert comparison["skipped"] == {}
    rows = comparison_rows(comparison)
    assert len(rows) == len(STRATEGIES) * len(store.names) * 4
    assert all(float(r["crpss_vs_ci"]) == 0.0 for r in rows if r["estrategia"] == "ci")
    for name, result in comparison["strategies"].items():
        assert np.asarray(result["rank_histogram"]).shape == (5, len(store.names), 5)
        assert np.allclose(np.asarray(result["rank_histogram"]).sum(-1), 1.0, atol=1e-3)
        assert result["ensemble"]["model_noise"] == STRATEGIES[name]["model_noise"]
    out = save_comparison(comparison, tmp_path / "cmp")
    with open(out / "comparacion_ensambles.csv", newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    assert tuple(table[0]) == COLUMNS and len(table) == len(rows)
    assert plot_strategy_scores(comparison, out).stat().st_size > 10_000
    assert plot_strategy_rank_histograms(comparison, out, "t2m").stat().st_size > 10_000
    only_ic = compare_strategies([make_model(store, noise=0, dropout=0.0)], store, "2018-03-01", "2018-03-20", base=base, steps=2, every=8, max_inits=1, verbose=False)
    assert list(only_ic["strategies"]) == ["ci"]
    assert set(only_ic["skipped"]) == {"ruido", "ci+ruido", "mc_dropout"}


def test_compare_script_end_to_end(store, setup, tmp_path):
    ckpt = tmp_path / "modelo.pt"
    save_forecaster(ckpt, setup[0])
    out = tmp_path / "salida"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "compare_ensembles.py"),
        "--checkpoints", str(ckpt), "--data", str(store.root),
        "--start", "2018-03-01", "--end", "2018-03-20",
        "--steps", "4", "--every", "8", "--max-inits", "1", "--members", "4", "--batch", "4",
        "--strategies", "ci", "ruido", "--variables", "z500", "t2m", "--device", "cpu", "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "resultados en" in proc.stdout
    for name in ("comparacion.json", "comparacion_ensambles.csv", "estrategias.png", "rangos_z500.png", "rangos_t2m.png"):
        assert (out / name).exists(), name
