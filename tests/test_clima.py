import copy

import numpy as np
import torch

from clima.data.convert import convert
from clima.data.dataset import N_DYNAMIC, ForecastDataset, dynamic_forcings
from clima.data.synthetic import make_synthetic
from clima.ensemble import EnsembleConfig, run_ensemble, smooth_noise
from clima.metrics import acc, bilinear, crps_ensemble, ensemble_spread, rmse
from clima.models.afno import AFNOConfig, AFNONet
from clima.models.forecaster import Forecaster, load_forecaster, save_forecaster
from clima.solar import SOLAR_CONSTANT, time_features, toa_insolation
from clima.train import TrainConfig, train

TINY = AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2, dropout=0.1)


def test_insolation_is_physical():
    times = np.array(["2020-06-21T12:00", "2020-12-21T12:00"], dtype="datetime64[s]")
    lat = np.linspace(-87, 87, 32)
    lon = np.arange(64) * 5.625
    sol = toa_insolation(times, lat, lon)
    assert sol.min() >= 0
    assert sol.max() <= 1.04 * SOLAR_CONSTANT
    north = np.argmin(np.abs(lat - 60))
    south = np.argmin(np.abs(lat + 60))
    assert sol[0, north, 0] > sol[0, south, 0]
    assert sol[1, south, 0] > sol[1, north, 0]
    equator = np.argmin(np.abs(lat))
    assert sol[0, equator, 32] == 0
    assert sol[0, -1, 32] > 0
    assert time_features(times).shape == (2, 4)


def test_convert_roundtrip(store, tmp_path):
    assert store.fields.shape == (len(store.times), len(store.names), 32, 64)
    assert np.all(np.diff(store.lat) > 0)
    ds = make_synthetic("2018-01-01", "2018-03-31T18", seed=1)
    c = store.index("t2m")
    src = ds["2m_temperature"].isel(time=10).transpose("latitude", "longitude").values
    assert np.allclose(store.fields[10, c], src)
    z = store.index("z500")
    src = ds["geopotential"].sel(level=500).isel(time=3).transpose("latitude", "longitude").values
    assert np.allclose(store.fields[3, z], src)
    convert(ds, store.root, 2018, 2018, verbose=False)


def test_stats_and_dataset(store):
    ds = ForecastDataset(store, "2018-01-01", "2018-02-28T18", rollout=3)
    states, forcings, t = ds[5]
    assert states.shape == (4, len(store.names), 32, 64)
    assert forcings.shape == (3, N_DYNAMIC, 32, 64)
    assert torch.isfinite(states).all()
    assert abs(float(states[:, 0].mean())) < 1.0
    assert len(ds) == len(store.valid_starts(store.time_range("2018-01-01", "2018-02-28T18"), 3))
    assert store.climatology.shape == (12, 4, len(store.names), 32, 64)
    assert dynamic_forcings(store, np.array([0, 1])).shape == (2, N_DYNAMIC, 32, 64)


def test_valid_starts_skip_gaps(store):
    gappy = copy.copy(store)
    gappy.times = store.times.copy()
    gappy.times[5:] = gappy.times[5:] + np.timedelta64(6, "h")
    starts = gappy.valid_starts(np.arange(10), 2)
    assert list(starts) == [0, 1, 2, 5, 6, 7]


def test_afno_shapes_and_persistence_at_init(store):
    net = AFNONet(7, 5, 32, 64, TINY)
    assert net(torch.randn(2, 7, 32, 64)).shape == (2, 5, 32, 64)
    model = Forecaster.from_store(store, TINY)
    x = torch.randn(3, len(store.names), 32, 64)
    f = torch.randn(3, N_DYNAMIC, 32, 64)
    model.eval()
    assert torch.allclose(model.step(x, f), x)


def test_metrics():
    lat = np.linspace(-80, 80, 9)
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    truth = np.random.default_rng(0).normal(size=(9, 12))
    assert np.allclose(rmse(truth, truth, w), 0)
    clim = np.zeros_like(truth)
    assert np.isclose(acc(truth, truth, clim, w), 1.0)
    point = np.repeat(truth[None] + 1.0, 5, axis=0)
    assert np.isclose(crps_ensemble(point, truth, w), 1.0)
    members = np.random.default_rng(1).normal(size=(200, 9, 12))
    assert crps_ensemble(members, np.zeros((9, 12)), w) < np.abs(members).mean()
    assert 0.9 < ensemble_spread(members, w) < 1.1
    glat = np.array([-10.0, 0.0, 10.0])
    glon = np.array([0.0, 90.0, 180.0, 270.0])
    field = glat[:, None] + 0 * glon[None, :]
    assert np.isclose(bilinear(field, glat, glon, 5.0, 45.0), 5.0)


def test_ensemble_control_and_spread(store):
    model = Forecaster.from_store(store, TINY).eval()
    with torch.no_grad():
        for p in model.net.head.parameters():
            p.normal_(0, 0.01)
    x0 = torch.randn(len(store.names), 32, 64)
    forc = torch.from_numpy(dynamic_forcings(store, np.arange(0, 4)))
    ens = EnsembleConfig(members=6, ic_scale=0.5, batch=4)
    traj = run_ensemble([model], x0, forc, ens, seed=0)
    assert traj.shape == (6, 5, len(store.names), 32, 64)
    det = model.rollout(x0.unsqueeze(0), forc.unsqueeze(0))[0]
    assert torch.allclose(traj[0], det, atol=1e-5)
    assert traj[:, -1].std(0).mean() > 0
    noise = smooth_noise((2, 3, 32, 64), 6.0, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert torch.allclose(noise.std(dim=(-2, -1)), torch.ones(2, 3), atol=1e-3)


def test_train_smoke_and_reload(store, tmp_path):
    cfg = TrainConfig(
        data_dir=str(store.root),
        train_range=("2018-01-01", "2018-02-28T18"),
        val_range=("2018-03-01", "2018-03-31T18"),
        model=TINY.to_dict(),
        stages=[{"rollout": 1, "epochs": 1, "lr": 1e-3}, {"rollout": 2, "epochs": 1, "lr": 1e-4}],
        batch_size=4,
        num_workers=0,
        steps_per_epoch=3,
        val_batches=2,
        warmup_steps=2,
        amp=False,
        device="cpu",
        out_dir=str(tmp_path / "run"),
    )
    path = train(cfg)
    model, extra = load_forecaster(path)
    assert "val_loss" in extra
    assert (tmp_path / "run" / "best.pt").exists()
    reloaded = Forecaster.from_store(store, TINY)
    save_forecaster(tmp_path / "x.pt", reloaded)
    again, _ = load_forecaster(tmp_path / "x.pt")
    x = torch.randn(1, len(store.names), 32, 64)
    f = torch.randn(1, N_DYNAMIC, 32, 64)
    assert torch.allclose(again.step(x, f), reloaded.eval().step(x, f))


def test_fair_crps_matches_numpy():
    from clima.metrics import crps_ensemble
    from clima.noise import fair_crps

    rng = np.random.default_rng(3)
    members = rng.normal(size=(5, 4, 6))
    truth = rng.normal(size=(4, 6))
    torch_value = fair_crps(torch.from_numpy(members), torch.from_numpy(truth)).numpy().mean()
    w = np.ones(4)
    assert np.isclose(torch_value, crps_ensemble(members, truth, w).mean())


def test_stochastic_model_spreads_without_ic_noise(store):
    cfg = AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2, noise_channels=2)
    model = Forecaster.from_store(store, cfg).eval()
    with torch.no_grad():
        for p in model.net.head.parameters():
            p.normal_(0, 0.05)
    x0 = torch.randn(len(store.names), 32, 64)
    forc = torch.from_numpy(dynamic_forcings(store, np.arange(0, 3)))
    traj = run_ensemble([model], x0, forc, EnsembleConfig(members=4, ic_scale=0.0, batch=4), seed=1)
    det = model.rollout(x0.unsqueeze(0), forc.unsqueeze(0))[0]
    assert torch.allclose(traj[0], det, atol=1e-5)
    assert traj[1:, -1].std(0).mean() > 0


def test_crps_stage_trains(store, tmp_path):
    cfg = TrainConfig(
        data_dir=str(store.root),
        train_range=("2018-01-01", "2018-02-28T18"),
        val_range=("2018-03-01", "2018-03-31T18"),
        model=AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2, noise_channels=2).to_dict(),
        stages=[{"rollout": 1, "epochs": 1, "lr": 1e-3}, {"rollout": 2, "epochs": 1, "lr": 1e-4, "loss": "crps", "members": 2}],
        batch_size=4,
        num_workers=0,
        steps_per_epoch=2,
        val_batches=2,
        warmup_steps=2,
        amp=False,
        device="cpu",
        out_dir=str(tmp_path / "crps"),
    )
    model, extra = load_forecaster(train(cfg))
    assert model.stochastic
    assert "val_crps" in extra
