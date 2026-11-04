import numpy as np
import pytest
import torch

from clima.ensemble import EnsembleConfig
from clima.evaluate import evaluate, save_report
from clima.events import EVENTS, REGIONS, THRESHOLDS_FILE, EventScores, derived_field, load_thresholds
from clima.metrics import (
    brier_decomposition_from_histogram,
    brier_from_histogram,
    brier_score,
    brier_skill_score,
    climatology_brier_from_histogram,
    fair_brier_from_histogram,
    merge_histogram,
    probability_histogram,
    rank_histogram,
    reliability_from_histogram,
    roc_auc,
    roc_from_histogram,
)
from clima.models.afno import AFNOConfig
from clima.models.forecaster import Forecaster
from clima.plots import plot_event_scores, plot_reliability, plot_roc

P_CLIM = 0.1


def scores(prob, obs, weights=None, bins=11):
    hist = probability_histogram(prob, obs, weights, bins)
    bs = brier_from_histogram(hist)
    ref = climatology_brier_from_histogram(hist, P_CLIM)
    return hist, float(bs), float(brier_skill_score(bs, ref)), float(roc_auc(*roc_from_histogram(hist)))


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    n = 100_000
    return rng, rng.random(n) < P_CLIM, rng.uniform(0.2, 1.0, n)


def test_perfect_forecast(sample):
    _, obs, w = sample
    hist, bs, bss, auc = scores(obs.astype(float), obs, w)
    assert bs == 0.0
    assert bss == 1.0
    assert auc == 1.0
    prob, freq, weight = reliability_from_histogram(hist)
    ok = weight > 0
    assert np.allclose(prob[ok], freq[ok])


def test_climatological_forecast(sample):
    _, obs, w = sample
    hist, bs, bss, auc = scores(np.full(obs.shape, P_CLIM), obs, w)
    assert abs(bss) < 1e-12
    assert auc == pytest.approx(0.5, abs=1e-12)
    assert bs == pytest.approx(brier_score(np.full(obs.shape, P_CLIM), obs, w))
    prob, _, weight = reliability_from_histogram(hist)
    assert (weight > 0).sum() == 1
    assert np.nanmax(prob) == pytest.approx(P_CLIM)


def test_random_forecast(sample):
    rng, obs, w = sample
    prob = rng.random(obs.shape)
    hist, bs, bss, auc = scores(prob, obs, w)
    assert auc == pytest.approx(0.5, abs=0.02)
    assert bss < -2.0
    assert bs == pytest.approx(brier_score(prob, obs, w), rel=1e-10)
    pofd, pod = roc_from_histogram(hist)
    assert (pofd[0], pod[0], pofd[-1], pod[-1]) == (1.0, 1.0, 0.0, 0.0)
    assert np.all(np.diff(pofd) <= 0) and np.all(np.diff(pod) <= 0)
    assert roc_auc(pofd[::-1], pod[::-1]) == pytest.approx(auc)


def test_ensemble_drawn_from_climatology_has_zero_fair_skill():
    rng = np.random.default_rng(1)
    members, n = 10, 400_000
    threshold = 1.2815515655446004
    counts = (rng.normal(size=(members, n)) > threshold).sum(0)
    obs = rng.normal(size=n) > threshold
    hist = probability_histogram(counts / members, obs, None, members + 1)
    ref = climatology_brier_from_histogram(hist, P_CLIM)
    assert float(brier_skill_score(brier_from_histogram(hist), ref)) == pytest.approx(-1.0 / members, abs=0.01)
    assert float(brier_skill_score(fair_brier_from_histogram(hist, members), ref)) == pytest.approx(0.0, abs=0.01)
    direct = np.mean((counts / members - obs) ** 2 - counts * (members - counts) / (members**2 * (members - 1)))
    assert float(fair_brier_from_histogram(hist, members)) == pytest.approx(direct)
    assert float(roc_auc(*roc_from_histogram(hist))) == pytest.approx(0.5, abs=0.01)


def test_calibrated_probabilities_are_reliable():
    rng = np.random.default_rng(2)
    prob = rng.beta(0.6, 3.0, 300_000)
    obs = rng.random(prob.shape) < prob
    hist = probability_histogram(prob, obs, None, 10)
    prob_bin, freq, weight = reliability_from_histogram(hist)
    ok = weight > 3000
    assert ok.sum() >= 6
    assert np.max(np.abs(prob_bin[ok] - freq[ok])) < 0.02
    rel, res, unc = brier_decomposition_from_histogram(hist)
    assert float(rel) < 1e-3
    assert float(brier_from_histogram(hist)) == pytest.approx(float(rel - res + unc), abs=2e-3)


def test_exchangeable_ensemble_is_calibrated_and_decomposes():
    rng = np.random.default_rng(4)
    members, n = 20, 200_000
    signal = rng.normal(size=n)
    truth = signal + rng.normal(size=n)
    ens = signal[None] + rng.normal(size=(members, n))
    assert np.allclose(rank_histogram(ens, truth) / n, 1 / (members + 1), atol=0.004)
    spread = np.sqrt(ens.var(axis=0, ddof=1).mean() * (members + 1) / members)
    skill = np.sqrt(((ens.mean(axis=0) - truth) ** 2).mean())
    assert spread / skill == pytest.approx(1.0, abs=0.02)
    threshold = np.quantile(truth, 0.9)
    hist = probability_histogram((ens > threshold).mean(0), truth > threshold, None, members + 1)
    rel, res, unc = brier_decomposition_from_histogram(hist)
    assert float(brier_from_histogram(hist)) == pytest.approx(float(rel - res + unc), abs=1e-12)
    _, freq, weight = reliability_from_histogram(hist, groups=11)
    assert np.all(np.diff(freq[weight > 2000]) > 0)
    assert 0.8 < float(roc_auc(*roc_from_histogram(hist))) < 1.0


def test_histogram_helpers():
    rng = np.random.default_rng(3)
    prob = rng.random((3, 500))
    obs = rng.random((3, 500)) < 0.3
    hist = probability_histogram(prob, obs, None, 21)
    assert hist.shape == (3, 21, 5)
    assert np.allclose(hist[..., :2].sum(axis=(-2, -1)), 500)
    merged = merge_histogram(hist, 11)
    assert merged.shape == (3, 11, 5)
    assert np.allclose(merged.sum(-2), hist.sum(-2))
    for k in range(3):
        assert float(brier_from_histogram(hist[k])) == pytest.approx(brier_score(prob[k], obs[k]))
    empty = probability_histogram(np.full(50, 0.2), np.zeros(50, dtype=bool), None, 11)
    assert np.isnan(roc_auc(*roc_from_histogram(empty)))
    assert np.isfinite(brier_skill_score(brier_from_histogram(empty), climatology_brier_from_histogram(empty, P_CLIM)))


def test_regions_partition_globe_and_cover_south_america(store):
    masks = {name: r.mask(store.lat, store.lon) for name, r in REGIONS.items()}
    bands = masks["tropicos"].astype(int) + masks["extratropicos_norte"] + masks["extratropicos_sur"]
    assert np.all(bands == 1)
    assert masks["global"].all()
    sa = masks["sudamerica"]
    i = np.argmin(np.abs(store.lat - (-34.6)))
    j = np.argmin(np.abs(store.lon - (360 - 58.4)))
    assert sa[i, j]
    assert not sa[:, np.argmin(np.abs(store.lon - 20.0))].any()
    assert sa.sum() == 96


def test_thresholds_match_climatological_frequency(store):
    events = list(EVENTS.values())
    thresholds = load_thresholds(store, events, verbose=False)
    assert (store.root / THRESHOLDS_FILE).exists()
    again = load_thresholds(store, events, verbose=False)
    assert all(np.array_equal(thresholds[k], again[k]) for k in thresholds)
    idx = store.time_range(*store.stats["range"])
    fields = np.asarray(store.fields[idx[0] : idx[-1] + 1])
    for ev in events:
        thr = np.stack([thresholds[ev.key][store.clim_index(int(t))] for t in idx])
        values = derived_field(fields, store.names, ev.variable)
        assert np.mean(values > thr) == pytest.approx(1 - ev.quantile, abs=0.01)
        assert np.isfinite(thresholds[ev.key]).all()
    ws = derived_field(fields[:2], store.names, "ws10")
    u, v = fields[:2, store.index("u10")], fields[:2, store.index("v10")]
    assert np.allclose(ws, np.sqrt(u**2 + v**2))


def test_event_scores_perfect_ensemble(store):
    t = 10
    steps = 3
    truth = np.asarray(store.fields[t : t + steps + 1], dtype=np.float32)
    acc = EventScores(store, 5, steps, verbose=False)
    acc.update(t, np.repeat(truth[None], 5, axis=0), truth)
    summary = acc.summary()
    for name in EVENTS:
        glob = summary[name]["regions"]["global"]
        assert np.allclose(glob["bss"], 1.0)
        assert np.allclose(glob["auc"], 1.0)
        assert np.allclose(glob["reliability"], 0.0)


def test_evaluate_reports_events_and_plots(store, tmp_path):
    torch.manual_seed(0)
    model = Forecaster.from_store(store, AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2)).eval()
    result = evaluate([model], store, "2018-03-01", "2018-03-20", steps=4, every=4, ens=EnsembleConfig(members=4, batch=4), max_inits=2, verbose=False)
    assert set(result["events"]) == set(EVENTS)
    assert set(result["regions"]) == set(REGIONS)
    for event in result["events"].values():
        for region in event["regions"].values():
            assert len(region["bss"]) == 5
            assert np.asarray(region["hist"]).shape == (5, 5, 5)
            auc = [a for a in region["auc"] if a is not None]
            assert all(0.0 <= a <= 1.0 for a in auc)
    out = save_report(result, tmp_path / "eval")
    lines = (out / "eventos.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + len(EVENTS) * len(REGIONS) * 4
    assert lines[0].startswith("evento,region,lead_h,base_rate,brier")
    for plot in (plot_event_scores, plot_reliability, plot_roc):
        assert plot(result, out).stat().st_size > 10_000
    plain = evaluate([model], store, "2018-03-01", "2018-03-20", steps=4, every=4, ens=EnsembleConfig(members=4, batch=4), max_inits=1, verbose=False, event_metrics=False)
    assert "events" not in plain
