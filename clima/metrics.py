from __future__ import annotations

import numpy as np


def lat_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (x * w[:, None]).mean(axis=(-2, -1))


def rmse(forecast: np.ndarray, truth: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sqrt(weighted_mean((forecast - truth) ** 2, w))


def acc(forecast: np.ndarray, truth: np.ndarray, clim: np.ndarray, w: np.ndarray) -> np.ndarray:
    fa = forecast - clim
    oa = truth - clim
    fa = fa - weighted_mean(fa, w)[..., None, None]
    oa = oa - weighted_mean(oa, w)[..., None, None]
    num = weighted_mean(fa * oa, w)
    den = np.sqrt(weighted_mean(fa**2, w) * weighted_mean(oa**2, w))
    return num / np.maximum(den, 1e-12)


def crps_ensemble(members: np.ndarray, truth: np.ndarray, w: np.ndarray, fair: bool = True) -> np.ndarray:
    M = members.shape[0]
    skill = np.abs(members - truth[None]).mean(axis=0)
    x = np.sort(members, axis=0)
    ranks = np.arange(1, M + 1, dtype=np.float64).reshape(-1, *([1] * (x.ndim - 1)))
    pair_sum = 2.0 * ((2 * ranks - M - 1) * x).sum(axis=0)
    denom = 2.0 * M * (M - 1) if fair and M > 1 else 2.0 * M * M
    spread = pair_sum / denom
    return weighted_mean(skill - spread, w)


def ensemble_spread(members: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = members.shape[0]
    var = members.var(axis=0, ddof=1) if M > 1 else np.zeros_like(members[0])
    return np.sqrt(weighted_mean(var, w) * (M + 1) / M)


def rank_histogram(members: np.ndarray, truth: np.ndarray, bins: int | None = None) -> np.ndarray:
    M = members.shape[0]
    ranks = (members < truth[None]).sum(axis=0).ravel()
    return np.bincount(ranks, minlength=M + 1)[: (bins or M + 1)]


def rank_counts(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    M = members.shape[0]
    below = (members < truth[None]).sum(axis=0)
    lead = below.shape[:-2]
    groups = int(np.prod(lead, dtype=np.int64))
    flat = below.reshape(groups, -1) + (np.arange(groups) * (M + 1))[:, None]
    return np.bincount(flat.ravel(), minlength=groups * (M + 1)).reshape(*lead, M + 1)


def rank_outlier_ratio(hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.float64)
    p = _ratio(hist, hist.sum(-1, keepdims=True))
    return (p[..., 0] + p[..., -1]) * hist.shape[-1] / 2


def rank_reliability_index(hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.float64)
    p = _ratio(hist, hist.sum(-1, keepdims=True))
    return np.abs(p - 1.0 / hist.shape[-1]).sum(-1)


def bilinear(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, plat: float, plon: float) -> np.ndarray:
    plon = plon % 360.0
    i = np.clip(np.searchsorted(lat, plat) - 1, 0, len(lat) - 2)
    dlon = lon[1] - lon[0]
    j = int(np.floor((plon - lon[0]) / dlon)) % len(lon)
    j2 = (j + 1) % len(lon)
    ty = (plat - lat[i]) / (lat[i + 1] - lat[i])
    tx = ((plon - lon[j]) % 360.0) / dlon
    top = field[..., i, j] * (1 - tx) + field[..., i, j2] * tx
    bottom = field[..., i + 1, j] * (1 - tx) + field[..., i + 1, j2] * tx
    return top * (1 - ty) + bottom * ty


def _ratio(num, den) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=np.broadcast_to(den, out.shape) > 0)
    return out


def brier_score(prob: np.ndarray, obs: np.ndarray, weights: np.ndarray | None = None) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    w = np.ones_like(prob) if weights is None else np.broadcast_to(np.asarray(weights, dtype=np.float64), prob.shape)
    return float(_ratio(((prob - obs) ** 2 * w).sum(), w.sum()))


def brier_skill_score(score, reference) -> np.ndarray:
    return 1.0 - _ratio(score, reference)


def probability_histogram(prob: np.ndarray, obs: np.ndarray, weights: np.ndarray | None = None, bins: int = 10) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    n = prob.shape[-1]
    lead_shape = prob.shape[:-1]
    p = prob.reshape(-1, n)
    o = np.asarray(obs).reshape(-1, n).astype(bool)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).reshape(n)
    groups = p.shape[0]
    idx = (np.clip(np.floor(p * bins).astype(np.int64), 0, bins - 1) + bins * np.arange(groups)[:, None]).ravel()
    ww = np.broadcast_to(w, p.shape).ravel()
    pr = p.ravel()
    ob = o.ravel()
    size = groups * bins
    hist = np.stack(
        [
            np.bincount(idx, ww * ~ob, size),
            np.bincount(idx, ww * ob, size),
            np.bincount(idx, ww * pr * ~ob, size),
            np.bincount(idx, ww * pr * ob, size),
            np.bincount(idx, ww * pr * pr, size),
        ],
        axis=-1,
    )
    return hist.reshape(*lead_shape, bins, 5)


def merge_histogram(hist: np.ndarray, groups: int) -> np.ndarray:
    bins = hist.shape[-2]
    if bins <= groups:
        return hist
    target = np.rint(np.arange(bins) * (groups - 1) / (bins - 1)).astype(int)
    out = np.zeros(hist.shape[:-2] + (groups, hist.shape[-1]))
    for b in range(bins):
        out[..., target[b], :] += hist[..., b, :]
    return out


def base_rate_from_histogram(hist: np.ndarray) -> np.ndarray:
    return _ratio(hist[..., 1].sum(-1), hist[..., :2].sum(axis=(-2, -1)))


def brier_from_histogram(hist: np.ndarray) -> np.ndarray:
    return _ratio((hist[..., 4] - 2 * hist[..., 3] + hist[..., 1]).sum(-1), hist[..., :2].sum(axis=(-2, -1)))


def fair_brier_from_histogram(hist: np.ndarray, members: int) -> np.ndarray:
    total = hist[..., :2].sum(axis=(-2, -1))
    spread = (hist[..., 2] + hist[..., 3] - hist[..., 4]).sum(-1)
    return brier_from_histogram(hist) - _ratio(spread, total * max(members - 1, 1))


def climatology_brier_from_histogram(hist: np.ndarray, p_clim: float) -> np.ndarray:
    w0 = hist[..., 0].sum(-1)
    w1 = hist[..., 1].sum(-1)
    return _ratio(w0 * p_clim**2 + w1 * (1.0 - p_clim) ** 2, w0 + w1)


def brier_decomposition_from_histogram(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = hist[..., 0] + hist[..., 1]
    total = w.sum(-1)
    prob = np.nan_to_num(_ratio(hist[..., 2] + hist[..., 3], w))
    freq = np.nan_to_num(_ratio(hist[..., 1], w))
    base = base_rate_from_histogram(hist)
    reliability = _ratio((w * (prob - freq) ** 2).sum(-1), total)
    resolution = _ratio((w * (freq - base[..., None]) ** 2).sum(-1), total)
    return reliability, resolution, base * (1.0 - base)


def reliability_from_histogram(hist: np.ndarray, groups: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if groups is not None:
        hist = merge_histogram(hist, groups)
    w = hist[..., 0] + hist[..., 1]
    return _ratio(hist[..., 2] + hist[..., 3], w), _ratio(hist[..., 1], w), w


def roc_from_histogram(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w0 = hist[..., 0]
    w1 = hist[..., 1]
    hits = np.flip(np.cumsum(np.flip(w1, -1), -1), -1)
    false = np.flip(np.cumsum(np.flip(w0, -1), -1), -1)
    zeros = np.zeros(w0.shape[:-1] + (1,))
    pod = _ratio(np.concatenate([hits, zeros], -1), w1.sum(-1, keepdims=True))
    pofd = _ratio(np.concatenate([false, zeros], -1), w0.sum(-1, keepdims=True))
    return pofd, pod


def roc_auc(pofd: np.ndarray, pod: np.ndarray) -> np.ndarray:
    pofd = np.asarray(pofd, dtype=np.float64)
    pod = np.asarray(pod, dtype=np.float64)
    return np.abs(0.5 * ((pofd[..., :-1] - pofd[..., 1:]) * (pod[..., :-1] + pod[..., 1:])).sum(-1))
