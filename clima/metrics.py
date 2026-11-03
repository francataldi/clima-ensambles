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
