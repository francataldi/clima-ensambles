from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .data.store import Era5Store
from .metrics import (
    base_rate_from_histogram,
    brier_decomposition_from_histogram,
    brier_from_histogram,
    brier_skill_score,
    climatology_brier_from_histogram,
    fair_brier_from_histogram,
    probability_histogram,
    roc_auc,
    roc_from_histogram,
)

THRESHOLDS_FILE = "event_thresholds.npz"


@dataclass(frozen=True)
class Event:
    name: str
    variable: str
    quantile: float
    label: str

    @property
    def key(self) -> str:
        return f"{self.variable}_q{round(self.quantile * 100):d}"


@dataclass(frozen=True)
class Region:
    name: str
    label: str
    lat: tuple[float, float]
    lon: tuple[float, float]

    def mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        la = (lat >= self.lat[0]) & (lat <= self.lat[1])
        lo0, lo1 = self.lon
        if lo1 - lo0 >= 360:
            lo = np.ones(len(lon), dtype=bool)
        else:
            a, b, x = lo0 % 360, lo1 % 360, np.asarray(lon) % 360
            lo = (x >= a) & (x <= b) if a <= b else (x >= a) | (x <= b)
        return la[:, None] & lo[None, :]


EVENTS = {
    "t2m_p90": Event("t2m_p90", "t2m", 0.9, "t2m > p90 climatológico"),
    "ws10_p90": Event("ws10_p90", "ws10", 0.9, "viento a 10 m > p90 climatológico"),
}

REGIONS = {
    "global": Region("global", "Global", (-90.0, 90.0), (0.0, 360.0)),
    "tropicos": Region("tropicos", "Trópicos", (-20.0, 20.0), (0.0, 360.0)),
    "extratropicos_norte": Region("extratropicos_norte", "Extratrópicos norte", (20.0, 90.0), (0.0, 360.0)),
    "extratropicos_sur": Region("extratropicos_sur", "Extratrópicos sur", (-90.0, -20.0), (0.0, 360.0)),
    "sudamerica": Region("sudamerica", "Sudamérica", (-56.0, 13.0), (278.0, 326.0)),
}


def sources(names: list[str], variable: str) -> list[str]:
    if variable in names:
        return [variable]
    if variable.startswith("ws") and f"u{variable[2:]}" in names and f"v{variable[2:]}" in names:
        return [f"u{variable[2:]}", f"v{variable[2:]}"]
    raise KeyError(f"no se puede construir {variable} con los canales disponibles")


def derived_field(x: np.ndarray, names: list[str], variable: str) -> np.ndarray:
    parts = sources(names, variable)
    if len(parts) == 1:
        return x[..., names.index(parts[0]), :, :]
    u, v = (x[..., names.index(p), :, :] for p in parts)
    return np.hypot(u, v)


def _read_rows(fields: np.ndarray, rows: np.ndarray, channels: list[int]) -> np.ndarray:
    runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
    return np.concatenate([np.asarray(fields[r[0] : r[-1] + 1][:, channels], dtype=np.float32) for r in runs])


def compute_thresholds(store: Era5Store, start: str, end: str, events: list[Event], verbose: bool = True) -> dict[str, np.ndarray]:
    idx = store.time_range(start, end)
    if len(idx) == 0:
        raise ValueError(f"no hay datos entre {start} y {end} para calcular umbrales")
    times = store.times[idx]
    months = times.astype("datetime64[M]").astype(int) % 12
    hours = ((times - times.astype("datetime64[D]")).astype(int) // 3600) // 6
    needed = sorted({store.index(s) for ev in events for s in sources(store.names, ev.variable)})
    sub = [store.names[c] for c in needed]
    shape = (12, 4, len(store.lat), len(store.lon))
    out = {ev.key: np.full(shape, np.nan, dtype=np.float32) for ev in events}
    for m in range(12):
        sel = months == m
        if not sel.any():
            continue
        block = _read_rows(store.fields, idx[sel], needed)
        hm = hours[sel]
        for ev in events:
            field = derived_field(block, sub, ev.variable)
            for h in range(4):
                mask = hm == h
                if mask.any():
                    out[ev.key][m, h] = np.quantile(field[mask], ev.quantile, axis=0)
        if verbose:
            print(f"umbrales climatologicos: mes {m + 1}/12", flush=True)
    for arr in out.values():
        empty = np.isnan(arr).all(axis=(-2, -1))
        if empty.any():
            arr[empty] = np.nanmean(arr[~empty], axis=0)
    return out


def load_thresholds(store: Era5Store, events: list[Event], start: str | None = None, end: str | None = None, verbose: bool = True) -> dict[str, np.ndarray]:
    if start is None or end is None:
        default = store.stats["range"] if store.stats else [str(store.times[0]), str(store.times[-1])]
        start, end = start or default[0], end or default[1]
    path = store.root / THRESHOLDS_FILE
    meta = {"start": start, "end": end}
    cached: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if "meta" in z.files and json.loads(str(z["meta"])) == meta:
                cached = {k: np.array(z[k]) for k in z.files if k != "meta"}
    missing = [ev for ev in events if ev.key not in cached]
    if missing:
        if verbose:
            print(f"calculando umbrales climatologicos {', '.join(ev.key for ev in missing)} con {start} a {end}", flush=True)
        cached.update(compute_thresholds(store, start, end, missing, verbose))
        np.savez(path, meta=np.array(json.dumps(meta)), **cached)
    return {ev.key: cached[ev.key] for ev in events}


class EventScores:
    def __init__(
        self,
        store: Era5Store,
        members: int,
        steps: int,
        events: list[Event] | None = None,
        regions: list[Region] | None = None,
        thresholds: dict[str, np.ndarray] | None = None,
        verbose: bool = True,
    ) -> None:
        self.store = store
        self.events = list(events or EVENTS.values())
        self.regions = list(regions or REGIONS.values())
        self.members = members
        self.thresholds = thresholds if thresholds is not None else load_thresholds(store, self.events, verbose=verbose)
        w = np.broadcast_to(store.latitude_weights()[:, None], (len(store.lat), len(store.lon)))
        self.masks = [r.mask(store.lat, store.lon) for r in self.regions]
        self.weights = [w[m] for m in self.masks]
        self.hist = np.zeros((len(self.events), len(self.regions), steps + 1, members + 1, 5))

    def update(self, t: int, members: np.ndarray, truth: np.ndarray) -> None:
        names = self.store.names
        index = [self.store.clim_index(t + k) for k in range(truth.shape[0])]
        for ei, ev in enumerate(self.events):
            thr = np.stack([self.thresholds[ev.key][m, h] for m, h in index])
            prob = (derived_field(members, names, ev.variable) > thr[None]).mean(axis=0)
            obs = derived_field(truth, names, ev.variable) > thr
            for ri, mask in enumerate(self.masks):
                self.hist[ei, ri] += probability_histogram(prob[:, mask], obs[:, mask], self.weights[ri], self.members + 1)

    def summary(self) -> dict:
        out = {}
        for ei, ev in enumerate(self.events):
            regions = {}
            for ri, region in enumerate(self.regions):
                h = self.hist[ei, ri]
                bs = brier_from_histogram(h)
                fair = fair_brier_from_histogram(h, self.members)
                ref = climatology_brier_from_histogram(h, 1.0 - ev.quantile)
                rel, res, unc = brier_decomposition_from_histogram(h)
                regions[region.name] = {
                    "base_rate": base_rate_from_histogram(h),
                    "brier": bs,
                    "brier_fair": fair,
                    "brier_clim": ref,
                    "bss": brier_skill_score(bs, ref),
                    "bss_fair": brier_skill_score(fair, ref),
                    "reliability": rel,
                    "resolution": res,
                    "uncertainty": unc,
                    "auc": roc_auc(*roc_from_histogram(h)),
                    "hist": h,
                }
            out[ev.name] = {"label": ev.label, "variable": ev.variable, "quantile": ev.quantile, "regions": regions}
        return out
