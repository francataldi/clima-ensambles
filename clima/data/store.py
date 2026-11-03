from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..channels import Channel

STEP = np.timedelta64(6, "h")


class Era5Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.channels = [Channel(**c) for c in self.meta["channels"]]
        self.names = [c.name for c in self.channels]
        self.times = np.load(self.root / "times.npy").astype("datetime64[s]")
        coords = np.load(self.root / "coords.npz")
        self.lat = coords["lat"]
        self.lon = coords["lon"]
        self.static = np.load(self.root / "static.npy")
        self._fields = None
        stats_path = self.root / "stats.json"
        self.stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
        clim_path = self.root / "climatology.npy"
        self.climatology = np.load(clim_path, mmap_mode="r") if clim_path.exists() else None

    @property
    def fields(self) -> np.ndarray:
        if self._fields is None:
            self._fields = np.load(self.root / "fields.npy", mmap_mode="r")
        return self._fields

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_fields"] = None
        state["climatology"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        clim_path = self.root / "climatology.npy"
        self.climatology = np.load(clim_path, mmap_mode="r") if clim_path.exists() else None

    def index(self, name: str) -> int:
        return self.names.index(name)

    def time_range(self, start: str, end: str) -> np.ndarray:
        t0 = np.datetime64(start, "s")
        t1 = np.datetime64(end, "s")
        return np.flatnonzero((self.times >= t0) & (self.times <= t1))

    def valid_starts(self, indices: np.ndarray, steps: int) -> np.ndarray:
        if len(indices) <= steps:
            return np.array([], dtype=np.int64)
        idx = indices[: len(indices) - steps]
        ok = np.ones(len(idx), dtype=bool)
        for k in range(1, steps + 1):
            nxt = idx + k
            ok &= nxt < len(self.times)
            ok[ok] &= self.times[nxt[ok]] - self.times[idx[ok]] == k * STEP
        return idx[ok]

    def array(self, key: str) -> np.ndarray:
        return np.asarray(self.stats[key], dtype=np.float32)

    def clim_index(self, t: int) -> tuple[int, int]:
        ts = self.times[t]
        month = ts.astype("datetime64[M]").astype(int) % 12
        hour = int((ts - ts.astype("datetime64[D]")).astype(int) // 3600) // 6
        return month, hour

    def climatology_at(self, t: int) -> np.ndarray:
        m, h = self.clim_index(t)
        return np.asarray(self.climatology[m, h])

    def latitude_weights(self) -> np.ndarray:
        w = np.cos(np.deg2rad(self.lat))
        return (w / w.mean()).astype(np.float32)
