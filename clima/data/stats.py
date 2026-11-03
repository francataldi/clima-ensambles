from __future__ import annotations

import json

import numpy as np

from .store import STEP, Era5Store


def compute_stats(store: Era5Store, start: str, end: str, chunk: int = 512, verbose: bool = True) -> dict:
    idx = store.time_range(start, end)
    fields = store.fields
    T, C, H, W = fields.shape
    w = store.latitude_weights()[None, :, None]
    s1 = np.zeros(C)
    s2 = np.zeros(C)
    d2 = np.zeros(C)
    n = 0
    nd = 0
    clim_sum = np.zeros((12, 4, C, H, W))
    clim_count = np.zeros((12, 4))
    months = (store.times.astype("datetime64[M]").astype(int)) % 12
    hours = ((store.times - store.times.astype("datetime64[D]")).astype(int) // 3600) // 6
    for k in range(0, len(idx), chunk):
        sel = idx[k : k + chunk]
        block = np.asarray(fields[sel[0] : sel[-1] + 1], dtype=np.float64)
        block = block[sel - sel[0]]
        s1 += (block * w).mean(axis=(2, 3)).sum(0)
        s2 += ((block**2) * w).mean(axis=(2, 3)).sum(0)
        n += len(sel)
        contiguous = store.times[sel[1:]] - store.times[sel[:-1]] == STEP
        if contiguous.any():
            diff = block[1:][contiguous] - block[:-1][contiguous]
            d2 += ((diff**2) * w).mean(axis=(2, 3)).sum(0)
            nd += int(contiguous.sum())
        for m in range(12):
            for h in range(4):
                mask = (months[sel] == m) & (hours[sel] == h)
                if mask.any():
                    clim_sum[m, h] += block[mask].sum(0)
                    clim_count[m, h] += mask.sum()
        if verbose and (k // chunk) % 10 == 0:
            print(f"estadisticas: {k + len(sel)}/{len(idx)}", flush=True)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-12))
    diff_std = np.sqrt(np.maximum(d2 / max(nd, 1), 1e-12))
    static = store.static.astype(np.float64)
    stats = {
        "names": store.names,
        "range": [start, end],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "diff_std": diff_std.tolist(),
        "static_mean": static.mean(axis=(1, 2)).tolist(),
        "static_std": np.maximum(static.std(axis=(1, 2)), 1e-6).tolist(),
    }
    climatology = (clim_sum / np.maximum(clim_count, 1)[:, :, None, None, None]).astype(np.float32)
    empty = clim_count == 0
    if empty.any():
        overall = (clim_sum.sum((0, 1)) / max(clim_count.sum(), 1)).astype(np.float32)
        climatology[empty] = overall
    (store.root / "stats.json").write_text(json.dumps(stats, indent=2))
    np.save(store.root / "climatology.npy", climatology)
    store.stats = stats
    store.climatology = np.load(store.root / "climatology.npy", mmap_mode="r")
    return stats
