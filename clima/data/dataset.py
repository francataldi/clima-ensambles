from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..solar import SOLAR_CONSTANT, time_features, toa_insolation
from .store import Era5Store


def forcings_for_times(times: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    insolation = toa_insolation(times, lat, lon) / SOLAR_CONSTANT
    feats = time_features(times)
    planes = np.broadcast_to(feats[:, :, None, None], (len(times), feats.shape[1], len(lat), len(lon)))
    return np.concatenate([insolation[:, None], planes], axis=1).astype(np.float32)


def dynamic_forcings(store: Era5Store, t_indices: np.ndarray) -> np.ndarray:
    return forcings_for_times(store.times[t_indices], store.lat, store.lon)


N_DYNAMIC = 5


class ForecastDataset(Dataset):
    def __init__(self, store: Era5Store, start: str, end: str, rollout: int = 1, stride: int = 1) -> None:
        if store.stats is None:
            raise RuntimeError("faltan las estadisticas: corré scripts/compute_stats.py")
        self.store = store
        self.rollout = rollout
        self.starts = store.valid_starts(store.time_range(start, end), rollout)[::stride]
        self.mean = store.array("mean")[:, None, None]
        self.std = store.array("std")[:, None, None]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.starts[i])
        window = np.array(self.store.fields[t : t + self.rollout + 1], dtype=np.float32)
        states = (window - self.mean) / self.std
        forcings = dynamic_forcings(self.store, np.arange(t, t + self.rollout))
        return torch.from_numpy(states), torch.from_numpy(forcings), torch.tensor(t)
