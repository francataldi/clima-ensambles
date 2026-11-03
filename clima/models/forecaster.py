from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import N_DYNAMIC
from ..data.store import Era5Store
from .afno import AFNOConfig, AFNONet


def static_planes(store: Era5Store) -> np.ndarray:
    smean = np.asarray(store.stats["static_mean"], dtype=np.float32)[:, None, None]
    sstd = np.asarray(store.stats["static_std"], dtype=np.float32)[:, None, None]
    static = (store.static - smean) / sstd
    la = np.deg2rad(store.lat)[:, None]
    lo = np.deg2rad(store.lon)[None, :]
    H, W = len(store.lat), len(store.lon)
    geo = np.stack(
        [
            np.broadcast_to(np.sin(la), (H, W)),
            np.broadcast_to(np.cos(la), (H, W)),
            np.broadcast_to(np.sin(lo), (H, W)),
            np.broadcast_to(np.cos(lo), (H, W)),
        ]
    )
    return np.concatenate([static, geo]).astype(np.float32)


class Forecaster(nn.Module):
    def __init__(
        self,
        n_channels: int,
        static: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        diff_std: np.ndarray,
        lat: np.ndarray,
        cfg: AFNOConfig = AFNOConfig(),
    ) -> None:
        super().__init__()
        H, W = static.shape[1:]
        self.n_channels = n_channels
        self.cfg = cfg
        self.register_buffer("static", torch.from_numpy(static).float())
        self.register_buffer("mean", torch.from_numpy(np.asarray(mean, dtype=np.float32)).view(-1, 1, 1))
        self.register_buffer("std", torch.from_numpy(np.asarray(std, dtype=np.float32)).view(-1, 1, 1))
        self.register_buffer("tendency", torch.from_numpy(np.asarray(diff_std, dtype=np.float32) / np.asarray(std, dtype=np.float32)).view(-1, 1, 1))
        w = np.cos(np.deg2rad(lat))
        self.register_buffer("lat_weights", torch.from_numpy((w / w.mean()).astype(np.float32)).view(1, 1, -1, 1))
        self.lat = np.asarray(lat)
        self.net = AFNONet(n_channels + N_DYNAMIC + static.shape[0], n_channels, H, W, cfg)

    @classmethod
    def from_store(cls, store: Era5Store, cfg: AFNOConfig) -> "Forecaster":
        return cls(
            len(store.names),
            static_planes(store),
            store.array("mean"),
            store.array("std"),
            store.array("diff_std"),
            store.lat,
            cfg,
        )

    def step(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        static = self.static.unsqueeze(0).expand(B, -1, -1, -1).to(state.dtype)
        x = torch.cat([state, forcing.to(state.dtype), static], dim=1)
        delta = self.net(x)
        return state + delta.float() * self.tendency

    def rollout(self, state: torch.Tensor, forcings: torch.Tensor) -> torch.Tensor:
        out = [state]
        for k in range(forcings.shape[1]):
            state = self.step(state, forcings[:, k])
            out.append(state)
        return torch.stack(out, dim=1)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def loss(self, pred: torch.Tensor, target: torch.Tensor, channel_weights: torch.Tensor | None = None) -> torch.Tensor:
        err = (pred - target) / self.tendency
        sq = err.pow(2) * self.lat_weights
        per_channel = sq.mean(dim=(0, 2, 3))
        if channel_weights is not None:
            per_channel = per_channel * channel_weights
        return per_channel.mean()


def enable_mc_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def save_forecaster(path, model: Forecaster, extra: dict | None = None) -> None:
    torch.save(
        {
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "cfg": model.cfg.to_dict(),
            "n_channels": model.n_channels,
            "static_shape": list(model.static.shape),
            "lat": model.lat.tolist(),
            "extra": extra or {},
        },
        path,
    )


def load_forecaster(path, device: str | torch.device = "cpu") -> tuple[Forecaster, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"]
    cfg = AFNOConfig(**payload["cfg"])
    model = Forecaster(
        payload["n_channels"],
        state["static"].numpy(),
        state["mean"].view(-1).numpy(),
        state["std"].view(-1).numpy(),
        (state["tendency"].view(-1) * state["std"].view(-1)).numpy(),
        np.asarray(payload["lat"]),
        cfg,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload.get("extra", {})
