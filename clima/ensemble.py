from __future__ import annotations

from dataclasses import dataclass

import torch

from .models.forecaster import Forecaster, enable_mc_dropout


@dataclass(frozen=True)
class EnsembleConfig:
    members: int = 50
    ic_scale: float = 0.5
    length_scale: float = 6.0
    mc_dropout: bool = False
    batch: int = 25


def smooth_noise(shape: tuple[int, ...], length_scale: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    M, C, H, W = shape
    white = torch.randn(shape, generator=generator, device="cpu").to(device)
    spec = torch.fft.rfft2(white, dim=(-2, -1))
    ky = torch.fft.fftfreq(H, device=device)[:, None] * H
    kx = torch.fft.rfftfreq(W, device=device)[None, :] * W
    k2 = ky**2 + kx**2
    filt = torch.exp(-k2 / (2 * (max(H, W) / length_scale) ** 2))
    field = torch.fft.irfft2(spec * filt, s=(H, W), dim=(-2, -1))
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    return field / field.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def perturb(model: Forecaster, state: torch.Tensor, cfg: EnsembleConfig, generator: torch.Generator, keep_control: bool) -> torch.Tensor:
    if cfg.ic_scale <= 0:
        return state
    noise = smooth_noise(tuple(state.shape), cfg.length_scale, generator, state.device)
    noise = noise * cfg.ic_scale * model.tendency
    if keep_control:
        noise[0] = 0.0
    return state + noise


@torch.no_grad()
def run_ensemble(
    models: list[Forecaster],
    state: torch.Tensor,
    forcings: torch.Tensor,
    cfg: EnsembleConfig,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    device = state.device
    steps = forcings.shape[0]
    outs = []
    for start in range(0, cfg.members, cfg.batch):
        n = min(cfg.batch, cfg.members - start)
        members = []
        for j in range(n):
            members.append(models[(start + j) % len(models)])
        x0 = state.unsqueeze(0).expand(n, -1, -1, -1).clone()
        x0 = perturb(models[0], x0, cfg, generator, keep_control=start == 0)
        f = forcings.unsqueeze(0).expand(n, -1, -1, -1, -1)
        result = torch.empty((n, steps + 1, *state.shape), device=device)
        for model in set(members):
            sel = [j for j, m in enumerate(members) if m is model]
            if cfg.mc_dropout:
                enable_mc_dropout(model)
            idx = torch.tensor(sel, device=device)
            result[idx] = model.rollout(x0[idx], f[idx])
            model.eval()
        outs.append(result)
    return torch.cat(outs, dim=0)
