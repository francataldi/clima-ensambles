from __future__ import annotations

import torch


def smooth_noise(
    shape: tuple[int, ...],
    length_scale: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    B, C, H, W = shape
    device = torch.device(device)
    if generator is None:
        white = torch.randn(shape, device=device)
    else:
        white = torch.randn(shape, generator=generator).to(device)
    spec = torch.fft.rfft2(white, dim=(-2, -1))
    ky = torch.fft.fftfreq(H, device=device)[:, None] * H
    kx = torch.fft.rfftfreq(W, device=device)[None, :] * W
    k2 = ky**2 + kx**2
    filt = torch.exp(-k2 / (2 * (max(H, W) / length_scale) ** 2))
    field = torch.fft.irfft2(spec * filt, s=(H, W), dim=(-2, -1))
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    return field / field.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def fair_crps(members: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M = members.shape[0]
    skill = (members - target.unsqueeze(0)).abs().mean(0)
    if M == 1:
        return skill
    spread = torch.zeros_like(skill)
    for i in range(M):
        for j in range(i + 1, M):
            spread = spread + (members[i] - members[j]).abs()
    return skill - spread / (M * (M - 1))
