from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AFNOConfig:
    dim: int = 256
    depth: int = 8
    num_blocks: int = 8
    patch: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path: float = 0.0
    sparsity: float = 0.01
    mode_fraction: float = 1.0
    noise_channels: int = 0
    noise_length_scale: float = 6.0

    def to_dict(self) -> dict:
        return asdict(self)


class AFNO2D(nn.Module):
    def __init__(self, dim: int, num_blocks: int, sparsity: float, mode_fraction: float) -> None:
        super().__init__()
        if dim % num_blocks:
            raise ValueError("dim must be divisible by num_blocks")
        self.dim = dim
        self.num_blocks = num_blocks
        self.block = dim // num_blocks
        self.sparsity = sparsity
        self.mode_fraction = mode_fraction
        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block, self.block))
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block))
        self.w2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block, self.block))
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        dtype = x.dtype
        x = x.float()
        B, H, W, D = x.shape
        z = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        z = z.reshape(B, H, W // 2 + 1, self.num_blocks, self.block)
        kept = max(1, int((W // 2 + 1) * self.mode_fraction))
        zk = z[:, :, :kept]
        w1, b1, w2, b2 = self.w1.float(), self.b1.float(), self.w2.float(), self.b2.float()
        r = F.relu(torch.einsum("...bi,bio->...bo", zk.real, w1[0]) - torch.einsum("...bi,bio->...bo", zk.imag, w1[1]) + b1[0])
        i = F.relu(torch.einsum("...bi,bio->...bo", zk.imag, w1[0]) + torch.einsum("...bi,bio->...bo", zk.real, w1[1]) + b1[1])
        r2 = torch.einsum("...bi,bio->...bo", r, w2[0]) - torch.einsum("...bi,bio->...bo", i, w2[1]) + b2[0]
        i2 = torch.einsum("...bi,bio->...bo", i, w2[0]) + torch.einsum("...bi,bio->...bo", r, w2[1]) + b2[1]
        out = torch.stack([r2, i2], dim=-1)
        out = F.softshrink(out, lambd=self.sparsity)
        zk = torch.view_as_complex(out.contiguous())
        z = torch.zeros_like(z)
        z[:, :, :kept] = zk
        z = z.reshape(B, H, W // 2 + 1, D)
        x = torch.fft.irfft2(z, s=(H, W), dim=(1, 2), norm="ortho")
        return x.to(dtype) + residual


class DropPath(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, cfg: AFNOConfig, drop_path: float) -> None:
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.filter = AFNO2D(cfg.dim, cfg.num_blocks, cfg.sparsity, cfg.mode_fraction)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.dim),
            nn.Dropout(cfg.dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.filter(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class AFNONet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, height: int, width: int, cfg: AFNOConfig) -> None:
        super().__init__()
        if height % cfg.patch or width % cfg.patch:
            raise ValueError("grid must be divisible by patch")
        self.cfg = cfg
        self.out_channels = out_channels
        self.h = height // cfg.patch
        self.w = width // cfg.patch
        self.embed = nn.Conv2d(in_channels, cfg.dim, kernel_size=cfg.patch, stride=cfg.patch)
        self.pos = nn.Parameter(torch.zeros(1, self.h, self.w, cfg.dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        rates = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()
        self.blocks = nn.ModuleList([Block(cfg, r) for r in rates])
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, out_channels * cfg.patch * cfg.patch, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.cfg.patch
        x = self.embed(x).permute(0, 2, 3, 1) + self.pos
        for block in self.blocks:
            x = block(x)
        x = self.head(self.norm(x))
        x = x.reshape(B, self.h, self.w, p, p, self.out_channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_channels, self.h * p, self.w * p)
