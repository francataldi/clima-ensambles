from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import ForecastDataset
from .data.store import Era5Store
from .models.afno import AFNOConfig
from .models.forecaster import Forecaster, load_forecaster, save_forecaster


@dataclass
class Stage:
    rollout: int
    epochs: int
    lr: float
    loss: str = "mse"
    members: int = 2


@dataclass
class TrainConfig:
    data_dir: str = "data/era5_64x32"
    train_range: tuple[str, str] = ("1990-01-01", "2017-12-31T18")
    val_range: tuple[str, str] = ("2018-01-01", "2019-12-31T18")
    model: dict = field(default_factory=lambda: AFNOConfig().to_dict())
    stages: list[dict] = field(
        default_factory=lambda: [
            {"rollout": 1, "epochs": 40, "lr": 5e-4},
            {"rollout": 2, "epochs": 6, "lr": 1e-4},
            {"rollout": 4, "epochs": 4, "lr": 5e-5},
        ]
    )
    batch_size: int = 16
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    num_workers: int = 4
    steps_per_epoch: int | None = None
    val_batches: int = 40
    val_rollout: int = 4
    amp: bool = True
    device: str = "auto"
    out_dir: str = "runs/afno"
    seed: int = 0
    resume: bool = True
    headline: tuple[str, ...] = ("z500", "t850", "t2m", "u10")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        base = cls()
        model = {**base.model, **data.pop("model", {})}
        for key in ("train_range", "val_range", "headline"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(model=model, **data)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loader(ds: ForecastDataset, cfg: TrainConfig, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
        persistent_workers=cfg.num_workers > 0,
    )


@torch.no_grad()
def validate(model: Forecaster, store: Era5Store, cfg: TrainConfig, device: torch.device) -> dict:
    ds = ForecastDataset(store, *cfg.val_range, rollout=cfg.val_rollout, stride=7)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    model.eval()
    heads = [store.index(h) for h in cfg.headline if h in store.names]
    w = model.lat_weights
    loss_sum = 0.0
    crps_sum = 0.0
    se = torch.zeros(cfg.val_rollout, len(heads), device=device)
    n = 0
    for b, (states, forcings, _) in enumerate(loader):
        if b >= cfg.val_batches:
            break
        states = states.to(device)
        forcings = forcings.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
            traj = model.rollout(states[:, 0], forcings)
        traj = traj.float()
        loss_sum += float(model.loss(traj[:, 1], states[:, 1])) * states.shape[0]
        if model.stochastic:
            members = 4
            x0 = states[:, 0].repeat(members, 1, 1, 1)
            f = forcings[:, :1].repeat(members, 1, 1, 1, 1)
            sampled = model.rollout(x0, f, sample=True)[:, 1].float()
            sampled = sampled.view(members, states.shape[0], *sampled.shape[1:])
            crps_sum += float(model.crps_loss(sampled, states[:, 1])) * states.shape[0]
        pred = model.denormalize(traj[:, 1:].reshape(-1, *traj.shape[2:])).view(*traj[:, 1:].shape)
        true = model.denormalize(states[:, 1:].reshape(-1, *states.shape[2:])).view(*states[:, 1:].shape)
        err = ((pred - true) ** 2 * w.unsqueeze(1)).mean(dim=(-2, -1))
        se += err[:, :, heads].sum(0)
        n += states.shape[0]
    rmse = torch.sqrt(se / max(n, 1)).cpu().numpy()
    result = {"val_loss": loss_sum / max(n, 1)}
    if model.stochastic:
        result["val_crps"] = crps_sum / max(n, 1)
    for k in range(cfg.val_rollout):
        for j, h in enumerate([h for h in cfg.headline if h in store.names]):
            result[f"rmse_{h}_{6 * (k + 1)}h"] = float(rmse[k, j])
    return result


def _cosine(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))


def train(cfg: TrainConfig) -> Path:
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    store = Era5Store(cfg.data_dir)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    model = Forecaster.from_store(store, AFNOConfig(**cfg.model)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.stages[0]["lr"], weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    state_path = out / "state.pt"
    position = {"stage": 0, "epoch": 0, "step": 0}
    best = float("inf")
    if cfg.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        position = state["position"]
        best = state["best"]
        print(f"retomando: etapa {position['stage']} epoca {position['epoch']}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"modelo con {n_params / 1e6:.1f} M parametros en {device}")
    metrics_path = out / "metrics.csv"
    metrics_file = open(metrics_path, "a", newline="", encoding="utf-8")
    writer = None
    use_amp = cfg.amp and device.type == "cuda"
    try:
        for si in range(position["stage"], len(cfg.stages)):
            stage = Stage(**cfg.stages[si])
            train_ds = ForecastDataset(store, *cfg.train_range, rollout=stage.rollout)
            loader = _loader(train_ds, cfg, True, device)
            per_epoch = min(len(loader), cfg.steps_per_epoch or len(loader))
            total = per_epoch * stage.epochs
            first_epoch = position["epoch"] if si == position["stage"] else 0
            step = position["step"] if si == position["stage"] else 0
            for epoch in range(first_epoch, stage.epochs):
                model.train()
                t0 = time.time()
                running = 0.0
                for b, (states, forcings, _) in enumerate(loader):
                    if b >= per_epoch:
                        break
                    lr = _cosine(step, total, cfg.warmup_steps if si == 0 else 50, stage.lr)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    states = states.to(device, non_blocking=True)
                    forcings = forcings.to(device, non_blocking=True)
                    loss = torch.zeros((), device=device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        if stage.loss == "crps":
                            B = states.shape[0]
                            M = stage.members
                            x = states[:, 0].repeat(M, 1, 1, 1)
                            for k in range(stage.rollout):
                                x = model.step(x, forcings[:, k].repeat(M, 1, 1, 1), model.sample_noise(M * B))
                                members = x.float().view(M, B, *x.shape[1:])
                                loss = loss + model.crps_loss(members, states[:, k + 1])
                        else:
                            x = states[:, 0]
                            for k in range(stage.rollout):
                                x = model.step(x, forcings[:, k])
                                loss = loss + model.loss(x.float(), states[:, k + 1])
                    loss = loss / stage.rollout
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    running += float(loss.detach())
                    step += 1
                train_loss = running / max(per_epoch, 1)
                val = validate(model, store, cfg, device)
                row = {
                    "stage": si,
                    "rollout": stage.rollout,
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 6),
                    "seconds": round(time.time() - t0, 1),
                    **{k: round(v, 5) for k, v in val.items()},
                }
                if writer is None:
                    writer = csv.DictWriter(metrics_file, fieldnames=list(row.keys()), extrasaction="ignore")
                    if metrics_path.stat().st_size == 0:
                        writer.writeheader()
                writer.writerow(row)
                metrics_file.flush()
                head = " ".join(f"{k}={v:.3f}" for k, v in val.items() if k.endswith("24h"))
                print(f"etapa {si} (rollout {stage.rollout}) epoca {epoch + 1}/{stage.epochs} | train {train_loss:.4f} | val {val['val_loss']:.4f} | {head} | {row['seconds']}s", flush=True)
                score = val["val_crps"] if stage.loss == "crps" and "val_crps" in val else val["val_loss"]
                save_forecaster(out / "last.pt", model, {"stage": si, "epoch": epoch + 1, **val})
                if score < best:
                    best = score
                    save_forecaster(out / "best.pt", model, {"stage": si, "epoch": epoch + 1, **val})
                position = {"stage": si, "epoch": epoch + 1, "step": step}
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "position": position, "best": best}, state_path)
            position = {"stage": si + 1, "epoch": 0, "step": 0}
            best = float("inf")
            save_forecaster(out / f"stage{si}_rollout{stage.rollout}.pt", model, {"stage": si})
    finally:
        metrics_file.close()
    return out / "last.pt"


def train_members(cfg: TrainConfig, seeds: list[int]) -> list[Path]:
    paths = []
    base = cfg.out_dir
    for seed in seeds:
        cfg.seed = seed
        cfg.out_dir = f"{base}_seed{seed}"
        paths.append(train(cfg))
    cfg.out_dir = base
    return paths


def load_members(paths: list[str], device: torch.device) -> list[Forecaster]:
    return [load_forecaster(p, device)[0] for p in paths]
