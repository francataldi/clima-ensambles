from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import ForecastDataset
from .data.store import Era5Store
from .models.afno import AFNOConfig
from .models.forecaster import Forecaster, load_forecaster, save_forecaster

LOG_NAME = "train_log.csv"
LEGACY_LOG = "metrics.csv"
LOG_FIRST = ("time", "stage", "rollout", "loss", "epoch", "epochs", "step", "lr", "train_loss", "grad_norm", "val_loss", "val_crps", "val_spread_skill", "score", "best")
LOG_LAST = ("seconds", "val_seconds", "samples_per_s", "gpu_mem_gb")


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
    until_stage: int | None = None
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

    def last_stage(self) -> int:
        if self.until_stage is None:
            return len(self.stages) - 1
        if not 0 <= self.until_stage < len(self.stages):
            raise ValueError(f"until_stage={self.until_stage} fuera de rango: hay {len(self.stages)} etapas (0 a {len(self.stages) - 1})")
        return self.until_stage


def _log_key(row: dict) -> tuple[int, int]:
    try:
        return int(float(row.get("stage"))), int(float(row.get("epoch")))
    except (TypeError, ValueError):
        return -1, -1


class TrainLog:
    def __init__(self, out_dir: str | Path, resume: bool = True) -> None:
        out = Path(out_dir)
        self.path = out / LOG_NAME
        rows = []
        if resume:
            rows = self.read(self.path) or self.read(out / LEGACY_LOG)
        unique = {_log_key(row): row for row in rows}
        self.rows = [unique[k] for k in sorted(unique)]

    @staticmethod
    def read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [{k: v for k, v in row.items() if k is not None} for row in csv.DictReader(f)]

    def truncate(self, stage: int, epoch: int) -> None:
        self.rows = [row for row in self.rows if _log_key(row) <= (stage, epoch)]

    def columns(self) -> list[str]:
        seen = list(dict.fromkeys(k for row in self.rows for k in row))
        first = [k for k in LOG_FIRST if k in seen]
        last = [k for k in LOG_LAST if k in seen]
        return first + [k for k in seen if k not in first and k not in last] + last

    def add(self, row: dict) -> bool:
        key = _log_key(row)
        self.rows = sorted([r for r in self.rows if _log_key(r) != key] + [row], key=_log_key)
        return self.save()

    def save(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns())
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(tmp, self.path)
        except OSError as exc:
            print(f"aviso: no se pudo actualizar {self.path} ({exc}); se reintenta en la proxima epoca", flush=True)
            return False
        return True


def _round(value):
    if isinstance(value, float):
        return float(f"{value:.6g}")
    return value


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


def _anomaly_correlation(pred: torch.Tensor, true: torch.Tensor, clim: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    fa = pred - clim
    oa = true - clim
    fa = fa - (fa * w).mean(dim=(-2, -1), keepdim=True)
    oa = oa - (oa * w).mean(dim=(-2, -1), keepdim=True)
    num = (fa * oa * w).mean(dim=(-2, -1))
    den = torch.sqrt((fa**2 * w).mean(dim=(-2, -1)) * (oa**2 * w).mean(dim=(-2, -1)))
    return num / den.clamp_min(1e-12)


@torch.no_grad()
def validate(model: Forecaster, store: Era5Store, cfg: TrainConfig, device: torch.device) -> dict:
    ds = ForecastDataset(store, *cfg.val_range, rollout=cfg.val_rollout, stride=7)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    model.eval()
    names = [h for h in cfg.headline if h in store.names]
    heads = [store.index(h) for h in names]
    w = model.lat_weights
    steps = cfg.val_rollout
    use_acc = store.climatology is not None and bool(heads)
    members = 4
    loss_sum = 0.0
    crps_sum = 0.0
    var_sum = torch.zeros(len(store.names), device=device)
    err_sum = torch.zeros(len(store.names), device=device)
    se = torch.zeros(steps, len(heads), device=device)
    acc_sum = torch.zeros(steps, len(heads), device=device)
    n = 0
    for b, (states, forcings, t0) in enumerate(loader):
        if b >= cfg.val_batches:
            break
        states = states.to(device)
        forcings = forcings.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
            traj = model.rollout(states[:, 0], forcings)
        traj = traj.float()
        loss_sum += float(model.loss(traj[:, 1], states[:, 1])) * states.shape[0]
        if model.stochastic:
            x0 = states[:, 0].repeat(members, 1, 1, 1)
            f = forcings[:, :1].repeat(members, 1, 1, 1, 1)
            sampled = model.rollout(x0, f, sample=True)[:, 1].float()
            sampled = sampled.view(members, states.shape[0], *sampled.shape[1:])
            crps_sum += float(model.crps_loss(sampled, states[:, 1])) * states.shape[0]
            var_sum += (sampled.var(dim=0) * w).mean(dim=(-2, -1)).sum(0)
            err_sum += ((sampled.mean(dim=0) - states[:, 1]) ** 2 * w).mean(dim=(-2, -1)).sum(0)
        pred = model.denormalize(traj[:, 1:].reshape(-1, *traj.shape[2:])).view(*traj[:, 1:].shape)
        true = model.denormalize(states[:, 1:].reshape(-1, *states.shape[2:])).view(*states[:, 1:].shape)
        err = ((pred - true) ** 2 * w.unsqueeze(1)).mean(dim=(-2, -1))
        se += err[:, :, heads].sum(0)
        if use_acc:
            clim = np.stack([[np.asarray(store.climatology_at(int(t) + k + 1))[heads] for k in range(steps)] for t in t0])
            clim = torch.from_numpy(clim).to(device)
            acc_sum += _anomaly_correlation(pred[:, :, heads], true[:, :, heads], clim, w).sum(0)
        n += states.shape[0]
    rmse = torch.sqrt(se / max(n, 1)).cpu().numpy()
    result = {"val_loss": loss_sum / max(n, 1)}
    if model.stochastic:
        result["val_crps"] = crps_sum / max(n, 1)
        ratio = torch.sqrt(var_sum * (members + 1) / members / err_sum.clamp_min(1e-12))
        result["val_spread_skill"] = float(ratio.mean())
    for k in range(steps):
        for j, h in enumerate(names):
            result[f"rmse_{h}_{6 * (k + 1)}h"] = float(rmse[k, j])
    if use_acc:
        acc = (acc_sum / max(n, 1)).cpu().numpy()
        for k in range(steps):
            for j, h in enumerate(names):
                result[f"acc_{h}_{6 * (k + 1)}h"] = float(acc[k, j])
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
    last_stage = cfg.last_stage()
    store = Era5Store(cfg.data_dir)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    model = Forecaster.from_store(store, AFNOConfig(**cfg.model)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.stages[0]["lr"], weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    state_path = out / "state.pt"
    position = {"stage": 0, "epoch": 0, "step": 0}
    best = float("inf")
    resumed = cfg.resume and state_path.exists()
    if resumed:
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        position = state["position"]
        best = state["best"]
        print(f"retomando: etapa {position['stage']} epoca {position['epoch']}")
    log = TrainLog(out, resume=resumed)
    log.truncate(position["stage"], position["epoch"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"modelo con {n_params / 1e6:.1f} M parametros en {device}")
    if position["stage"] > last_stage:
        print(f"nada para entrenar: el estado guardado ya paso la etapa {last_stage}")
        return out / "last.pt"
    use_amp = cfg.amp and device.type == "cuda"
    for si in range(position["stage"], last_stage + 1):
        stage = Stage(**cfg.stages[si])
        train_ds = ForecastDataset(store, *cfg.train_range, rollout=stage.rollout)
        loader = _loader(train_ds, cfg, True, device)
        per_epoch = min(len(loader), cfg.steps_per_epoch or len(loader))
        total = per_epoch * stage.epochs
        first_epoch = position["epoch"] if si == position["stage"] else 0
        step = position["step"] if si == position["stage"] else 0
        for epoch in range(first_epoch, stage.epochs):
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            running = torch.zeros((), device=device)
            grad_norm = torch.zeros((), device=device)
            lr = stage.lr
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
                grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).detach()
                opt.step()
                running += loss.detach()
                step += 1
            train_loss = float(running) / max(per_epoch, 1)
            mean_grad_norm = float(grad_norm) / max(per_epoch, 1)
            seconds = time.time() - t0
            t1 = time.time()
            val = validate(model, store, cfg, device)
            val_seconds = time.time() - t1
            score = val["val_crps"] if stage.loss == "crps" and "val_crps" in val else val["val_loss"]
            improved = score < best
            row = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "stage": si,
                "rollout": stage.rollout,
                "loss": stage.loss,
                "epoch": epoch + 1,
                "epochs": stage.epochs,
                "step": step,
                "lr": lr,
                "train_loss": train_loss,
                "grad_norm": mean_grad_norm,
                **val,
                "score": score,
                "best": int(improved),
                "seconds": seconds,
                "val_seconds": val_seconds,
                "samples_per_s": per_epoch * cfg.batch_size / max(seconds, 1e-9),
            }
            if device.type == "cuda":
                row["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            log.add({k: _round(v) for k, v in row.items()})
            head = " ".join(f"{k}={v:.3f}" for k, v in val.items() if k.endswith("24h"))
            print(f"etapa {si} (rollout {stage.rollout}) epoca {epoch + 1}/{stage.epochs} | train {train_loss:.4f} | val {val['val_loss']:.4f} | {head} | {seconds:.1f}s", flush=True)
            save_forecaster(out / "last.pt", model, {"stage": si, "epoch": epoch + 1, **val})
            if improved:
                best = score
                save_forecaster(out / "best.pt", model, {"stage": si, "epoch": epoch + 1, **val})
            position = {"stage": si, "epoch": epoch + 1, "step": step}
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "position": position, "best": best}, state_path)
        position = {"stage": si + 1, "epoch": 0, "step": 0}
        best = float("inf")
        save_forecaster(out / f"stage{si}_rollout{stage.rollout}.pt", model, {"stage": si})
    if last_stage < len(cfg.stages) - 1:
        print(f"detenido despues de la etapa {last_stage}: al volver a correr sin --until-stage se retoma en la etapa {last_stage + 1}")
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
