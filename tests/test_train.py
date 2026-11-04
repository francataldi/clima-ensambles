import csv

import pytest
import torch

from clima.models.afno import AFNOConfig
from clima.train import LOG_NAME, TrainConfig, TrainLog, train

TINY = AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2, dropout=0.1, noise_channels=2)


def make_cfg(store, out, **overrides):
    base = dict(
        data_dir=str(store.root),
        train_range=("2018-01-01", "2018-02-28T18"),
        val_range=("2018-03-01", "2018-03-31T18"),
        model=TINY.to_dict(),
        stages=[{"rollout": 1, "epochs": 1, "lr": 1e-3}, {"rollout": 2, "epochs": 1, "lr": 1e-4, "loss": "crps", "members": 2}],
        batch_size=4,
        num_workers=0,
        steps_per_epoch=2,
        val_batches=1,
        warmup_steps=2,
        amp=False,
        device="cpu",
        out_dir=str(out),
    )
    base.update(overrides)
    return TrainConfig(**base)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_until_stage_stops_and_resumes(store, tmp_path):
    out = tmp_path / "run"
    train(make_cfg(store, out, until_stage=0))
    state = torch.load(out / "state.pt", weights_only=False)
    assert state["position"] == {"stage": 0, "epoch": 1, "step": 2}
    assert (out / "stage0_rollout1.pt").exists()
    assert not (out / "stage1_rollout2.pt").exists()
    assert [(r["stage"], r["epoch"]) for r in read_rows(out / LOG_NAME)] == [("0", "1")]
    train(make_cfg(store, out, until_stage=0))
    assert len(read_rows(out / LOG_NAME)) == 1
    train(make_cfg(store, out))
    rows = read_rows(out / LOG_NAME)
    assert [(r["stage"], r["epoch"], r["loss"]) for r in rows] == [("0", "1", "mse"), ("1", "1", "crps")]
    expected = {"time", "lr", "train_loss", "grad_norm", "val_loss", "val_crps", "val_spread_skill", "score", "best", "seconds", "val_seconds", "samples_per_s", "rmse_z500_24h", "acc_t2m_6h"}
    assert expected <= set(rows[0])
    assert all(-1.0 <= float(rows[1][f"acc_{v}_24h"]) <= 1.0 for v in ("z500", "t850", "t2m", "u10"))
    assert float(rows[1]["val_spread_skill"]) > 0
    assert torch.load(out / "state.pt", weights_only=False)["position"]["stage"] == 1
    assert (out / "stage1_rollout2.pt").exists()


def test_resume_imports_legacy_metrics(store, tmp_path):
    out = tmp_path / "legacy"
    train(make_cfg(store, out, until_stage=0))
    (out / LOG_NAME).unlink()
    (out / "metrics.csv").write_text("stage,rollout,epoch,train_loss,seconds,val_loss\n0,1,1,9.5,1.0,8.5\n0,1,1,7.5,1.0,6.5\n", encoding="utf-8")
    train(make_cfg(store, out))
    rows = read_rows(out / LOG_NAME)
    assert [(r["stage"], r["epoch"]) for r in rows] == [("0", "1"), ("1", "1")]
    assert rows[0]["train_loss"] == "7.5"
    assert rows[0]["time"] == ""
    assert rows[1]["time"] != ""


def test_fresh_run_ignores_previous_log(store, tmp_path):
    out = tmp_path / "fresh"
    train(make_cfg(store, out))
    assert len(read_rows(out / LOG_NAME)) == 2
    train(make_cfg(store, out, resume=False, until_stage=0))
    assert [(r["stage"], r["epoch"]) for r in read_rows(out / LOG_NAME)] == [("0", "1")]


def test_until_stage_out_of_range(store, tmp_path):
    with pytest.raises(ValueError, match="fuera de rango"):
        train(make_cfg(store, tmp_path / "x", until_stage=2))
    with pytest.raises(ValueError):
        make_cfg(store, tmp_path / "x", until_stage=-1).last_stage()
    assert not (tmp_path / "x").exists()


def test_train_log_upsert_columns_and_write_errors(tmp_path):
    log = TrainLog(tmp_path)
    assert log.add({"stage": 0, "epoch": 2, "train_loss": 1.0, "seconds": 3.0})
    assert log.add({"stage": 0, "epoch": 1, "train_loss": 2.0, "extra": 5})
    assert log.add({"stage": 0, "epoch": 2, "train_loss": 0.5, "seconds": 2.0})
    rows = read_rows(tmp_path / LOG_NAME)
    assert [(r["epoch"], r["train_loss"]) for r in rows] == [("1", "2.0"), ("2", "0.5")]
    assert list(rows[0]) == ["stage", "epoch", "train_loss", "extra", "seconds"]
    resumed = TrainLog(tmp_path)
    resumed.truncate(0, 1)
    assert [r["epoch"] for r in resumed.rows] == ["1"]
    assert TrainLog(tmp_path, resume=False).rows == []
    blocked = TrainLog(tmp_path / "bloqueado")
    blocked.path.mkdir(parents=True)
    assert not blocked.add({"stage": 0, "epoch": 1})
    assert len(blocked.rows) == 1
