from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

UNITS = {"z500": ("m²/s²", 1.0), "t850": ("K", 1.0), "t2m": ("K", 1.0), "u10": ("m/s", 1.0), "msl": ("Pa", 1.0)}


def plot_skill(result: dict, out_dir: str | Path, variables: tuple[str, ...] = ("z500", "t850", "t2m", "u10")) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = result["channels"]
    leads = np.asarray(result["lead_hours"]) / 24.0
    variables = tuple(v for v in variables if v in names)
    fig, axes = plt.subplots(2, len(variables), figsize=(4.2 * len(variables), 7), squeeze=False)
    for j, var in enumerate(variables):
        c = names.index(var)
        ax = axes[0, j]
        for key, label, style in (
            ("det_rmse", "modelo", "-"),
            ("mean_rmse", "media del ensamble", "-"),
            ("persist_rmse", "persistencia", "--"),
            ("clim_rmse", "climatología", ":"),
        ):
            ax.plot(leads[1:], np.asarray(result[key])[1:, c], style, label=label)
        ax.set_title(f"{var}: RMSE")
        ax.set_xlabel("días")
        ax.set_ylabel(UNITS.get(var, ("", 1))[0])
        ax.grid(alpha=0.3)
        ax2 = axes[1, j]
        ax2.plot(leads[1:], np.asarray(result["crps"])[1:, c], label="CRPS")
        ax2.plot(leads[1:], np.asarray(result["spread"])[1:, c], "--", label="dispersión")
        ax2.plot(leads[1:], np.asarray(result["mean_rmse"])[1:, c], ":", label="RMSE media")
        ax2.set_title(f"{var}: ensamble")
        ax2.set_xlabel("días")
        ax2.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    path = out / "skill.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rank_histograms(result: dict, out_dir: str | Path, variables: tuple[str, ...] = ("z500", "t850", "t2m", "u10")) -> Path:
    out = Path(out_dir)
    variables = tuple(v for v in variables if v in result["rank_histogram_last_lead"])
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 3), squeeze=False)
    for j, var in enumerate(variables):
        h = np.asarray(result["rank_histogram_last_lead"][var])
        axes[0, j].bar(np.arange(len(h)), h, width=1.0)
        axes[0, j].axhline(1.0 / len(h), color="k", ls="--", lw=1)
        axes[0, j].set_title(f"{var}: histograma de rangos a {result['lead_hours'][-1] // 24} días")
    fig.tight_layout()
    path = out / "rank_histograms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def animate_field(members: np.ndarray, lat: np.ndarray, lon: np.ndarray, times: list[str], title: str, path: str | Path, offset: float = 0.0) -> Path:
    mean = members.mean(axis=0) + offset
    spread = members.std(axis=0)
    vmin, vmax = np.percentile(mean, [2, 98])
    smax = np.percentile(spread, 99)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    extent = (lon[0], lon[-1], lat[0], lat[-1])
    im1 = a1.imshow(mean[0], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="RdYlBu_r", aspect="auto")
    im2 = a2.imshow(spread[0], origin="lower", extent=extent, vmin=0, vmax=smax, cmap="magma", aspect="auto")
    fig.colorbar(im1, ax=a1)
    fig.colorbar(im2, ax=a2)
    a1.set_title(f"{title}: media")
    a2.set_title(f"{title}: dispersión")
    label = fig.suptitle(times[0])

    def update(k):
        im1.set_data(mean[k])
        im2.set_data(spread[k])
        label.set_text(times[k])
        return im1, im2, label

    anim = FuncAnimation(fig, update, frames=mean.shape[0], interval=300)
    anim.save(str(path), writer=PillowWriter(fps=3))
    plt.close(fig)
    return Path(path)


def plot_point_plume(series: np.ndarray, times: list[str], threshold: float, title: str, path: str | Path, truth: np.ndarray | None = None) -> Path:
    leads = np.arange(series.shape[1])
    prob = (series > threshold).mean(axis=0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for m in range(series.shape[0]):
        a1.plot(leads, series[m], color="tab:blue", alpha=0.15, lw=1)
    a1.plot(leads, series.mean(axis=0), color="tab:blue", lw=2.5, label="media del ensamble")
    if truth is not None:
        a1.plot(leads, truth, color="k", lw=2, label="lo que pasó")
    a1.axhline(threshold, color="tab:red", ls="--", lw=1)
    a1.set_title(title)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.bar(leads, prob, color="tab:red")
    a2.set_ylim(0, 1)
    a2.set_ylabel(f"P > {threshold:g}")
    step = max(1, len(times) // 10)
    a2.set_xticks(leads[::step], [times[i] for i in range(0, len(times), step)], rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
