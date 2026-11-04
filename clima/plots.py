from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.animation import FuncAnimation, PillowWriter

from .metrics import reliability_from_histogram, roc_from_histogram

UNITS = {"z500": ("m²/s²", 1.0), "t850": ("K", 1.0), "t2m": ("K", 1.0), "u10": ("m/s", 1.0), "msl": ("Pa", 1.0)}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
EVENT_REGIONS = ("tropicos", "extratropicos_norte", "extratropicos_sur", "sudamerica")


@contextmanager
def theme():
    style = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "font.size": 9,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
    }
    with plt.rc_context(style):
        yield


def figure(rows: int, cols: int, panel: tuple[float, float], header: float = 0.8):
    width = cols * panel[0]
    height = rows * panel[1] + header
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - header / height), h_pad=0.08, w_pad=0.08)
    return fig, axes


def header(fig, title: str, subtitle: str | None = None, handles: list | None = None, labels: list[str] | None = None) -> None:
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.16 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 1 - 0.44 / h, subtitle, ha="left", va="top", fontsize=9, color=INK_2)
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.006, 1 - 0.62 / h), ncol=len(handles), borderaxespad=0.0)


def _leads_to_show(lead_hours: list[int], wanted: tuple[int, ...] = (24, 72, 120)) -> list[int]:
    chosen = [h for h in wanted if h in lead_hours]
    if len(chosen) >= 2:
        return chosen
    positive = [h for h in lead_hours if h > 0]
    if len(positive) <= 3:
        return positive
    return [positive[i] for i in np.linspace(0, len(positive) - 1, 3).round().astype(int)]


def _lead_label(hours: int) -> str:
    return f"{hours // 24} d" if hours % 24 == 0 else f"{hours} h"


def _event_regions(event: dict) -> list[str]:
    names = [r for r in EVENT_REGIONS if r in event["regions"]]
    return names or [r for r in event["regions"] if r != "global"] or list(event["regions"])


def _array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def plot_skill(result: dict, out_dir: str | Path, variables: tuple[str, ...] = ("z500", "t850", "t2m", "u10")) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = result["channels"]
    days = np.asarray(result["lead_hours"]) / 24.0
    variables = tuple(v for v in variables if v in names) or tuple(names[:4])
    with theme():
        fig, axes = figure(2, len(variables), (3.7, 2.9), header=1.0)
        for j, var in enumerate(variables):
            c = names.index(var)
            ax = axes[0, j]
            for key, label, color, style in (
                ("persist_rmse", "persistencia", MUTED, "--"),
                ("clim_rmse", "climatología", AXIS, ":"),
                ("det_rmse", "control", SERIES[0], "-"),
                ("mean_rmse", "media del ensamble", SERIES[1], "-"),
            ):
                ax.plot(days[1:], _array(result[key])[1:, c], style, color=color, lw=1.8 if style == "-" else 1.4, label=label)
            ax.set_title(f"{var} ({channel_units(var)})")
            if j == 0:
                ax.set_ylabel("RMSE")
            ax2 = axes[1, j]
            ax2.plot(days[1:], _array(result["mean_rmse"])[1:, c], color=SERIES[1], label="RMSE de la media")
            ax2.plot(days[1:], _array(result["spread"])[1:, c], color=SERIES[2], label="dispersión")
            ax2.plot(days[1:], _array(result["crps"])[1:, c], color=SERIES[6], label="CRPS")
            ax2.set_xlabel("plazo (días)")
            if j == 0:
                ax2.set_ylabel("ensamble")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        more, more_labels = axes[1, 0].get_legend_handles_labels()
        subtitle = f"{result['inits']} inicializaciones · {result['ensemble']['members']} miembros · arriba: error del control y de la media · abajo: dispersión vs error"
        header(fig, "Skill por plazo", subtitle, handles + more[1:], labels + more_labels[1:])
        path = out / "skill.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_rank_histograms(result: dict, out_dir: str | Path, variables: tuple[str, ...] = ("z500", "t850", "t2m", "u10")) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variables = tuple(v for v in variables if v in result["rank_histogram_last_lead"]) or tuple(result["rank_histogram_last_lead"])[:4]
    lead = result["lead_hours"][-1]
    with theme():
        fig, axes = figure(1, len(variables), (3.4, 2.5), header=0.8)
        top = max(1.5, max(float(np.max(_array(result["rank_histogram_last_lead"][v]))) * len(result["rank_histogram_last_lead"][v]) for v in variables) * 1.1)
        for j, var in enumerate(variables):
            ax = axes[0, j]
            h = _array(result["rank_histogram_last_lead"][var])
            ax.bar(np.arange(len(h)), h * len(h), width=1.0, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
            ax.axhline(1.0, color=INK_2, lw=0.9)
            ax.set_ylim(0, top)
            ax.set_xlim(-0.5, len(h) - 0.5)
            ax.set_xticks([0, len(h) - 1], ["1", str(len(h))])
            ax.grid(axis="x", visible=False)
            ax.set_title(var)
            ax.set_xlabel("rango de la verdad")
            ax.text(0.98, 0.92, f"extremos ×{(h[0] + h[-1]) * len(h) / 2:.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_2)
            if j == 0:
                ax.set_ylabel("frecuencia / uniforme")
        header(fig, f"Histogramas de rangos a {_lead_label(lead)}", "1 = plano (ensamble confiable) · forma de U: poca dispersión · forma de domo: demasiada")
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


def _ramp(n: int) -> list[str]:
    if n <= 3:
        return list(RAMP[1 : 1 + n])
    return [RAMP[i] for i in np.linspace(0, len(RAMP) - 1, n).round().astype(int)]


def plot_event_scores(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = result["events"]
    labels = {k: v["label"] for k, v in result.get("regions", {}).items()}
    days = np.asarray(result["lead_hours"]) / 24.0
    with theme():
        fig, axes = figure(2, len(events), (5.4, 3.4), header=1.0)
        for j, event in enumerate(events.values()):
            regions = _event_regions(event)
            for row, key, ref, ylabel in ((0, "bss", 0.0, "BSS contra climatología"), (1, "auc", 0.5, "área bajo la curva ROC")):
                ax = axes[row, j]
                ax.axhline(ref, color=AXIS, lw=1.0, zorder=1)
                if "global" in event["regions"]:
                    ax.plot(days[1:], _array(event["regions"]["global"][key])[1:], color=MUTED, lw=1.2, label="Global")
                for i, region in enumerate(regions):
                    ax.plot(days[1:], _array(event["regions"][region][key])[1:], color=SERIES[i], marker="o", ms=3.5, label=labels.get(region, region))
                ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(event["label"])
                else:
                    ax.set_xlabel("plazo (días)")
                    low = np.nanmin(_array([event["regions"][r]["auc"][1:] for r in regions]))
                    ax.set_ylim(min(0.45, low - 0.02) if np.isfinite(low) else 0.45, 1.0)
        handles, names = axes[0, 0].get_legend_handles_labels()
        subtitle = f"{result['inits']} inicializaciones · {result['ensemble']['members']} miembros · BSS = 1 − Brier / Brier de la climatología"
        header(fig, "Eventos extremos: skill probabilístico por región", subtitle, handles, names)
        path = out / "eventos_skill.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def _probability_axes(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])


def plot_reliability(result: dict, out_dir: str | Path, leads: list[int] | None = None, groups: int = 11) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = result["events"]
    labels = {k: v["label"] for k, v in result.get("regions", {}).items()}
    leads = leads or _leads_to_show(result["lead_hours"])
    colors = _ramp(len(leads))
    regions = _event_regions(next(iter(events.values())))
    with theme():
        fig, axes = figure(len(events), len(regions), (3.2, 3.3), header=1.0)
        for i, event in enumerate(events.values()):
            clim = 1.0 - event["quantile"]
            for j, region in enumerate(regions):
                ax = axes[i, j]
                hist = _array(event["regions"][region]["hist"])
                ax.fill_between([clim, 1], [clim, (1 + clim) / 2], [1, 1], color=SERIES[0], alpha=0.07, lw=0)
                ax.fill_between([0, clim], [0, 0], [clim / 2, clim], color=SERIES[0], alpha=0.07, lw=0)
                ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0)
                ax.plot([0, 1], [clim / 2, (1 + clim) / 2], color=GRID, lw=1.0)
                ax.axhline(clim, color=GRID, lw=1.0)
                ax.axvline(clim, color=GRID, lw=1.0)
                inset = ax.inset_axes((0.06, 0.62, 0.34, 0.3))
                for k, lead in enumerate(leads):
                    li = result["lead_hours"].index(lead)
                    prob, freq, weight = reliability_from_histogram(hist[li], groups)
                    ok = weight > 0
                    ax.plot(prob[ok], freq[ok], color=colors[k], marker="o", ms=3.5, label=_lead_label(lead))
                    share = weight / max(weight.sum(), 1e-12)
                    inset.step(np.linspace(0, 1, len(share)), np.maximum(share, 1e-4), where="mid", color=colors[k], lw=1.0)
                inset.set_yscale("log")
                inset.set_ylim(1e-4, 2)
                inset.set_xlim(-0.05, 1.05)
                inset.set_xticks([])
                inset.set_yticks([])
                inset.minorticks_off()
                inset.grid(False)
                inset.set_facecolor(SURFACE)
                for side in ("top", "right"):
                    inset.spines[side].set_visible(True)
                inset.set_title("nitidez (log)", fontsize=7, color=MUTED, pad=2, loc="center", fontweight="normal")
                _probability_axes(ax)
                ax.set_title(labels.get(region, region), fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{event['label']}\nfrecuencia observada")
                if i == len(events) - 1:
                    ax.set_xlabel("probabilidad pronosticada")
        handles, names = axes[0, 0].get_legend_handles_labels()
        header(fig, "Diagramas de confiabilidad", "diagonal: confiabilidad perfecta · sombreado: aporta skill frente a la climatología (p = 1 − cuantil)", handles, [f"plazo {n}" for n in names])
        path = out / "confiabilidad.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_roc(result: dict, out_dir: str | Path, leads: list[int] | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = result["events"]
    labels = {k: v["label"] for k, v in result.get("regions", {}).items()}
    leads = leads or _leads_to_show(result["lead_hours"])
    colors = _ramp(len(leads))
    regions = _event_regions(next(iter(events.values())))
    with theme():
        fig, axes = figure(len(events), len(regions), (3.2, 3.3), header=0.6)
        for i, event in enumerate(events.values()):
            for j, region in enumerate(regions):
                ax = axes[i, j]
                hist = _array(event["regions"][region]["hist"])
                auc = _array(event["regions"][region]["auc"])
                ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0)
                for k, lead in enumerate(leads):
                    li = result["lead_hours"].index(lead)
                    pofd, pod = roc_from_histogram(hist[li])
                    value = "s/d" if not np.isfinite(auc[li]) else f"{auc[li]:.2f}"
                    ax.plot(pofd, pod, color=colors[k], marker="o", ms=2.5, label=f"{_lead_label(lead)} · AUC {value}")
                _probability_axes(ax)
                ax.set_title(labels.get(region, region), fontsize=9)
                ax.legend(loc="lower right", fontsize=7, handlelength=1.2)
                if j == 0:
                    ax.set_ylabel(f"{event['label']}\ntasa de aciertos")
                if i == len(events) - 1:
                    ax.set_xlabel("tasa de falsas alarmas")
        header(fig, "Curvas ROC", "cada punto es un umbral sobre la fracción de miembros que pronostica el evento")
        path = out / "roc.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


STRATEGY_ORDER = ("ci", "ruido", "ci+ruido", "mc_dropout")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
CHANNEL_UNITS = {"t": "K", "u": "m/s", "v": "m/s", "z": "m²/s²", "q": "kg/kg", "msl": "Pa", "ws": "m/s"}


def channel_units(name: str) -> str:
    if name in CHANNEL_UNITS:
        return CHANNEL_UNITS[name]
    for prefix in ("ws", "t", "u", "v", "z", "q"):
        if name.startswith(prefix):
            return CHANNEL_UNITS[prefix]
    return ""


def strategy_styles(names) -> dict[str, tuple[str, str]]:
    order = list(STRATEGY_ORDER) + [n for n in names if n not in STRATEGY_ORDER]
    return {n: (SERIES[order.index(n) % len(SERIES)], MARKERS[order.index(n) % len(MARKERS)]) for n in names}


def plot_strategy_scores(comparison: dict, out_dir: str | Path, variables: tuple[str, ...] = ("z500", "t850", "t2m", "u10")) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = comparison["strategies"]
    labels = comparison.get("labels", {})
    first = next(iter(results.values()))
    variables = [v for v in variables if v in first["channels"]] or first["channels"][:4]
    days = np.asarray(first["lead_hours"]) / 24.0
    styles = strategy_styles(list(results))
    reference = _array(results["ci"]["crps"]) if "ci" in results and len(results) > 1 else None
    rows = [("crps", "CRPS")] + ([("crpss", "CRPSS frente a solo CI")] if reference is not None else []) + [("spread_skill", "dispersión / RMSE de la media")]
    with theme():
        fig, axes = figure(len(rows), len(variables), (3.7, 2.8), header=1.0)
        for j, var in enumerate(variables):
            for name, r in results.items():
                c = r["channels"].index(var)
                color, marker = styles[name]
                crps = _array(r["crps"])[:, c]
                series = {"crps": crps, "spread_skill": _array(r["spread_skill"])[:, c]}
                if reference is not None:
                    series["crpss"] = 1.0 - crps / np.maximum(reference[:, c], 1e-12)
                for i, (key, _) in enumerate(rows):
                    axes[i, j].plot(days[1:], series[key][1:], color=color, marker=marker, ms=3.5, label=labels.get(name, name))
            for i, (key, ylabel) in enumerate(rows):
                ax = axes[i, j]
                if key == "spread_skill":
                    ax.axhline(1.0, color=AXIS, lw=1.0, zorder=1)
                    ax.set_xlabel("plazo (días)")
                if key == "crpss":
                    ax.axhline(0.0, color=AXIS, lw=1.0, zorder=1)
                if j == 0:
                    ax.set_ylabel(ylabel)
            axes[0, j].set_title(f"{var} ({channel_units(var)})")
        handles, names = axes[0, 0].get_legend_handles_labels()
        members = first["ensemble"]["members"]
        header(fig, "Estrategias de ensamble", f"{first['inits']} inicializaciones · {members} miembros por estrategia · mismas fechas y semilla", handles, names)
        path = out / "estrategias.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_strategy_rank_histograms(comparison: dict, out_dir: str | Path, variable: str = "z500", leads: list[int] | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = comparison["strategies"]
    labels = comparison.get("labels", {})
    first = next(iter(results.values()))
    leads = leads or _leads_to_show(first["lead_hours"])
    styles = strategy_styles(list(results))
    with theme():
        fig, axes = figure(len(results), len(leads), (2.9, 1.75), header=0.75)
        for i, (name, r) in enumerate(results.items()):
            c = r["channels"].index(variable)
            ranks = _array(r["rank_histogram"])[:, c]
            top = max(1.5, float(np.nanmax(ranks[[r["lead_hours"].index(h) for h in leads]] * ranks.shape[-1])) * 1.1)
            for j, lead in enumerate(leads):
                ax = axes[i, j]
                h = ranks[r["lead_hours"].index(lead)]
                density = h * len(h)
                ax.bar(np.arange(len(h)), density, width=1.0, color=styles[name][0], edgecolor=SURFACE, linewidth=0.6)
                ax.axhline(1.0, color=INK_2, lw=0.9)
                ax.set_ylim(0, top)
                ax.set_xlim(-0.5, len(h) - 0.5)
                ax.set_xticks([0, len(h) - 1], ["1", str(len(h))])
                ax.grid(axis="x", visible=False)
                ratio = (h[0] + h[-1]) * len(h) / 2
                ax.text(0.98, 0.92, f"extremos ×{ratio:.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK_2)
                if i == 0:
                    ax.set_title(f"plazo {_lead_label(lead)}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(labels.get(name, name), fontsize=8)
                if i == len(results) - 1:
                    ax.set_xlabel("rango de la verdad")
        header(fig, f"Histogramas de rangos de {variable}", "barras normalizadas: 1 = histograma plano · extremos ×k: verdad fuera del ensamble k veces más de lo esperado")
        path = out / f"rangos_{variable}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


DIVERGING = ("#8f1f1e", "#e34948", "#f6c9c4", "#f0efec", "#cde2fb", "#3987e5", "#104281")


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("rojo_azul", DIVERGING)


def _cell_text(key: str, value: float) -> str:
    if not np.isfinite(value):
        return "–"
    if key == "cambio_rmse_pct":
        text = f"{value:+.0f}" if abs(value) >= 9.5 else f"{value:+.1f}"
    elif key == "cambio_acc":
        text = f"{value:+.2f}"
    else:
        text = f"{value:.2f}"
    if float(text) == 0:
        return "0"
    return text.replace("-", "−")


def plot_scorecard(card: dict, out_dir: str | Path) -> Path:
    from matplotlib.colors import Normalize

    from .scorecard import PANELS

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variables, leads = card["variables"], card["leads"]
    panels = [p for p in PANELS if np.isfinite(card[p[0]]).any()]
    cols = 2 if len(panels) > 1 else 1
    rows = int(np.ceil(len(panels) / cols))
    cmap = diverging_cmap()
    with theme():
        fig, axes = figure(rows, cols, (0.58 * len(leads) + 1.3, 0.27 * len(variables) + 1.0), header=0.85)
        for ax in axes.ravel()[len(panels) :]:
            ax.set_visible(False)
        for ax, (key, title, description, invert) in zip(axes.ravel(), panels, strict=False):
            values = np.asarray(card[key], dtype=np.float64)
            shown = -values if invert else values
            finite = np.abs(shown[np.isfinite(shown)])
            vmax = float(np.clip(np.percentile(finite, 95) if finite.size else 1.0, 1e-3, 100.0 if key == "cambio_rmse_pct" else 1.0))
            norm = Normalize(-vmax, vmax)
            ax.pcolormesh(np.ma.masked_invalid(shown), cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=1.5)
            for i in range(len(variables)):
                for j in range(len(leads)):
                    v = values[i, j]
                    rgba = cmap(norm(shown[i, j])) if np.isfinite(v) else (1, 1, 1, 1)
                    light = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2] > 0.55
                    ax.text(j + 0.5, i + 0.5, _cell_text(key, v), ha="center", va="center", fontsize=7, color=INK if light else "#ffffff")
            ax.set_xticks(np.arange(len(leads)) + 0.5, [_lead_label(h) for h in leads])
            ax.set_yticks(np.arange(len(variables)) + 0.5, variables)
            ax.invert_yaxis()
            ax.tick_params(length=0)
            ax.grid(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_title(title, pad=16)
            ax.text(0.0, 1.012, description, transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom")
        subtitle = f"{card.get('inits')} inicializaciones · {card.get('members')} miembros · azul = mejor, rojo = peor"
        header(fig, "Scorecard", subtitle)
        path = out / "scorecard.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path
