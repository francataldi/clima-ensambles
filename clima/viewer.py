from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path

import numpy as np

from .events import EVENTS, Event, derived_field, load_thresholds, sources

TEMPLATE = Path(__file__).with_name("viewer.html")
COASTLINES = Path(__file__).with_name("coastlines.json")
KELVIN = 273.15
MISSING = 65535


@dataclass(frozen=True)
class ViewVariable:
    name: str
    label: str
    units: str
    scale: float = 1.0
    offset: float = 0.0
    cmap: str = "viridis"
    center: float | None = None
    contour: float = 0.0
    contours: bool = False
    digits: int = 1

    def to_display(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.scale + self.offset


VARIABLES = {
    "t2m": ViewVariable("t2m", "Temperatura a 2 m", "°C", offset=-KELVIN, cmap="RdBu_r", center=0.0, contour=5.0),
    "ws10": ViewVariable("ws10", "Viento a 10 m", "m/s", cmap="viridis", contour=4.0),
    "msl": ViewVariable("msl", "Presión a nivel del mar", "hPa", scale=0.01, cmap="PuOr_r", center=1013.0, contour=4.0, contours=True),
    "z500": ViewVariable("z500", "Altura de 500 hPa", "dam", scale=1 / 98.0665, cmap="Spectral_r", contour=6.0, contours=True, digits=0),
    "t850": ViewVariable("t850", "Temperatura en 850 hPa", "°C", offset=-KELVIN, cmap="RdBu_r", center=0.0, contour=5.0),
    "ws250": ViewVariable("ws250", "Corriente en chorro (250 hPa)", "m/s", cmap="magma", contour=15.0, digits=0),
    "q850": ViewVariable("q850", "Humedad específica en 850 hPa", "g/kg", scale=1000.0, cmap="YlGnBu", contour=2.0),
}

DEFAULT_VARIABLES = ("t2m", "ws10", "msl", "z500", "t850", "ws250", "q850")
VIEW_LUTS = {"spread": "YlOrRd", "error": "RdBu_r", "prob": "PuRd"}


def available_variables(names: list[str], variables=DEFAULT_VARIABLES) -> list[str]:
    out = []
    for name in variables:
        if name not in VARIABLES:
            raise KeyError(f"variable de visor desconocida: {name}")
        try:
            sources(names, name)
        except KeyError:
            continue
        out.append(name)
    return out


def colormap_lut(name: str) -> str:
    import matplotlib

    rgb = (matplotlib.colormaps[name](np.linspace(0.0, 1.0, 256))[:, :3] * 255).round().astype(np.uint8)
    return base64.b64encode(rgb.tobytes()).decode("ascii")


def quantize_u16(x: np.ndarray, lo: float, step: float) -> np.ndarray:
    q = np.full(x.shape, MISSING, dtype=np.uint16)
    ok = np.isfinite(x)
    q[ok] = np.clip(np.round((x[ok] - lo) / step), 0, MISSING - 1).astype(np.uint16)
    return q


def dequantize_u16(q: np.ndarray, lo: float, step: float) -> np.ndarray:
    x = lo + q.astype(np.float64) * step
    x[q == MISSING] = np.nan
    return x


def quantize_deviations(dev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(dev).max(axis=(0, 2, 3)) / 127.0
    scale = np.where(scale > 0, scale, 1e-12)
    q = np.clip(np.round(dev / scale[None, :, None, None]), -127, 127).astype(np.int8)
    return q, scale


def _robust_range(values: list[np.ndarray], low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    finite = np.concatenate([v[np.isfinite(v)].ravel() for v in values if v is not None])
    lo, hi = np.percentile(finite, [low, high])
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    return float(lo), float(hi)


def _round(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


def encode_variable(spec: ViewVariable, members: np.ndarray, truth: np.ndarray | None, threshold: np.ndarray | None) -> tuple[dict, list[np.ndarray], np.ndarray]:
    members = spec.to_display(members)
    truth = None if truth is None else spec.to_display(truth)
    threshold = None if threshold is None else spec.to_display(threshold)
    mean = members.mean(axis=0)
    parts = [a for a in (mean, truth, threshold) if a is not None]
    lo = float(min(np.nanmin(a) for a in parts if np.isfinite(a).any()))
    hi = float(max(np.nanmax(a) for a in parts if np.isfinite(a).any()))
    step = max(hi - lo, 1e-9) / (MISSING - 1)
    dev, dev_scale = quantize_deviations(members - mean[None])
    vmin, vmax = _robust_range([mean, truth])
    center = spec.center if spec.center is not None and vmin < spec.center < vmax else None
    spread = members.std(axis=0, ddof=1) if members.shape[0] > 1 else np.zeros_like(mean)
    info = {
        "name": spec.name,
        "label": spec.label,
        "units": spec.units,
        "digits": spec.digits,
        "cmap": spec.cmap,
        "range": [_round(vmin), _round(vmax)],
        "center": center,
        "contour": spec.contour,
        "contours": spec.contours,
        "lo": lo,
        "step": step,
        "dev_scale": [float(s) for s in dev_scale],
        "spread_max": _round(max(float(np.percentile(spread, 99.5)), 1e-6)),
        "has_truth": truth is not None and bool(np.isfinite(truth).any()),
        "has_threshold": threshold is not None,
    }
    if info["has_truth"]:
        err = np.abs(mean - truth)
        info["error_max"] = _round(max(float(np.nanpercentile(err, 99.5)), 1e-6))
    blocks = [quantize_u16(mean, lo, step)]
    if info["has_truth"]:
        blocks.append(quantize_u16(truth, lo, step))
    if threshold is not None:
        blocks.append(quantize_u16(threshold, lo, step))
    return info, blocks, dev


def build_payload(variables: list[tuple[dict, list[np.ndarray], np.ndarray]]) -> tuple[bytes, list[dict]]:
    chunks = []
    offset = 0
    infos = []
    for info, blocks, _ in variables:
        names = ["mean"] + (["truth"] if info["has_truth"] else []) + (["threshold"] if info["has_threshold"] else [])
        info["offsets"] = {}
        for name, block in zip(names, blocks, strict=True):
            data = block.astype("<u2").tobytes()
            info["offsets"][name] = offset
            chunks.append(data)
            offset += len(data)
        infos.append(info)
    for info, _, dev in variables:
        data = dev.transpose(1, 0, 2, 3).astype(np.int8).tobytes()
        info["offsets"]["dev"] = offset
        chunks.append(data)
        offset += len(data)
    return b"".join(chunks), infos


def load_coastlines() -> list[list[int]]:
    return json.loads(COASTLINES.read_text(encoding="utf-8"))["lineas"]


def build_viewer(
    members: np.ndarray,
    names: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    init: str,
    lead_hours: list[int],
    truth: np.ndarray | None = None,
    thresholds: dict[str, np.ndarray] | None = None,
    variables=DEFAULT_VARIABLES,
    title: str = "Pronóstico por ensamble",
    subtitle: str = "",
    truth_label: str = "ERA5",
    compress: bool = True,
    extra: dict | None = None,
) -> str:
    members = np.asarray(members)
    if members.ndim != 5:
        raise ValueError("members debe tener forma (miembros, plazos, canales, lat, lon)")
    if members.shape[1] != len(lead_hours):
        raise ValueError("lead_hours no coincide con los plazos de members")
    chosen = available_variables(names, variables)
    if not chosen:
        raise ValueError("ninguna variable del visor está disponible en los canales")
    encoded = []
    for name in chosen:
        spec = VARIABLES[name]
        field = derived_field(members, names, name)
        true_field = None if truth is None else derived_field(np.asarray(truth), names, name)
        threshold = None if not thresholds or name not in thresholds else np.asarray(thresholds[name])
        encoded.append(encode_variable(spec, field, true_field, threshold))
    payload, infos = build_payload(encoded)
    raw = len(payload)
    if compress:
        payload = gzip.compress(payload, compresslevel=9, mtime=0)
    luts = {name: colormap_lut(name) for name in sorted({i["cmap"] for i in infos} | set(VIEW_LUTS.values()))}
    meta = {
        "version": 1,
        "title": title,
        "subtitle": subtitle,
        "init": str(np.datetime64(init, "m")),
        "lead_hours": [int(h) for h in lead_hours],
        "members": int(members.shape[0]),
        "lat": [round(float(x), 4) for x in lat],
        "lon": [round(float(x), 4) for x in lon],
        "variables": infos,
        "luts": luts,
        "view_luts": VIEW_LUTS,
        "truth_label": truth_label,
        "gzip": compress,
        "raw_bytes": raw,
        "coast": load_coastlines(),
        **{k: v for k, v in (extra or {}).items() if v is not None},
    }
    return render_viewer(meta, payload)


def render_viewer(meta: dict, payload: bytes) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__TITLE__", html_escape(meta.get("title", ""))).replace("__META__", meta_json)
    return html.replace("__PAYLOAD__", base64.b64encode(payload).decode("ascii"))


def rebuild_viewer(html: str, **updates) -> str:
    meta, payload = read_viewer(html)
    meta.update({k: v for k, v in updates.items() if v is not None})
    if meta.get("gzip"):
        payload = gzip.compress(payload, compresslevel=9, mtime=0)
    return render_viewer(meta, payload)


def read_viewer(html: str) -> tuple[dict, bytes]:
    meta_start = html.index('<script id="meta" type="application/json">') + len('<script id="meta" type="application/json">')
    meta = json.loads(html[meta_start : html.index("</script>", meta_start)].replace("<\\/", "</"))
    data_start = html.index('<script id="payload" type="application/octet-stream">') + len('<script id="payload" type="application/octet-stream">')
    payload = base64.b64decode(html[data_start : html.index("</script>", data_start)].strip())
    if meta["gzip"]:
        payload = gzip.decompress(payload)
    return meta, payload


def decode_variable(meta: dict, payload: bytes, name: str) -> dict[str, np.ndarray]:
    info = next(v for v in meta["variables"] if v["name"] == name)
    L, M = len(meta["lead_hours"]), meta["members"]
    H, W = len(meta["lat"]), len(meta["lon"])
    n = L * H * W
    out = {}
    for key in ("mean", "truth", "threshold"):
        if key in info["offsets"]:
            q = np.frombuffer(payload, dtype="<u2", count=n, offset=info["offsets"][key]).reshape(L, H, W)
            out[key] = dequantize_u16(q, info["lo"], info["step"])
    dev = np.frombuffer(payload, dtype=np.int8, count=n * M, offset=info["offsets"]["dev"]).reshape(L, M, H, W).transpose(1, 0, 2, 3)
    out["members"] = out["mean"][None] + dev * np.asarray(info["dev_scale"])[None, :, None, None]
    return out


def clim_index_of(time: np.datetime64) -> tuple[int, int]:
    time = np.datetime64(time, "s")
    month = int(time.astype("datetime64[M]").astype(int) % 12)
    hour = int((time - time.astype("datetime64[D]")).astype("timedelta64[h]").astype(int)) // 6
    return month, hour


def event_variables(names: list[str]) -> list[Event]:
    events = []
    for ev in EVENTS.values():
        try:
            sources(names, ev.variable)
        except KeyError:
            continue
        events.append(ev)
    return events


def viewer_thresholds(store, events: list[Event], valid: np.ndarray, verbose: bool = True) -> dict[str, np.ndarray]:
    table = load_thresholds(store, events, verbose=verbose)
    index = [clim_index_of(t) for t in valid]
    return {ev.variable: np.stack([table[ev.key][m, h] for m, h in index]) for ev in events}


def forecast_viewer(
    models,
    store,
    init: str,
    steps: int,
    every: int,
    ens,
    variables=DEFAULT_VARIABLES,
    device=None,
    seed: int = 0,
    title: str = "Pronóstico por ensamble",
    subtitle: str = "",
    truth_label: str = "ERA5",
    compress: bool = True,
    verbose: bool = True,
    extra: dict | None = None,
) -> str:
    import torch

    from .forecast import forecast

    t, phys = forecast(models, store, init, steps, ens, device or torch.device("cpu"), seed)
    idx = list(range(0, steps + 1, max(every, 1)))
    start = store.times[t]
    valid = start + np.asarray(idx) * np.timedelta64(6, "h")
    truth = np.full((len(idx), *phys.shape[2:]), np.nan, dtype=np.float32)
    for j, k in enumerate(idx):
        if t + k < len(store.times) and store.times[t + k] == valid[j]:
            truth[j] = store.fields[t + k]
    has_truth = bool(np.isfinite(truth).any())
    thresholds = viewer_thresholds(store, event_variables(store.names), valid, verbose) if store.stats else None
    return build_viewer(
        phys[:, idx],
        store.names,
        store.lat,
        store.lon,
        str(start),
        [6 * k for k in idx],
        truth if has_truth else None,
        thresholds,
        variables,
        title,
        subtitle,
        truth_label,
        compress,
        extra,
    )
