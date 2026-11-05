import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from clima.ensemble import EnsembleConfig
from clima.events import derived_field
from clima.models.afno import AFNOConfig
from clima.models.forecaster import Forecaster, save_forecaster
from clima.viewer import MISSING, TEMPLATE, VARIABLES, build_viewer, decode_variable, dequantize_u16, forecast_viewer, quantize_u16, read_viewer, rebuild_viewer

ROOT = Path(__file__).resolve().parents[1]


def tiny_model(store):
    torch.manual_seed(0)
    model = Forecaster.from_store(store, AFNOConfig(dim=16, depth=1, num_blocks=2, patch=2, dropout=0.1, noise_channels=2)).eval()
    with torch.no_grad():
        for p in model.net.head.parameters():
            p.normal_(0, 0.05)
    return model


@pytest.fixture(scope="module")
def sample(store):
    rng = np.random.default_rng(0)
    truth = np.asarray(store.fields[40:43], dtype=np.float32)
    members = truth[None] + rng.normal(0, 1, (6, *truth.shape)).astype(np.float32) * truth.std(axis=(0, 2, 3), keepdims=True)[None] * 0.1
    truth = truth.copy()
    truth[2] = np.nan
    thresholds = {"t2m": truth[0, store.index("t2m")][None].repeat(3, 0) + 2.0, "ws10": np.full((3, 32, 64), 7.5)}
    return members, truth, thresholds


@pytest.fixture(scope="module")
def viewer_html(store):
    return forecast_viewer([tiny_model(store)], store, "2018-03-01T00", 8, 2, EnsembleConfig(members=5, batch=5), verbose=False)


@pytest.mark.parametrize("compress", [True, False])
def test_payload_roundtrip_within_quantization(store, sample, compress):
    members, truth, thresholds = sample
    html = build_viewer(members, store.names, store.lat, store.lon, "2018-01-11T00", [0, 6, 12], truth, thresholds, ("t2m", "ws10", "z500", "msl"), compress=compress)
    meta, payload = read_viewer(html)
    assert meta["gzip"] is compress
    assert len(payload) == meta["raw_bytes"]
    assert [v["name"] for v in meta["variables"]] == ["t2m", "ws10", "z500", "msl"]
    for info in meta["variables"]:
        spec = VARIABLES[info["name"]]
        decoded = decode_variable(meta, payload, info["name"])
        exact = spec.to_display(derived_field(members, store.names, info["name"]))
        tol = info["step"] / 2 + 1e-6 * max(1.0, abs(info["lo"]))
        assert np.abs(decoded["mean"] - exact.mean(0)).max() <= tol
        member_tol = tol + np.asarray(info["dev_scale"])[None, :, None, None] / 2 + 1e-9
        assert (np.abs(decoded["members"] - exact) <= member_tol).all()
        true = spec.to_display(derived_field(truth, store.names, info["name"]))
        assert np.isnan(decoded["truth"][2]).all()
        assert np.nanmax(np.abs(decoded["truth"] - true)) <= tol
        assert info["has_threshold"] == (info["name"] in thresholds)
        if info["has_threshold"]:
            assert np.abs(decoded["threshold"] - spec.to_display(thresholds[info["name"]])).max() <= tol
        assert info["range"][0] < info["range"][1]
        assert info["spread_max"] > 0 and info["error_max"] > 0
        assert set(meta["luts"]) >= {info["cmap"], *meta["view_luts"].values()}
    t2m = meta["variables"][0]
    assert t2m["units"] == "°C" and t2m["center"] == 0.0


def test_quantize_handles_missing():
    x = np.array([[1.0, np.nan], [3.0, 2.5]])
    q = quantize_u16(x, 1.0, 0.5)
    assert q[0, 1] == MISSING
    back = dequantize_u16(q, 1.0, 0.5)
    assert np.isnan(back[0, 1]) and np.allclose(back[[0, 1, 1], [0, 0, 1]], [1.0, 3.0, 2.5])


def test_html_is_single_self_contained_file(viewer_html):
    head, _, rest = viewer_html.partition('<script id="payload" type="application/octet-stream">')
    tail = rest.partition("</script>")[2]
    page = head + tail
    lowered = page.lower()
    for forbidden in ("http://", "https://", "<link", "@import", " src=", "<iframe", "<img"):
        assert forbidden not in lowered, forbidden
    style = page.split("<style>", 1)[1].split("</style>", 1)[0]
    assert "url(" not in style
    assert page.count("<script") == 2
    assert "__META__" not in viewer_html and "__PAYLOAD__" not in viewer_html
    assert "<title>Pronóstico por ensamble</title>" in viewer_html


def test_template_has_no_comments():
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "<!--" not in template
    assert "/*" not in template
    script = template.split("<script>\n", 1)[1]
    assert not re.search(r"(^|\s)//", script)


def test_forecast_viewer_contents(store, viewer_html):
    meta, payload = read_viewer(viewer_html)
    assert meta["lead_hours"] == [0, 12, 24, 36, 48]
    assert meta["members"] == 5
    assert meta["init"] == "2018-03-01T00:00"
    assert len(meta["lat"]) == 32 and len(meta["lon"]) == 64
    assert len(meta["coast"]) > 100
    names = [v["name"] for v in meta["variables"]]
    assert names == ["t2m", "ws10", "msl", "z500", "t850", "ws250", "q850"]
    flags = {v["name"]: (v["has_truth"], v["has_threshold"]) for v in meta["variables"]}
    assert flags["t2m"] == (True, True) and flags["ws10"] == (True, True) and flags["z500"] == (True, False)
    t2m = decode_variable(meta, payload, "t2m")
    era5 = np.asarray(store.fields[store.time_range("2018-03-01T00", "2018-03-01T00")[0], store.index("t2m")]) - 273.15
    assert np.abs(t2m["truth"][0] - era5).max() < 0.01
    assert np.abs(t2m["members"][0, 0] - era5).max() < 0.05
    assert t2m["members"][:, -1].std(axis=0).mean() > 0


def test_rebuild_keeps_data_and_adds_presentation(viewer_html):
    meta, payload = read_viewer(viewer_html)
    rebuilt = rebuild_viewer(viewer_html, repo="https://github.com/usuario/repo", model_text="texto del modelo")
    new_meta, new_payload = read_viewer(rebuilt)
    assert new_payload == payload
    assert new_meta["repo"] == "https://github.com/usuario/repo"
    assert new_meta["model_text"] == "texto del modelo"
    assert {k: v for k, v in new_meta.items() if k not in ("repo", "model_text")} == meta
    assert 'id="intro"' in rebuilt and 'id="repo"' in rebuilt


def test_viewer_script_rebuild(viewer_html, tmp_path):
    source = tmp_path / "viejo.html"
    source.write_text(viewer_html, encoding="utf-8")
    out = tmp_path / "docs" / "index.html"
    script = Path(__file__).resolve().parents[1] / "scripts" / "viewer.py"
    proc = subprocess.run([sys.executable, str(script), "--rebuild", str(source), "--out", str(out)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    meta, _ = read_viewer(out.read_text(encoding="utf-8"))
    assert meta["repo"].startswith("https://github.com/")
    assert meta["model_text"]


def test_viewer_script_end_to_end(store, tmp_path):
    ckpt = tmp_path / "modelo.pt"
    save_forecaster(ckpt, tiny_model(store))
    out = tmp_path / "visor" / "visor.html"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "viewer.py"), "--checkpoints", str(ckpt), "--data", str(store.root),
        "--init", "2018-03-02T06", "--steps", "4", "--every", "1", "--members", "3", "--variables", "t2m", "z500",
        "--device", "cpu", "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    meta, _ = read_viewer(out.read_text(encoding="utf-8"))
    assert [v["name"] for v in meta["variables"]] == ["t2m", "z500"]
    assert meta["lead_hours"] == [0, 6, 12, 18, 24]
    assert "condiciones iniciales + ruido" in meta["subtitle"]


def _browser(playwright):
    try:
        return playwright.chromium.launch()
    except Exception:
        pass
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    for exe in sorted(Path(root).glob("chromium-*/chrome-*/chrome"), reverse=True) if root else []:
        try:
            return playwright.chromium.launch(executable_path=str(exe))
        except Exception:
            continue
    pytest.skip("no hay Chromium disponible para Playwright")


def test_viewer_runs_in_browser(viewer_html, tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    path = tmp_path / "visor.html"
    path.write_text(viewer_html, encoding="utf-8")
    with sync_api.sync_playwright() as p:
        browser = _browser(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(path.as_uri())
            page.wait_for_function("document.body.dataset.ready === '1' || document.body.dataset.error", timeout=30000)
            assert page.evaluate("document.body.dataset.error || ''") == ""
            assert page.locator("#point").inner_text().startswith("Buenos Aires")
            box = page.locator("#map").bounding_box()
            page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.3)
            page.wait_for_function("document.getElementById('point').textContent.startsWith('Punto elegido')", timeout=5000)
            disabled = {}
            for i in range(page.locator("#var option").count()):
                page.select_option("#var", str(i))
                for view in ("miembro", "media", "dispersion", "verdad", "error", "prob"):
                    button = page.locator(f"#view button[data-v={view}]")
                    if button.is_disabled():
                        disabled.setdefault(i, []).append(view)
                        continue
                    button.click()
                    assert button.get_attribute("aria-pressed") == "true"
                page.wait_for_timeout(30)
            assert all(views == ["prob"] for views in disabled.values()) and len(disabled) == 5
            page.click("#proj button[data-v=globo]")
            page.keyboard.press("ArrowRight")
            page.wait_for_function("document.getElementById('lead-badge').textContent === '+12 h'", timeout=5000)
            page.click("#theme")
            assert page.evaluate("document.documentElement.dataset.theme") in ("dark", "light")
            page.wait_for_timeout(100)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(200)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert errors == []
        finally:
            browser.close()
