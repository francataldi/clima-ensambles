from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..channels import PRESSURE, STATIC, SURFACE

LEVELS = (250, 500, 850, 1000)


def wb2_grid(n_lon: int = 64, n_lat: int = 32) -> tuple[np.ndarray, np.ndarray]:
    dlat = 180.0 / n_lat
    lat = -90.0 + dlat / 2 + dlat * np.arange(n_lat)
    lon = 360.0 / n_lon * np.arange(n_lon)
    return lat, lon


def make_synthetic(start: str, end: str, seed: int = 0, n_lon: int = 64, n_lat: int = 32) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="6h")
    lat, lon = wb2_grid(n_lon, n_lat)
    t = np.arange(len(times))[:, None, None]
    la = np.deg2rad(lat)[None, None, :]
    lo = np.deg2rad(lon)[None, :, None]
    doy = times.dayofyear.values[:, None, None]
    season = np.sin(2 * np.pi * doy / 365.25)
    hour = times.hour.values[:, None, None]
    diurnal = np.cos(2 * np.pi * hour / 24 + lo)
    waves = sum(
        np.sin(k * lo - 2 * np.pi * t / (16 + 8 * k) + p) * np.cos(la) ** k
        for k, p in zip((2, 3, 5), rng.uniform(0, 2 * np.pi, 3))
    )
    noise = rng.normal(0, 0.05, size=(len(times), n_lon, n_lat))
    base = waves + noise
    data = {}
    temp = 288 - 40 * np.sin(la) ** 2 + 8 * season * np.sin(la) + 3 * diurnal * np.cos(la) + 4 * base
    data["2m_temperature"] = (("time", "longitude", "latitude"), temp.astype(np.float32))
    data["10m_u_component_of_wind"] = (("time", "longitude", "latitude"), (5 * np.cos(2 * la) + 3 * base).astype(np.float32))
    data["10m_v_component_of_wind"] = (("time", "longitude", "latitude"), (2 * np.roll(base, 3, axis=1)).astype(np.float32))
    data["mean_sea_level_pressure"] = (("time", "longitude", "latitude"), (101300 + 800 * base - 500 * np.cos(2 * la)).astype(np.float32))
    lev = np.array(LEVELS, dtype=np.float64)[None, :, None, None]
    lb = base[:, None]
    lt = temp[:, None]
    data["geopotential"] = (("time", "level", "longitude", "latitude"), (9.81 * (16000 * np.log(1000 / lev)) + 600 * lb).astype(np.float32))
    data["temperature"] = (("time", "level", "longitude", "latitude"), (lt - 0.0065 * 16000 * np.log(1000 / lev) * 0.5 + 2 * lb).astype(np.float32))
    data["u_component_of_wind"] = (("time", "level", "longitude", "latitude"), (10 * (1000 / lev) ** 0.3 * np.cos(2 * la)[:, None] + 4 * lb).astype(np.float32))
    data["v_component_of_wind"] = (("time", "level", "longitude", "latitude"), (3 * np.roll(lb, 5, axis=2) * np.ones_like(lev)).astype(np.float32))
    data["specific_humidity"] = (("time", "level", "longitude", "latitude"), (0.01 * np.exp(-(1000 - lev) / 300) * np.cos(la)[:, None] ** 2 * (1 + 0.1 * lb)).astype(np.float32))
    land = (np.sin(3 * np.deg2rad(lon))[:, None] * np.cos(np.deg2rad(lat))[None, :] > 0.2).astype(np.float32)
    data["land_sea_mask"] = (("longitude", "latitude"), land)
    data["geopotential_at_surface"] = (("longitude", "latitude"), (land * 3000).astype(np.float32))
    ds = xr.Dataset(data, coords={"time": times, "level": list(LEVELS), "longitude": lon, "latitude": lat})
    expected = {v for v, *_ in SURFACE} | {v for v, *_ in PRESSURE} | set(STATIC)
    assert expected <= set(ds.data_vars)
    return ds
