from __future__ import annotations

import numpy as np

SOLAR_CONSTANT = 1361.0


def toa_insolation(times: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype="datetime64[s]")
    days = (t - np.datetime64("2000-01-01T12:00:00", "s")).astype(np.float64) / 86400.0
    gamma = 2.0 * np.pi * ((days % 365.2422) / 365.2422)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    distance = 1.000110 + 0.034221 * np.cos(gamma) + 0.001280 * np.sin(gamma) + 0.000719 * np.cos(2 * gamma) + 0.000077 * np.sin(2 * gamma)
    hours = (t - t.astype("datetime64[D]")).astype(np.float64) / 3600.0
    phi = np.deg2rad(lat)[None, :, None]
    hour_angle = np.deg2rad((hours[:, None, None] - 12.0) * 15.0 + lon[None, None, :])
    d = decl[:, None, None]
    cos_zenith = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(hour_angle)
    return (SOLAR_CONSTANT * distance[:, None, None] * np.clip(cos_zenith, 0.0, None)).astype(np.float32)


def time_features(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype="datetime64[s]")
    year_start = t.astype("datetime64[Y]")
    doy = (t - year_start).astype(np.float64) / 86400.0
    hours = (t - t.astype("datetime64[D]")).astype(np.float64) / 3600.0
    a = 2 * np.pi * doy / 365.2422
    b = 2 * np.pi * hours / 24.0
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], axis=-1).astype(np.float32)
