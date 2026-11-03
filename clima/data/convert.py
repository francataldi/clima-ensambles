from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr

from ..channels import STATIC, Channel, default_channels

WB2_ERA5_64x32 = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"


def open_wb2(path: str = WB2_ERA5_64x32) -> xr.Dataset:
    if path.startswith("gs://"):
        return xr.open_zarr(path, storage_options={"token": "anon"}, consolidated=True, chunks=None)
    return xr.open_zarr(path, consolidated=None, chunks=None)


def _grid(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    return ds["latitude"].values.astype(np.float64), ds["longitude"].values.astype(np.float64)


def _field(ds: xr.Dataset, channel: Channel) -> xr.DataArray:
    da = ds[channel.variable]
    if channel.level is not None:
        da = da.sel(level=channel.level)
    return da.transpose("time", "latitude", "longitude")


def convert(
    ds: xr.Dataset,
    out_dir: str | Path,
    start_year: int,
    end_year: int,
    channels: list[Channel] | None = None,
    verbose: bool = True,
) -> Path:
    channels = channels or default_channels()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = ds.sortby("latitude").sortby("longitude")
    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31T23:59"))
    times = ds["time"].values.astype("datetime64[s]")
    lat, lon = _grid(ds)
    shape = (len(times), len(channels), len(lat), len(lon))
    meta = {
        "channels": [asdict(c) for c in channels],
        "shape": list(shape),
        "start_year": start_year,
        "end_year": end_year,
        "step_hours": 6,
    }
    meta_path = out / "meta.json"
    fields_path = out / "fields.npy"
    progress_path = out / "progress.json"
    resume = meta_path.exists() and fields_path.exists() and json.loads(meta_path.read_text()) == meta
    if resume:
        fields = np.lib.format.open_memmap(fields_path, mode="r+")
        done = set(json.loads(progress_path.read_text())) if progress_path.exists() else set()
    else:
        fields = np.lib.format.open_memmap(fields_path, mode="w+", dtype=np.float32, shape=shape)
        done = set()
        meta_path.write_text(json.dumps(meta, indent=2))
        np.save(out / "times.npy", times.astype(np.int64))
        np.savez(out / "coords.npz", lat=lat, lon=lon)
        static = np.zeros((len(STATIC), len(lat), len(lon)), dtype=np.float32)
        for i, name in enumerate(STATIC):
            if name in ds:
                da = ds[name]
                if "time" in da.dims:
                    da = da.isel(time=0)
                static[i] = da.transpose("latitude", "longitude").values
        np.save(out / "static.npy", static)
    years = times.astype("datetime64[Y]").astype(int) + 1970
    for year in range(start_year, end_year + 1):
        if year in done:
            continue
        idx = np.flatnonzero(years == year)
        if len(idx) == 0:
            continue
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        sub = ds.isel(time=slice(i0, i1))
        block = np.empty((i1 - i0, len(channels), len(lat), len(lon)), dtype=np.float32)
        for ci, channel in enumerate(channels):
            block[:, ci] = _field(sub, channel).values
        fields[i0:i1] = block
        fields.flush()
        done.add(year)
        progress_path.write_text(json.dumps(sorted(done)))
        if verbose:
            print(f"{year}: {i1 - i0} pasos guardados", flush=True)
    return out
