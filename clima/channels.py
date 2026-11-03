from __future__ import annotations

from dataclasses import dataclass

SURFACE = (
    ("2m_temperature", "t2m", "K"),
    ("10m_u_component_of_wind", "u10", "m/s"),
    ("10m_v_component_of_wind", "v10", "m/s"),
    ("mean_sea_level_pressure", "msl", "Pa"),
)

PRESSURE = (
    ("geopotential", "z", "m2/s2", (250, 500, 850)),
    ("temperature", "t", "K", (250, 500, 850)),
    ("u_component_of_wind", "u", "m/s", (250, 500, 850)),
    ("v_component_of_wind", "v", "m/s", (250, 500, 850)),
    ("specific_humidity", "q", "kg/kg", (500, 850)),
)

STATIC = ("land_sea_mask", "geopotential_at_surface")

HEADLINE = ("z500", "t850", "t2m", "u10")


@dataclass(frozen=True)
class Channel:
    name: str
    variable: str
    units: str
    level: int | None = None


def default_channels() -> list[Channel]:
    channels = [Channel(short, var, units) for var, short, units in SURFACE]
    for var, short, units, levels in PRESSURE:
        channels.extend(Channel(f"{short}{lev}", var, units, lev) for lev in levels)
    return channels


def channel_names(channels: list[Channel]) -> list[str]:
    return [c.name for c in channels]
