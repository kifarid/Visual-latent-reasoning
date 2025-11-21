"""Helper utilities for label post-processing during export tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pytz
from astral import LocationInfo
from astral.sun import sun


@dataclass(frozen=True)
class SteeringLabelConfig:
    """Configuration options for steering-angle classification."""

    angle_unit: str = "deg"  # 'deg' | 'centideg' | 'millideg'
    straight_threshold: float = 10.0
    boundary_threshold: float = 30.0

    def to_degrees(self, value: float) -> float:
        if self.angle_unit == "deg":
            return value
        if self.angle_unit == "centideg":
            return value / 100.0
        if self.angle_unit == "millideg":
            return value / 1000.0
        raise ValueError(f"Unsupported angle_unit: {self.angle_unit}")


def direction_from_angle(angle_deg: float) -> str:
    if angle_deg < 0:
        return "right"
    if angle_deg > 0:
        return "left"
    return "straight"


def label_steering_angle(angle_deg: float, cfg: SteeringLabelConfig) -> str:
    value_deg = cfg.to_degrees(angle_deg)
    abs_angle = abs(value_deg)
    if abs_angle <= cfg.straight_threshold:
        return "straight"
    if abs_angle <= cfg.boundary_threshold:
        return "boundary"
    return "turn"


def turn_label_5(
    angle_deg: Optional[float],
    cfg: SteeringLabelConfig,
    velocity: Optional[float] = None,
) -> Optional[str]:
    """Five-way steering label with slow-speed override treated as straight."""

    if angle_deg is None:
        return None
    if velocity is not None and float(velocity) < 0.05:
        return "straight"
    base = label_steering_angle(float(angle_deg), cfg)
    if base == "straight":
        return "straight"
    direction = direction_from_angle(float(angle_deg))
    if base == "boundary":
        if direction == "left":
            return "boundary left"
        if direction == "right":
            return "boundary right"
        return "straight"
    return direction if direction in ("left", "right") else "straight"


def speed_group_tertiles(
    speed_value: Optional[float],
    q40: float,
    q60: float,
) -> Optional[str]:
    if speed_value is None:
        return None
    value = float(speed_value)
    if value < q40:
        return "slow"
    if value < q60:
        return "medium"
    return "fast"


def time_of_day_3(
    timestamp_ms: int,
    tz_name: str,
    latitude: float,
    longitude: float,
    cache: dict,
) -> str:
    tz = pytz.timezone(tz_name)
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz)
    date_key = dt.date()
    if date_key not in cache:
        location = LocationInfo("Tokyo", "Japan", tz_name, latitude=latitude, longitude=longitude)
        sun_times = sun(location.observer, date=date_key, tzinfo=tz)
        cache[date_key] = (
            sun_times["dawn"],
            sun_times["sunrise"],
            sun_times["sunset"],
            sun_times["dusk"],
        )
    dawn, sunrise, sunset, dusk = cache[date_key]
    if dawn <= dt < sunrise:
        return "dawn"
    if sunrise <= dt < sunset:
        return "day"
    if sunset <= dt < dusk:
        return "dusk"
    return "night"


def map_time_of_day(
    timestamps_ms: Iterable[int],
    tz_name: str,
    latitude: float,
    longitude: float,
) -> list[str]:
    cache: dict = {}
    return [
        time_of_day_3(ts, tz_name=tz_name, latitude=latitude, longitude=longitude, cache=cache)
        for ts in timestamps_ms
    ]


__all__ = [
    "SteeringLabelConfig",
    "direction_from_angle",
    "label_steering_angle",
    "turn_label_5",
    "speed_group_tertiles",
    "time_of_day_3",
    "map_time_of_day",
]
