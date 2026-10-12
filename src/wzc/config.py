"""Camera and zone config: YAML loaded into Pydantic models, checked at startup.

A bad config stops the program with a clear message. Nothing runs on defaults the user
did not write down.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wzc.geometry import polygon_area

SUPPORTED_CLASSES = {"person"}


class Zone(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    colour: Literal["red", "green"]
    polygon: list[tuple[float, float]]
    track_repeats: bool
    snapshot_on_enter: bool

    @field_validator("polygon")
    @classmethod
    def _polygon_shape(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("a zone polygon needs at least 3 points")
        if polygon_area(v) < 1.0:
            raise ValueError("zone polygon has no area (points on one line?)")
        return v

    @model_validator(mode="after")
    def _green_rules(self) -> Zone:
        # Green zones count traffic only: no per-person repeats, no snapshots.
        if self.colour == "green" and (self.track_repeats or self.snapshot_on_enter):
            raise ValueError(
                f"zone {self.id}: green zones must have track_repeats and snapshot_on_enter false"
            )
        return self


class Detector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    tracker: Path
    conf: float = Field(gt=0, lt=1)
    imgsz: int = Field(ge=320, le=1920)
    device: str


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    camera: str = Field(pattern=r"^[a-z0-9_]+$")
    source: Path
    recording_start: datetime
    frame_size: tuple[int, int]
    classes: list[str]
    anchor: Literal["bottom_center"]
    enter_frames: int = Field(ge=1, le=60)
    exit_frames: int = Field(ge=1, le=60)
    lost_frames: int = Field(ge=1, le=600)
    detector: Detector
    zones: list[Zone]

    @field_validator("recording_start")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("recording_start needs a timezone, e.g. 2026-01-01T08:00:00Z")
        return v

    @field_validator("classes")
    @classmethod
    def _classes(cls, v: list[str]) -> list[str]:
        unknown = set(v) - SUPPORTED_CLASSES
        if not v or unknown:
            raise ValueError(f"classes must be a subset of {sorted(SUPPORTED_CLASSES)}")
        return v

    @model_validator(mode="after")
    def _zones(self) -> CameraConfig:
        ids = [z.id for z in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError("zone ids must be unique")
        colours = sorted(z.colour for z in self.zones)
        if colours != ["green", "red"]:
            raise ValueError("expected exactly one red and one green zone")
        w, h = self.frame_size
        for z in self.zones:
            for x, y in z.polygon:
                if not (0 <= x <= w and 0 <= y <= h):
                    raise ValueError(f"zone {z.id}: point ({x}, {y}) is outside the {w}x{h} frame")
        return self

    def zone(self, zone_id: str) -> Zone:
        for z in self.zones:
            if z.id == zone_id:
                return z
        raise KeyError(zone_id)


def load_camera_config(path: str | Path) -> CameraConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CameraConfig.model_validate(data)
