"""Event types and the video clock.

Events are facts: written once, never changed. Every count the dashboard shows is
rebuilt from them later.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

TRACK_STARTED = "track_started"
ZONE_ENTERED = "zone_entered"
ZONE_EXITED = "zone_exited"
SNAPSHOT_SAVED = "snapshot_saved"
TRACK_LOST = "track_lost"

EVENT_TYPES = {TRACK_STARTED, ZONE_ENTERED, ZONE_EXITED, SNAPSHOT_SAVED, TRACK_LOST}


@dataclass(frozen=True)
class Event:
    event_id: str
    event_type: str
    occurred_at: str  # video time, ISO 8601 UTC
    camera_id: str
    track_id: int | None
    zone_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {self.event_type}")


def new_event_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class VideoClock:
    """Turns a frame index into video time.

    The footage has no wall clock, so time 0 is recording_start from the config.
    Never use the computer's clock for event times.
    """

    recording_start: datetime
    fps: float

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.recording_start.tzinfo is None:
            raise ValueError("recording_start must be timezone-aware")

    def seconds(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def iso(self, frame_idx: int) -> str:
        t = self.recording_start + timedelta(seconds=self.seconds(frame_idx))
        return t.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


IdFactory = Callable[[], str]
