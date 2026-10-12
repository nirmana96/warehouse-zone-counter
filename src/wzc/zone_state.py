"""Enter/exit state machine per track and zone.

Takes tracks, returns events. Knows nothing about video files, SQL or the dashboard,
so it can be tested with plain lists.

Rules:
- A track enters a zone after its foot point is inside for `enter_frames` observations
  in a row, and leaves after `exit_frames` observations in a row outside. Someone standing
  on the line does not flicker in and out.
- The entry time is the first frame of the inside streak, the exit time the first frame
  of the outside streak, so dwell is not padded by the confirmation delay.
- Repeat entries are numbered per track only for zones with track_repeats (the red zone).
- A track missing for more than `lost_frames` frames is closed: any open visit is exited
  with reason "track_lost", then a track_lost event is written.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from wzc.config import CameraConfig, Zone
from wzc.detect_track import Track
from wzc.events import (
    TRACK_LOST,
    TRACK_STARTED,
    ZONE_ENTERED,
    ZONE_EXITED,
    Event,
    IdFactory,
    VideoClock,
    new_event_id,
)
from wzc.geometry import as_contour, foot_point, inside

log = logging.getLogger(__name__)


@dataclass
class _ZoneTrack:
    inside: bool = False
    in_streak: int = 0
    in_start: int = 0
    out_streak: int = 0
    out_start: int = 0
    entered_frame: int = 0
    visit_no: int | None = None


@dataclass
class _TrackState:
    first_seen: int
    last_seen: int
    zones: dict[str, _ZoneTrack] = field(default_factory=dict)


class ZoneStateMachine:
    def __init__(
        self,
        cfg: CameraConfig,
        clock: VideoClock,
        id_factory: IdFactory = new_event_id,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._new_id = id_factory
        self._zones: list[Zone] = list(cfg.zones)
        self._contours: dict[str, np.ndarray] = {z.id: as_contour(z.polygon) for z in cfg.zones}
        self._tracks: dict[int, _TrackState] = {}
        # Visit counter per (track, zone) for zones with track_repeats. Kept for the whole
        # run, so a track that leaves and comes back gets visit 2, 3, ...
        self._visit_counts: dict[tuple[int, str], int] = {}

    # ---- public -------------------------------------------------------------------------

    def step(self, frame_idx: int, tracks: list[Track]) -> list[Event]:
        events: list[Event] = []
        for t in tracks:
            state = self._tracks.get(t.track_id)
            if state is None:
                state = _TrackState(first_seen=frame_idx, last_seen=frame_idx)
                state.zones = {z.id: _ZoneTrack() for z in self._zones}
                self._tracks[t.track_id] = state
                events.append(
                    self._event(
                        TRACK_STARTED, frame_idx, t.track_id, None, {"box": _round_box(t.xyxy)}
                    )
                )
            state.last_seen = frame_idx
            foot = foot_point(t.xyxy)
            for z in self._zones:
                is_in = inside(foot, self._contours[z.id])
                events.extend(
                    self._update_zone(frame_idx, t.track_id, z, state.zones[z.id], is_in, foot)
                )

        seen = {t.track_id for t in tracks}
        for track_id in [tid for tid in self._tracks if tid not in seen]:
            state = self._tracks[track_id]
            if frame_idx - state.last_seen > self._cfg.lost_frames:
                events.extend(self._close_track(frame_idx, track_id, reason="track_lost"))
        return events

    def flush(self, frame_idx: int) -> list[Event]:
        """Close every open track at the end of the video so no visit is left open."""
        events: list[Event] = []
        for track_id in list(self._tracks):
            events.extend(self._close_track(frame_idx, track_id, reason="stream_ended"))
        return events

    def zones_occupied(self, track_id: int) -> list[str]:
        """Zone ids this track is currently inside (for drawing the overlay)."""
        state = self._tracks.get(track_id)
        if state is None:
            return []
        return [zid for zid, zt in state.zones.items() if zt.inside]

    # ---- internals ----------------------------------------------------------------------

    def _update_zone(
        self,
        frame_idx: int,
        track_id: int,
        zone: Zone,
        zt: _ZoneTrack,
        is_in: bool,
        foot: tuple[float, float],
    ) -> list[Event]:
        if is_in:
            zt.out_streak = 0
            if zt.in_streak == 0:
                zt.in_start = frame_idx
            zt.in_streak += 1
            if not zt.inside and zt.in_streak >= self._cfg.enter_frames:
                return [self._enter(frame_idx, track_id, zone, zt, foot)]
            return []

        zt.in_streak = 0
        if zt.inside:
            if zt.out_streak == 0:
                zt.out_start = frame_idx
            zt.out_streak += 1
            if zt.out_streak >= self._cfg.exit_frames:
                return [
                    self._exit(track_id, zone, zt, exit_frame=zt.out_start, reason="walked_out")
                ]
        return []

    def _enter(
        self, frame_idx: int, track_id: int, zone: Zone, zt: _ZoneTrack, foot: tuple[float, float]
    ) -> Event:
        zt.inside = True
        zt.entered_frame = zt.in_start
        payload: dict = {
            "colour": zone.colour,
            "foot_point": [round(foot[0], 1), round(foot[1], 1)],
            "confirmed_at": self._clock.iso(frame_idx),
        }
        if zone.track_repeats:
            key = (track_id, zone.id)
            self._visit_counts[key] = self._visit_counts.get(key, 0) + 1
            zt.visit_no = self._visit_counts[key]
            payload["visit_no"] = zt.visit_no
        else:
            zt.visit_no = None
        return self._event(ZONE_ENTERED, zt.in_start, track_id, zone.id, payload)

    def _exit(
        self, track_id: int, zone: Zone, zt: _ZoneTrack, exit_frame: int, reason: str
    ) -> Event:
        exit_frame = max(exit_frame, zt.entered_frame)
        payload: dict = {
            "colour": zone.colour,
            "entered_at": self._clock.iso(zt.entered_frame),
            "dwell_s": round(self._clock.seconds(exit_frame - zt.entered_frame), 3),
            "reason": reason,
        }
        if zt.visit_no is not None:
            payload["visit_no"] = zt.visit_no
        zt.inside = False
        zt.in_streak = 0
        zt.out_streak = 0
        zt.visit_no = None
        return self._event(ZONE_EXITED, exit_frame, track_id, zone.id, payload)

    def _close_track(self, frame_idx: int, track_id: int, reason: str) -> list[Event]:
        state = self._tracks.pop(track_id)
        events: list[Event] = []
        for z in self._zones:
            zt = state.zones[z.id]
            if zt.inside:
                # Already walking out when lost: use where the outside streak began.
                exit_frame = zt.out_start if zt.out_streak > 0 else state.last_seen + 1
                events.append(self._exit(track_id, z, zt, exit_frame=exit_frame, reason=reason))
        events.append(
            self._event(
                TRACK_LOST,
                frame_idx,
                track_id,
                None,
                {"last_seen": self._clock.iso(state.last_seen), "reason": reason},
            )
        )
        return events

    def _event(
        self,
        event_type: str,
        frame_idx: int,
        track_id: int | None,
        zone_id: str | None,
        payload: dict,
    ) -> Event:
        return Event(
            event_id=self._new_id(),
            event_type=event_type,
            occurred_at=self._clock.iso(frame_idx),
            camera_id=self._cfg.camera,
            track_id=track_id,
            zone_id=zone_id,
            payload={"frame": frame_idx, **payload},
        )


def _round_box(xyxy: tuple[float, float, float, float]) -> list[float]:
    return [round(v, 1) for v in xyxy]
