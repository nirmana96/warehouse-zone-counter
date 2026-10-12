"""Preview run: track people on one camera, test zones, print the events.

Usage:
    uv run python tools/run_tracking.py --config configs/zones/cam_04.yaml --max-frames 900 --show
    uv run python tools/run_tracking.py --config configs/zones/cam_04.yaml \
        --events-out outputs/cam_04_events.jsonl

This is the check that tracking and zone logic work before the SQL event store exists.
--show opens a local window with zones, boxes, foot points and track IDs. It is for your
eyes only: faces are not blurred yet (that arrives with privacy.py), so do not screen-record
it and nothing from it is saved. The JSONL file holds events only, no images.

The counts printed at the end come from the events of this run. From the next step on,
the dashboard counts come from SQL projections instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from wzc.config import CameraConfig, load_camera_config
from wzc.detect_track import Track, Tracker
from wzc.events import ZONE_ENTERED, ZONE_EXITED, Event, VideoClock
from wzc.geometry import foot_point
from wzc.video import VideoReader
from wzc.zone_state import ZoneStateMachine

COLOURS_BGR = {"red": (40, 40, 230), "green": (60, 180, 60)}
NEUTRAL = (230, 200, 60)


def draw_overlay(
    frame: np.ndarray,
    cfg: CameraConfig,
    tracks: list[Track],
    zsm: ZoneStateMachine,
    counts: Counter,
) -> np.ndarray:
    img = frame.copy()
    layer = img.copy()
    for z in cfg.zones:
        pts = np.array(z.polygon, dtype=np.int32)
        cv2.fillPoly(layer, [pts], COLOURS_BGR[z.colour])
        cv2.polylines(img, [pts], True, COLOURS_BGR[z.colour], 2)
    img = cv2.addWeighted(layer, 0.2, img, 0.8, 0)

    red_ids = {z.id for z in cfg.zones if z.colour == "red"}
    for t in tracks:
        occupied = set(zsm.zones_occupied(t.track_id))
        colour = (
            COLOURS_BGR["red"]
            if occupied & red_ids
            else (COLOURS_BGR["green"] if occupied else NEUTRAL)
        )
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        fx, fy = (int(v) for v in foot_point(t.xyxy))
        cv2.circle(img, (fx, fy), 6, colour, -1)
        cv2.putText(
            img, f"ID {t.track_id}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2
        )

    text = f"red entries {counts['red']}   green visits {counts['green']}"
    cv2.putText(img, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(img, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--show", action="store_true", help="local preview window (not blurred, never saved)"
    )
    parser.add_argument("--events-out", type=Path, default=None, help="write events as JSON lines")
    parser.add_argument("--quiet", action="store_true", help="do not print each event")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_camera_config(args.config)

    with VideoReader(cfg.source) as video:
        if tuple(video.frame_size) != tuple(cfg.frame_size):
            raise SystemExit(
                f"video is {video.frame_size} but config says {cfg.frame_size}; redraw the zones"
            )
        clock = VideoClock(cfg.recording_start, video.fps)
        tracker = Tracker(cfg.detector, cfg.classes)
        zsm = ZoneStateMachine(cfg, clock)

        out = None
        if args.events_out:
            args.events_out.parent.mkdir(parents=True, exist_ok=True)
            out = args.events_out.open("w", encoding="utf-8")

        counts: Counter = Counter()
        repeats: Counter = Counter()
        all_events: list[Event] = []

        def handle(events: list[Event]) -> None:
            for e in events:
                all_events.append(e)
                if e.event_type == ZONE_ENTERED:
                    counts[e.payload["colour"]] += 1
                    if "visit_no" in e.payload:
                        repeats[e.track_id] = e.payload["visit_no"]
                if out:
                    out.write(json.dumps(asdict(e)) + "\n")
                if not args.quiet and e.event_type in (ZONE_ENTERED, ZONE_EXITED):
                    parts = [e.occurred_at, f"{e.event_type:<12}", f"id={e.track_id:<4}", e.zone_id]
                    if "visit_no" in e.payload:
                        parts.append(f"visit {e.payload['visit_no']}")
                    if "dwell_s" in e.payload:
                        parts.append(f"dwell {e.payload['dwell_s']}s ({e.payload['reason']})")
                    print("  ".join(str(p) for p in parts))

        start = time.perf_counter()
        last_idx = -1
        try:
            for idx, frame in video.frames(args.max_frames):
                tracks = tracker.update(frame)
                handle(zsm.step(idx, tracks))
                last_idx = idx
                if args.show:
                    cv2.imshow("tracking preview", draw_overlay(frame, cfg, tracks, zsm, counts))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            handle(zsm.flush(last_idx + 1))
            if out:
                out.close()
            if args.show:
                cv2.destroyAllWindows()
        elapsed = time.perf_counter() - start

    frames = last_idx + 1
    print("\n--- summary ---")
    print(f"frames {frames}, {elapsed:.1f} s, {frames / elapsed:.1f} FPS (includes model warm-up)")
    print(f"events {len(all_events)}")
    print(f"red entries {counts['red']}, green visits {counts['green']}")
    multi = {tid: n for tid, n in repeats.items() if n >= 2}
    print(f"track IDs in red: {len(repeats)}, with 2+ red entries: {multi or 'none'}")
    if args.events_out:
        print(f"events written to {args.events_out}")


if __name__ == "__main__":
    main()
