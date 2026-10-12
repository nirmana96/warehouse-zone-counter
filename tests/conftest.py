import itertools
from datetime import UTC, datetime

import pytest

from wzc.config import CameraConfig
from wzc.events import VideoClock

# Red zone: x 0..100. Green zone: x 200..300. Both span y 0..100.
CONFIG = {
    "camera": "cam_test",
    "source": "data/none.mp4",
    "recording_start": "2026-01-01T08:00:00Z",
    "frame_size": [400, 200],
    "classes": ["person"],
    "anchor": "bottom_center",
    "enter_frames": 5,
    "exit_frames": 5,
    "lost_frames": 10,
    "detector": {
        "model": "yolo26n.pt",
        "tracker": "configs/trackers/botsort_reid.yaml",
        "conf": 0.3,
        "imgsz": 640,
        "device": "cpu",
    },
    "zones": [
        {
            "id": "red_a",
            "colour": "red",
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "track_repeats": True,
            "snapshot_on_enter": True,
        },
        {
            "id": "green_a",
            "colour": "green",
            "polygon": [[200, 0], [300, 0], [300, 100], [200, 100]],
            "track_repeats": False,
            "snapshot_on_enter": False,
        },
    ],
}


@pytest.fixture
def cfg_dict():
    import copy

    return copy.deepcopy(CONFIG)


@pytest.fixture
def cfg(cfg_dict):
    return CameraConfig.model_validate(cfg_dict)


@pytest.fixture
def clock():
    return VideoClock(datetime(2026, 1, 1, 8, 0, tzinfo=UTC), fps=10.0)


@pytest.fixture
def ids():
    counter = itertools.count(1)
    return lambda: f"e{next(counter)}"
