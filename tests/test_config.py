import pytest
from pydantic import ValidationError

from wzc.config import CameraConfig


def test_valid_config_loads(cfg):
    assert cfg.zone("red_a").colour == "red"


def test_polygon_needs_three_points(cfg_dict):
    cfg_dict["zones"][0]["polygon"] = [[0, 0], [10, 10]]
    with pytest.raises(ValidationError, match="at least 3 points"):
        CameraConfig.model_validate(cfg_dict)


def test_green_zone_cannot_track_repeats(cfg_dict):
    cfg_dict["zones"][1]["track_repeats"] = True
    with pytest.raises(ValidationError, match="green zones"):
        CameraConfig.model_validate(cfg_dict)


def test_needs_one_red_and_one_green(cfg_dict):
    cfg_dict["zones"][1]["colour"] = "red"
    cfg_dict["zones"][1]["track_repeats"] = True
    with pytest.raises(ValidationError, match="exactly one red and one green"):
        CameraConfig.model_validate(cfg_dict)


def test_point_outside_frame_is_rejected(cfg_dict):
    cfg_dict["zones"][0]["polygon"][2] = [500, 100]
    with pytest.raises(ValidationError, match="outside the 400x200 frame"):
        CameraConfig.model_validate(cfg_dict)


def test_recording_start_needs_timezone(cfg_dict):
    cfg_dict["recording_start"] = "2026-01-01T08:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        CameraConfig.model_validate(cfg_dict)


def test_unknown_key_is_rejected(cfg_dict):
    cfg_dict["enter_frame"] = 3  # typo for enter_frames
    with pytest.raises(ValidationError):
        CameraConfig.model_validate(cfg_dict)
