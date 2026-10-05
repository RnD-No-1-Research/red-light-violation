"""Common policy applies to arbitrary cameras without leaking per-camera state."""

from dataclasses import replace
from math import hypot
import threading

import pytest
import yaml

from controllers.violation_controller import ViolationController
from models.types import Detection
from tests.test_multi_camera import bundle
from tests.test_occlusion_review import packet
from utils.config_loader import (
    ROOT,
    LATE_DETECTION_DEFAULTS,
    late_detection_settings,
    load_cameras,
    load_settings,
    read_yaml,
)


def test_existing_new_and_rtsp_inherit(settings, camera):
    cameras = load_cameras(ROOT / "config/cameras.yaml")
    cameras += [
        replace(camera, camera_id="new_camera"),
        replace(camera, camera_id="rtsp_new", source="rtsp://example.invalid/live"),
    ]
    for config in cameras:
        assert late_detection_settings(config, settings) == LATE_DETECTION_DEFAULTS
        controller = ViolationController(config, settings, bundle(), threading.Event())
        assert controller.late.enabled
        assert controller.late.require_forward_motion
        assert controller.late.start == config.point1
        assert controller.late.direction == config.direction


@pytest.mark.parametrize(
    "camera_change,global_change,expected_enabled,expected_motion",
    [
        ({}, {"late_detection_enabled": False}, False, True),
        ({"late_detection_enabled": False}, {}, False, True),
        (
            {"late_detection_enabled": True},
            {"late_detection_enabled": False},
            True,
            True,
        ),
        ({"late_detection_require_forward_motion": True}, {}, True, True),
        (
            {"late_detection_require_forward_motion": False},
            {"late_detection_require_forward_motion": True},
            True,
            False,
        ),
    ],
)
def test_override_precedence(
    camera, settings, camera_change, global_change, expected_enabled, expected_motion
):
    controller = ViolationController(
        replace(camera, **camera_change),
        {**settings, **global_change},
        bundle(),
        threading.Event(),
    )
    assert controller.late.enabled is expected_enabled
    assert controller.late.require_forward_motion is expected_motion


def test_geometry_and_same_id_isolated_for_all_camera_orientations(settings):
    controllers = [
        ViolationController(c, settings, bundle(), threading.Event())
        for c in load_cameras(ROOT / "config/cameras.yaml")
    ]
    # Include a vertical line to avoid encoding any horizontal-image assumption.
    config = replace(
        controllers[0].camera,
        camera_id="vertical_new",
        point1=(100, 0),
        point2=(100, 200),
        direction="negative_to_positive",
    )
    controllers.append(
        ViolationController(config, settings, bundle(), threading.Event())
    )
    for controller in controllers:
        d = controller.late
        d.update(packet(1, 0), [], "RED", "RED", set())
        d.update(packet(2, 1), [], "RED", "RED", set())
        ax, ay = d.start
        bx, by = d.end
        length = hypot(bx - ax, by - ay)
        sign = 1 if d.direction == "negative_to_positive" else -1
        x = round((ax + bx) / 2 + sign * -(by - ay) / length * 10)
        y = round((ay + by) / 2 + sign * (bx - ax) / length * 10)
        vehicle = Detection((x - 10, y - 20, x + 10, y + 20), 0.9, "motorcycle", 7)
        results = []
        for i in range(3):
            results = d.update(
                packet(3 + i, 1.1 + i * 0.1), [vehicle], "RED", "RED", set()
            )
        assert not results
        assert 7 in d.suspected
        assert d.new_suspicions[0].details["reason"] == "SUSPECTED_LATE_DIRECTION_UNKNOWN"
    # Reset one camera cannot change another camera's history.
    controllers[0].late.reset()
    assert not controllers[0].late.seen
    assert 7 in controllers[1].late.seen


@pytest.mark.parametrize(
    "change",
    [
        {"late_detection_enabled": "true"},
        {"late_detection_require_forward_motion": 0},
        {"late_detection_band_height_ratio": 3},
        {"late_detection_min_red_seconds": 0},
        {"late_detection_enabled": None},
    ],
)
def test_invalid_common_config_rejected(tmp_path, change):
    data = read_yaml(ROOT / "config/settings.yaml")
    data.update(change)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(path)


def test_missing_common_keys_use_defaults_and_camera_can_override_numeric(
    tmp_path, camera
):
    data = read_yaml(ROOT / "config/settings.yaml")
    for key in LATE_DETECTION_DEFAULTS:
        data.pop(key, None)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = load_settings(path)
    assert late_detection_settings(camera, settings) == LATE_DETECTION_DEFAULTS
    changed = replace(
        camera, late_detection_band_height_ratio=0.5, late_detection_min_red_seconds=2
    )
    effective = late_detection_settings(changed, settings)
    assert effective["late_detection_band_height_ratio"] == 0.5
    assert effective["late_detection_min_red_seconds"] == 2
    assert settings["late_detection_band_height_ratio"] == 1
