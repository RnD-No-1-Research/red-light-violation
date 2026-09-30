"""Kiểm tra ByteTrack thật với detection tổng hợp, không tải YOLO weights."""

from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes

from models.vehicle_detector import CameraTracker


def _tracker() -> CameraTracker:
    return CameraTracker(
        SimpleNamespace(
            track_high_thresh=0.4,
            track_low_thresh=0.1,
            new_track_thresh=0.4,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )
    )


def _boxes(x: float) -> Boxes:
    return Boxes(np.array([[x, 10, x + 30, 40, 0.9, 2]], dtype=np.float32), (100, 200))


def test_camera_reset_does_not_reuse_active_ids() -> None:
    """Camera mới/reset không reset bộ đếm khiến camera khác bị trùng ID."""
    a = _tracker()
    first = int(a.update(_boxes(10))[0][4])
    b = _tracker()
    second = int(b.update(_boxes(100))[0][4])
    assert first != second
    assert int(a.update(_boxes(12))[0][4]) == first
    b.reset()
    third = int(b.update(_boxes(100))[0][4])
    assert third not in (first, second)
    assert int(a.update(_boxes(14))[0][4]) == first
