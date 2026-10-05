"""Kiểm tra lớp xe custom, COCO và tên xe máy viết khác nhau."""

from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
from ultralytics.engine.results import Boxes

from models._yolo import LockedYOLO
from models.vehicle_detector import VehicleDetector


def detector_with_names(
    monkeypatch: pytest.MonkeyPatch, names: dict
) -> VehicleDetector:
    """Giả bước nạp checkpoint để test ánh xạ mà không cần weights."""

    def init(self: LockedYOLO, path: Path, device: str, imgsz: int) -> None:
        self.model = SimpleNamespace(names=names)
        self.lock, self.device, self.imgsz = RLock(), device, imgsz

    monkeypatch.setattr(LockedYOLO, "__init__", init)
    return VehicleDetector(Path("unused.pt"), "cpu", 640)


def test_custom_four_classes_and_motorcycle_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Giữ class ID của model custom, chuẩn hóa tên trong kết quả tracking."""
    names = {0: "bus", 1: "car", 2: "motobike", 3: "truck"}
    detector = detector_with_names(monkeypatch, names)
    assert detector.classes == [0, 1, 2, 3]
    assert detector.class_names == {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}
    seen = []
    boxes = Boxes(np.array([[10, 10, 30, 40, 0.9, 2]], dtype=np.float32), (100, 100))

    def predict(frame: np.ndarray, **kwargs: object) -> list:
        seen.append(kwargs["classes"])
        return [SimpleNamespace(names=names, boxes=boxes)]

    detector.model.predict = predict
    detector.trackers = {
        "test": SimpleNamespace(
            update=lambda boxes, frame: np.array([[10, 10, 30, 40, 7, 0.9, 2, 0]])
        )
    }
    result = detector.track(np.zeros((100, 100, 3), np.uint8), "test", 0.5)
    assert seen == [[0, 1, 2, 3]]
    assert result[0].class_name == "motorcycle"
    assert result[0].track_id == 7


def test_coco_ids_are_not_replaced_by_custom_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Không lấy ID 0..3 của custom áp vào COCO; loại lớp person."""
    detector = detector_with_names(
        monkeypatch,
        {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
    )
    assert detector.classes == [1, 2, 3, 5, 7]
    assert detector.class_names[2] == "car"


def test_subset_alias_and_unknown_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chấp nhận model ít lớp, bỏ lớp lạ, báo lỗi khi không có lớp xe."""
    detector = detector_with_names(monkeypatch, {0: " MotorBike ", 1: "person"})
    assert detector.class_names == {0: "motorcycle"}
    with pytest.raises(ValueError, match="vehicle_model"):
        detector_with_names(monkeypatch, {0: "person", 1: "traffic light"})
