"""Kiểm thử polygon, bbox vi phạm bền vững và công cụ vẽ vùng."""

import csv
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import yaml
from ultralytics.engine.results import Boxes

from controllers.violation_controller import ViolationController
from models.vehicle_detector import VehicleDetector
from scripts import draw_polygon
from tests.test_multi_camera import FakeTraffic, FakeVehicle, bundle
from utils.config_loader import CameraConfig, load_cameras, validate_geometry
from utils.polygon import Polygon, box_in_polygon, contains_point, parse_polygon

REGION = ((10.0, 20.0), (100.0, 20.0), (100.0, 110.0), (10.0, 110.0))


def test_polygon_edges_concavity_and_bottom_center() -> None:
    """Nhận biên, polygon lõm và dùng chân bbox thay vì tâm bbox."""
    assert contains_point((10, 50), REGION)
    assert not contains_point((9, 50), REGION)
    assert contains_point((999, 999), ())
    assert box_in_polygon((20, 0, 40, 25), REGION)
    assert not box_in_polygon((20, 50, 40, 120), REGION)
    concave = parse_polygon([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]])
    assert contains_point((2, 8), concave)
    assert not contains_point((8, 8), concave)
    assert contains_point((2, 8), tuple(reversed(concave)))


@pytest.mark.parametrize(
    "points",
    [
        [[1, 1], [2, 2]],
        [[1, 1], [2, 2], [3, 3]],
        [[0, 0], [10, 10], [0, 10], [10, 0]],
        [[0, 0], [5, 5], [0, 0], [0, 5]],
        [[0, 0], [-1, 5], [3, 3]],
        [[0, 0], [True, 5], [3, 3]],
        [[0, 0], [float("nan"), 5], [3, 3]],
        "bad",
    ],
)
def test_invalid_polygon(points: object) -> None:
    """Loại hình suy biến, tự cắt hoặc tọa độ không hợp lệ."""
    with pytest.raises(ValueError):
        parse_polygon(points)


def test_polygon_config_and_bounds(camera: CameraConfig, tmp_path: Path) -> None:
    """Đọc vùng riêng từng camera và validate kích thước frame."""
    row = {
        "camera_id": "a",
        "source": camera.source,
        "stop_line": {"point1": [0, 60], "point2": [159, 60]},
        "traffic_light_roi": [0, 0, 10, 10],
        "detection_polygon": [list(p) for p in REGION],
    }
    path = tmp_path / "cameras.yaml"
    path.write_text(
        yaml.safe_dump(
            {"cameras": [row, {**row, "camera_id": "b", "detection_polygon": []}]}
        ),
        encoding="utf-8",
    )
    a, b = load_cameras(path)
    assert a.detection_polygon == REGION
    assert not b.detection_polygon
    validate_geometry(a, (160, 120))
    with pytest.raises(ValueError, match="Polygon"):
        validate_geometry(
            replace(a, detection_polygon=((0, 0), (160, 0), (5, 5))), (160, 120)
        )


def test_detector_filters_before_tracker() -> None:
    """YOLO boxes ngoài vùng không được đưa vào ByteTrack, kể cả frame rỗng."""
    detector = VehicleDetector.__new__(VehicleDetector)
    detector.lock = threading.RLock()
    detector.device, detector.imgsz, detector.classes = "cpu", 640, [2]
    boxes = Boxes(
        np.array(
            [[20, 30, 40, 50, 0.9, 2], [110, 30, 150, 50, 0.9, 2]], dtype=np.float32
        ),
        (120, 160),
    )
    result = SimpleNamespace(boxes=boxes, names={2: "car"})
    detector.model = SimpleNamespace(predict=lambda *args, **kwargs: [result])
    seen = []

    def update(filtered: Boxes, frame: np.ndarray) -> np.ndarray:
        """Ghi lại các detection thực sự tới tracker."""
        seen.append(filtered.xyxy.copy())
        return (
            np.array([[20, 30, 40, 50, 1, 0.9, 2, 0]], dtype=np.float32)
            if len(filtered)
            else np.empty((0, 8))
        )

    detector.trackers = {"a": SimpleNamespace(update=update)}
    tracks = detector.track(np.zeros((120, 160, 3), np.uint8), "a", 0.4, REGION)
    assert [t.track_id for t in tracks] == [1]
    assert len(seen[-1]) == 1
    assert (
        detector.track(
            np.zeros((120, 160, 3), np.uint8), "a", 0.4, ((0, 0), (5, 0), (5, 5))
        )
        == []
    )
    assert len(seen[-1]) == 0


class PolygonVehicle(FakeVehicle):
    """Fake vẫn trả mọi box để kiểm tra chốt lọc ở controller."""

    def track(
        self,
        frame: np.ndarray,
        camera_id: str,
        confidence: float,
        polygon: Polygon = (),
    ) -> list:
        """Giữ chuyển động fixture nhưng chấp nhận polygon của camera."""
        return super().track(frame, camera_id, confidence)


def test_red_box_persists_after_light_changes(
    camera: CameraConfig, settings: dict
) -> None:
    """Một sự kiện CSV; các frame sau vi phạm vẫn đỏ dù đèn đã xanh."""

    class ChangingLight(FakeTraffic):
        """Đỏ ở lúc cắt vạch, xanh ở các frame sau."""

        count = 0

        def detect(
            self,
            frame: np.ndarray,
            confidence: float,
            *,
            canvas_size: int = 0,
            fallback_roi: np.ndarray | None = None,
        ) -> str:
            """Chuyển màu sau frame thứ ba."""
            self.count += 1
            return "RED" if self.count <= 3 else "GREEN"

    models = bundle()
    models.vehicle, models.traffic = PolygonVehicle(), ChangingLight()
    frames = []
    config = replace(camera, detection_polygon=REGION)
    controller = ViolationController(
        config,
        settings,
        models,
        threading.Event(),
        frame_callback=lambda cid, f: frames.append(f.copy()),
    )
    assert controller.run()
    assert len(frames) == 6
    for index, bottom in enumerate([70, 80, 90, 100], start=2):
        assert tuple(frames[index][bottom, 35]) == (0, 0, 255)
    with (
        settings["output_base_dir"] / camera.camera_id / "violations.csv"
    ).open() as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_outside_polygon_no_boxes_or_events(
    camera: CameraConfig,
    settings: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Không vẽ hoặc ghi vi phạm cho track ngoài vùng dù fake model trả nó."""
    from views.display import Display

    seen = []
    original = Display.annotate

    def record(
        self: Display, frame: np.ndarray, tracks: list, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Ghi lại danh sách bbox mà View thực sự nhận."""
        seen.extend(tracks)
        return original(self, frame, tracks, *args, **kwargs)

    monkeypatch.setattr(Display, "annotate", record)
    models = bundle()
    models.vehicle = PolygonVehicle()
    config = replace(
        camera, detection_polygon=((110, 20), (150, 20), (150, 110), (110, 110))
    )
    assert ViolationController(config, settings, models, threading.Event()).run()
    assert seen == []
    with (
        settings["output_base_dir"] / camera.camera_id / "violations.csv"
    ).open() as handle:
        assert list(csv.DictReader(handle)) == []


def test_polygon_editor_saves_only_selected_camera(
    video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Click polygon và lưu; không đổi vạch/ROI/source của hai camera."""
    row = {
        "camera_id": "a",
        "source": str(video),
        "enabled": True,
        "stop_line": {"point1": [0, 60], "point2": [159, 60]},
        "traffic_light_roi": [0, 0, 10, 10],
    }
    other = {"camera_id": "b", "source": "${PRIVATE_RTSP}", "enabled": False}
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump({"cameras": [row, other]}), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["draw_polygon.py", "--camera_id", "a", "--cameras", str(path)]
    )
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("s"))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)

    def click(window: str, callback: Any) -> None:
        """Chọn bốn đỉnh theo chiều quanh vùng."""
        for x, y in REGION:
            callback(cv2.EVENT_LBUTTONDOWN, int(x), int(y), 0, None)

    monkeypatch.setattr(cv2, "setMouseCallback", click)
    assert draw_polygon.main() == 0
    result = yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"]
    assert result[0].pop("detection_polygon") == [list(p) for p in REGION]
    assert result == [row, other]
