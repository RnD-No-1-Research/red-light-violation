"""Kiểm thử pipeline với video thật, model giả và output thật."""

import csv
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from controllers.camera_manager import CameraManager
from models.types import Detection, OCRResult
from utils.config_loader import CameraConfig
from utils.video_source import FramePacket


class FakeVehicle:
    """Model giả cho xe đi qua vạch ở frame thứ ba của mỗi camera."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def reset_camera(self, camera_id: str, confidence: float) -> None:
        """Khởi tạo lịch sử độc lập cho camera."""
        with self.lock:
            self.counts[camera_id] = 0

    def track(
        self, frame: np.ndarray, camera_id: str, confidence: float
    ) -> list[Detection]:
        """Cả hai camera cố ý dùng cùng ID để kiểm tra tách nghiệp vụ."""
        with self.lock:
            index = self.counts[camera_id]
            self.counts[camera_id] += 1
        bottom = [40, 50, 70, 80, 90, 100][index]
        return [Detection((30, bottom - 20, 70, bottom), 0.9, "car", 1)]

    def drop_camera(self, camera_id: str) -> None:
        """Giữ số frame để test có thể kiểm tra sau khi chạy."""
        return None


class FakeTraffic:
    """Model giả trả màu do test lựa chọn."""

    def __init__(self, color: str = "RED") -> None:
        self.color = color

    def detect(
        self,
        frame: np.ndarray,
        confidence: float,
        *,
        canvas_size: int = 0,
        fallback_roi: np.ndarray | None = None,
    ) -> str:
        """Trả màu cố định để test điều kiện nghiệp vụ."""
        return self.color


class FakePlate:
    """Model giả trả bbox biển nằm trong crop xe."""

    def detect(self, frame: np.ndarray, confidence: float) -> Detection:
        """Trả vùng biển hợp lệ để ghi crop."""
        return Detection((1, 1, 20, 10), 0.9, "plate")


class FakeOCR:
    """OCR giả có thể trả văn bản hợp lệ hoặc phát sinh lỗi."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def read(self, frame: np.ndarray) -> OCRResult:
        """Mô phỏng hai dòng biển số hoặc lỗi inference."""
        if self.fail:
            raise RuntimeError("OCR failure")
        return OCRResult("30E1 123.45", 0.9)


def bundle(color: str = "RED", ocr_fail: bool = False) -> SimpleNamespace:
    """Tạo bundle giả có cùng interface với model thật."""
    return SimpleNamespace(
        vehicle=FakeVehicle(),
        traffic=FakeTraffic(color),
        plate=FakePlate(),
        ocr=FakeOCR(ocr_fail),
    )


def _rows(directory: Path) -> list[dict]:
    with (directory / "violations.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_two_cameras_isolated(camera: CameraConfig, settings: dict) -> None:
    """Hai camera xuất riêng video, bằng chứng và CSV dù trùng track ID."""
    cameras = [camera, replace(camera, camera_id="cam_02")]
    models = bundle()
    assert CameraManager(settings, cameras, models).run() == 0
    for config in cameras:
        folder = settings["output_base_dir"] / config.camera_id
        rows = _rows(folder)
        assert len(rows) == 1
        assert rows[0]["camera_id"] == config.camera_id
        assert rows[0]["plate_text"] == "30E1-12345"
        assert (folder / rows[0]["evidence_path"]).is_file()
        assert len(list((folder / "evidence").glob("*.jpg"))) == 3
        cap = cv2.VideoCapture(str(next((folder / "videos").glob("*.mp4"))))
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
        cap.release()


def test_bad_camera_does_not_stop_good(camera: CameraConfig, settings: dict) -> None:
    """Lỗi nguồn ở camera B không ngăn camera A xử lý đủ video."""
    bad = replace(
        camera,
        camera_id="bad",
        source=str(Path(camera.source).with_name("missing.mp4")),
    )
    models = bundle()
    manager = CameraManager(settings, [camera, bad], models)
    assert manager.run() == 1
    assert manager.failures == ["bad"]
    assert models.vehicle.counts[camera.camera_id] == 6
    assert len(_rows(settings["output_base_dir"] / camera.camera_id)) == 1


@pytest.mark.parametrize("color", ["GREEN", "YELLOW", "UNKNOWN"])
def test_non_red_no_evidence(camera: CameraConfig, settings: dict, color: str) -> None:
    """Camera vẫn ghi video/CSV header khi không có vi phạm."""
    assert CameraManager(settings, [camera], bundle(color)).run() == 0
    assert _rows(settings["output_base_dir"] / camera.camera_id) == []


def test_ocr_failure_still_saves(camera: CameraConfig, settings: dict) -> None:
    """Lỗi OCR không làm mất vi phạm và các ảnh đã crop."""
    assert CameraManager(settings, [camera], bundle(ocr_fail=True)).run() == 0
    rows = _rows(settings["output_base_dir"] / camera.camera_id)
    assert rows[0]["plate_text"] == "UNKNOWN"
    assert rows[0]["plate_confidence"] == "0.0000"


def test_reconnect_does_not_bridge_crossing(
    camera: CameraConfig,
    settings: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Xe trước vạch và sau vạch thuộc hai phiên không được tính là cắt vạch."""

    class ReconnectedSource:
        """Nguồn giả trả hai frame ở hai phiên kết nối."""

        is_rtsp = True
        error = None

        def __init__(self, *args: object) -> None:
            self.index = 0

        def read_frame(self) -> FramePacket | None:
            """Mã hóa vị trí xe trong pixel để fake model đọc."""
            if self.index == 2:
                return None
            index = self.index
            self.index += 1
            frame = np.full((120, 160, 3), index, dtype=np.uint8)
            return FramePacket(frame, datetime.now(timezone.utc), index + 1, index)

        def get_fps(self) -> float:
            """FPS output cố định cho test."""
            return 10.0

        def release(self) -> None:
            """Không có capture native trong fixture này."""
            return None

    class PositionVehicle(FakeVehicle):
        """Trả cùng ID qua hai phiên, vị trí trước/sau vạch."""

        def track(
            self, frame: np.ndarray, camera_id: str, confidence: float
        ) -> list[Detection]:
            """Đọc vị trí từ frame, không phụ thuộc reset bộ đếm."""
            bottom = 40 if frame[0, 0, 0] == 0 else 80
            return [Detection((30, bottom - 20, 70, bottom), 0.9, "car", 1)]

    monkeypatch.setattr(
        "controllers.violation_controller.VideoSource", ReconnectedSource
    )
    models = bundle()
    models.vehicle = PositionVehicle()
    assert CameraManager(settings, [camera], models).run() == 0
    folder = settings["output_base_dir"] / camera.camera_id
    assert _rows(folder) == []
    assert len(list((folder / "videos").glob("*.mp4"))) == 2
