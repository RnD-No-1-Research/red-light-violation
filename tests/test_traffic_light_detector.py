"""Kiểm tra tiền xử lý ROI và UNKNOWN, không cần tải weights."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from controllers.violation_controller import ViolationController
from models.traffic_light_detector import TrafficLightDetector
from tests.test_multi_camera import bundle
from utils.config_loader import CameraConfig


@pytest.mark.parametrize("canvas_size", [0, 192, 20])
def test_padding_preserves_pixels_and_unknown(canvas_size: int) -> None:
    """Đệm không kéo giãn/cắt ảnh; không có detection vẫn phải là UNKNOWN."""
    detector = TrafficLightDetector.__new__(TrafficLightDetector)
    detector.lock = threading.RLock()
    detector.device, detector.imgsz, detector.labels = "cpu", 640, {}
    roi = np.arange(59 * 29 * 3, dtype=np.uint8).reshape(59, 29, 3)
    before = roi.copy()
    seen = []

    def predict(image: np.ndarray, **kwargs: object) -> list:
        """Thu ảnh đến model và trả detection rỗng."""
        seen.append((image.copy(), kwargs))
        return [SimpleNamespace(boxes=[])]

    detector.model = SimpleNamespace(predict=predict)
    assert detector.detect(roi, 0.25, canvas_size=canvas_size) == "UNKNOWN"
    image, options = seen[-1]
    if canvas_size:
        side = max(canvas_size, 59)
        assert image.shape == (side, side, 3)
        x, y = (side - 29) // 2, (side - 59) // 2
        np.testing.assert_array_equal(image[y : y + 59, x : x + 29], before)
        image[y : y + 59, x : x + 29] = 0
        assert not image.any()
    else:
        np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(roi, before)
    assert options["conf"] == 0.25
    assert options["imgsz"] == (320 if canvas_size == 192 else 160)
    assert detector.detect(roi[:0], 0.25, canvas_size=canvas_size) == "UNKNOWN"
    assert len(seen) == (2 if canvas_size else 1)


def test_controller_passes_camera_canvas(camera: CameraConfig, settings: dict) -> None:
    """Từng camera chuyển đúng tùy chọn ROI tới model dùng chung."""
    models = bundle()
    seen = []

    def detect(
        frame: np.ndarray,
        confidence: float,
        *,
        canvas_size: int = 0,
        fallback_roi: np.ndarray | None = None,
    ) -> str:
        """Ghi hình dạng ROI gốc và tùy chọn camera."""
        seen.append((frame.shape, canvas_size, fallback_roi.shape))
        return "UNKNOWN"

    models.traffic.detect = detect
    controller = ViolationController(
        replace(
            camera,
            traffic_light_canvas_size=192,
            traffic_light_fallback_roi=(1, 2, 10, 12),
        ),
        settings,
        models,
        threading.Event(),
    )
    assert controller.run()
    assert seen == [((20, 20, 3), 192, (12, 10, 3))] * 6


@pytest.mark.parametrize("primary", ["RED", "GREEN", "YELLOW", "UNKNOWN"])
def test_fallback_only_when_primary_unknown(primary: str) -> None:
    """ROI dự phòng không thay màu hợp lệ; cả hai lần dùng cùng ngưỡng."""
    detector = TrafficLightDetector.__new__(TrafficLightDetector)
    detector.lock = threading.RLock()
    seen = []

    def predict(roi: np.ndarray, confidence: float) -> str:
        """Hai ROI có pixel khác nhau để phát hiện lấy nhầm crop."""
        seen.append((roi.copy(), confidence))
        return primary if len(seen) == 1 else "GREEN"

    detector._predict_color = predict
    roi = np.ones((88, 19, 3), np.uint8)
    fallback = np.full((59, 29, 3), 42, np.uint8)
    color = detector.detect(roi, 0.25, canvas_size=192, fallback_roi=fallback)
    assert color == ("GREEN" if primary == "UNKNOWN" else primary)
    assert len(seen) == (2 if primary == "UNKNOWN" else 1)
    assert all(conf == 0.25 for _, conf in seen)
    if primary == "UNKNOWN":
        padded = seen[-1][0]
        np.testing.assert_array_equal(padded[66:125, 81:110], fallback)
