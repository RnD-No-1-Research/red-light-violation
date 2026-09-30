"""Fixture cấu hình và video tổng hợp, không cần GPU hoặc weights."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.config_loader import ROOT, CameraConfig, load_settings

# Chỉ dùng trong kiểm thử tracker thật; không tải model.
_config = ROOT / "output/.test_ultralytics"
_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_config))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


@pytest.fixture
def settings(tmp_path: Path) -> dict:
    """Trả settings có output tạm và cửa sổ một frame để test nghiệp vụ."""
    result = load_settings(ROOT / "config/settings.yaml")
    result.update(output_base_dir=tmp_path / "output", light_smoothing_window=1, show_video=False)
    return result


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """Sinh video nhỏ có sáu frame, không chứa nội dung giao thông thật."""
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((120, 160, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def camera(video: Path) -> CameraConfig:
    """Trả camera với vạch ngang và ROI nằm trong video fixture."""
    return CameraConfig(
        "cam_01", "Test", str(video), (0, 60), (159, 60), (0, 0, 20, 20)
    )
