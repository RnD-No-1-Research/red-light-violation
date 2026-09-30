"""Validate YAML độc lập với video và AI."""

from pathlib import Path

import pytest
import yaml
from dataclasses import replace

from utils.config_loader import ROOT, load_cameras, validate_geometry


def _row() -> dict:
    return {
        "camera_id": "cam_01",
        "source": "data/videos/a.mp4",
        "enabled": True,
        "stop_line": {"point1": [1, 10], "point2": [90, 10]},
        "traffic_light_roi": [1, 1, 10, 10],
    }


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump({"cameras": rows}), encoding="utf-8")
    return path


def test_examples_and_disabled(tmp_path: Path) -> None:
    """Cấu hình mẫu có hai camera; camera tắt được giữ placeholder."""
    assert len(load_cameras(ROOT / "config/cameras.yaml")) == 2
    rows = [_row(), {"camera_id": "off", "enabled": False}]
    cameras = load_cameras(_write(tmp_path, rows), root=tmp_path)
    assert len(cameras) == 1
    assert cameras[0].source == str((tmp_path / "data/videos/a.mp4").resolve())


@pytest.mark.parametrize(
    "change",
    [
        {"camera_id": "../escape"},
        {"enabled": "false"},
        {"source": ""},
        {"traffic_light_roi": [0, 0, 0, 10]},
        {"traffic_light_roi": [0, 0, 10.5, 10]},
        {"traffic_light_canvas_size": -1},
        {"traffic_light_canvas_size": True},
        {"traffic_light_canvas_size": 1.5},
        {"traffic_light_canvas_size": 4097},
        {"traffic_light_fallback_roi": [0, 0, -1, 20]},
        {"traffic_light_fallback_roi": [0, 0, 10]},
        {"traffic_light_fallback_roi": [0, 0, True, 20]},
        {"stop_line": {"point1": [0, 0], "point2": [0, 0]}},
        {"crossing_direction": "sideways"},
        {"vehicle_conf": 2},
    ],
)
def test_invalid(tmp_path: Path, change: dict) -> None:
    """Báo lỗi sớm cho thông số không hợp lệ."""
    with pytest.raises(ValueError):
        load_cameras(_write(tmp_path, [{**_row(), **change}]))


def test_duplicate_and_bounds(tmp_path: Path) -> None:
    """Ngăn ghi chung output do trùng ID và tọa độ ngoài frame."""
    with pytest.raises(ValueError, match="Trùng"):
        load_cameras(_write(tmp_path, [_row(), _row()]))
    camera = load_cameras(_write(tmp_path, [_row()]))[0]
    with pytest.raises(ValueError):
        validate_geometry(camera, (20, 20))


def test_environment_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Đọc RTSP từ biến môi trường, không cần đưa mật khẩu vào YAML."""
    monkeypatch.setenv("TEST_RTSP", "rtsp://example.invalid/stream")
    row = {**_row(), "source": "${TEST_RTSP}"}
    assert load_cameras(_write(tmp_path, [row]))[0].source.startswith("rtsp://")


def test_malformed_yaml(tmp_path: Path) -> None:
    """YAML sai cú pháp được đổi thành lỗi cấu hình dễ đọc."""
    path = tmp_path / "broken.yaml"
    path.write_text("cameras: [", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        load_cameras(path)


def test_light_fallback_config_and_bounds(tmp_path: Path) -> None:
    """Đọc ROI dự phòng từng camera và chặn tọa độ ngoài frame."""
    row = {
        **_row(),
        "traffic_light_fallback_roi": [5, 6, 7, 8],
        "traffic_light_canvas_size": 192,
    }
    camera = load_cameras(_write(tmp_path, [row]))[0]
    assert camera.traffic_light_fallback_roi == (5, 6, 7, 8)
    assert camera.traffic_light_canvas_size == 192
    validate_geometry(camera, (100, 100))
    with pytest.raises(ValueError, match="dự phòng"):
        validate_geometry(
            replace(camera, traffic_light_fallback_roi=(99, 99, 7, 8)), (100, 100)
        )
    default = load_cameras(_write(tmp_path, [_row()]))[0]
    assert default.traffic_light_canvas_size == 0
    assert default.traffic_light_fallback_roi is None
