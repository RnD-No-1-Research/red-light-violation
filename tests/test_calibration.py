"""Kiểm tra công cụ hiệu chỉnh chỉ sửa đúng mục camera được chọn."""

import sys
from pathlib import Path

import cv2
import pytest
import yaml

from scripts import draw_stop_line


def test_save_only_selected_camera(
    video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mô phỏng hai click và S; giữ nguyên source và giá trị camera khác."""
    selected = {
        "camera_id": "selected",
        "source": str(video),
        "enabled": False,
        "stop_line": {"point1": [0, 0], "point2": [0, 0]},
        "traffic_light_roi": [1, 1, 10, 10],
    }
    other = {"camera_id": "other", "source": "${CAM_RTSP_URL}", "enabled": False}
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump({"cameras": [selected, other]}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["draw_stop_line.py", "--camera_id", "selected", "--cameras", str(path)],
    )
    for name in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("s"))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)

    def click_twice(window: str, callback: object) -> None:
        """Mô phỏng điểm đầu/cuối trong biên video."""
        callback(cv2.EVENT_LBUTTONDOWN, 10, 60, 0, None)
        callback(cv2.EVENT_LBUTTONDOWN, 140, 60, 0, None)

    monkeypatch.setattr(cv2, "setMouseCallback", click_twice)
    assert draw_stop_line.main() == 0
    result = yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"]
    assert result[0]["stop_line"] == {"point1": [10, 60], "point2": [140, 60]}
    assert result[0]["enabled"] is False
    assert result[0]["source"] == str(video)
    assert result[1] == other
