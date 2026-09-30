"""Chọn vạch và ROI bằng chuột, lưu đúng camera trong YAML."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import yaml  # noqa: E402

from main import configure_logging  # noqa: E402
from utils.config_loader import (  # noqa: E402
    CameraConfig,
    read_yaml,
    resolve_path,
    validate_geometry,
)
from utils.video_source import VideoSource  # noqa: E402


def main() -> int:
    """Mở camera đã chọn, click hai điểm; R chọn ROI, D đổi chiều, S lưu."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera_id", required=True)
    parser.add_argument("--cameras", type=Path, default=ROOT / "config/cameras.yaml")
    args = parser.parse_args()
    configure_logging()
    document = read_yaml(args.cameras)
    matches = [
        r for r in document.get("cameras", []) if r.get("camera_id") == args.camera_id
    ]
    if len(matches) != 1:
        parser.error("camera_id phải tồn tại duy nhất trong YAML")
    row = matches[0]
    raw_source = os.path.expandvars(str(row.get("source", "")))
    if not raw_source or "${" in raw_source:
        parser.error("Chưa cấu hình source hoặc biến môi trường của camera")
    source_path = (
        raw_source
        if raw_source.lower().startswith(("rtsp://", "rtsps://"))
        else str(resolve_path(raw_source))
    )
    source = VideoSource(source_path, args.camera_id)
    window = f"Calibrate {args.camera_id}"
    points: list[tuple[int, int]] = []
    roi = tuple(row.get("traffic_light_roi", [0, 0, 0, 0]))
    direction = row.get("crossing_direction", "negative_to_positive")

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Thu hai điểm vạch; click tiếp theo bắt đầu chọn lại."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) == 2:
                points.clear()
            points.append((x, y))

    try:
        packet = source.read_frame()
        if packet is None:
            raise OSError("Không đọc được frame đầu của camera")
        frame = packet.image
        source.release()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        print(
            f"[{args.camera_id}] Click 2 points; R=ROI, D=direction, S=save, Esc=cancel"
        )
        while True:
            preview = frame.copy()
            for point in points:
                cv2.circle(preview, point, 5, (0, 255, 255), -1)
            if len(points) == 2:
                cv2.line(preview, points[0], points[1], (0, 255, 255), 2)
                dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
                length = max(1.0, (dx * dx + dy * dy) ** 0.5)
                sign = 1 if direction == "negative_to_positive" else -1
                mid = (
                    (points[0][0] + points[1][0]) // 2,
                    (points[0][1] + points[1][1]) // 2,
                )
                end = (
                    int(mid[0] - sign * dy / length * 60),
                    int(mid[1] + sign * dx / length * 60),
                )
                cv2.arrowedLine(preview, mid, end, (0, 0, 255), 2)
            x, y, w, h = roi
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(
                preview,
                "Click 2 points | R: ROI | D: direction | S: save | ESC",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return 0
            if key == ord("d"):
                direction = (
                    "positive_to_negative"
                    if direction == "negative_to_positive"
                    else "negative_to_positive"
                )
            if key == ord("r"):
                selected = cv2.selectROI(
                    "Select traffic light ROI",
                    frame,
                    showCrosshair=True,
                    fromCenter=False,
                )
                cv2.destroyWindow("Select traffic light ROI")
                if selected[2] > 0 and selected[3] > 0:
                    roi = tuple(int(v) for v in selected)
            if key == ord("s"):
                if len(points) != 2 or points[0] == points[1] or min(roi[2:]) <= 0:
                    print(
                        f"[{args.camera_id}] Select 2 distinct points and a valid ROI"
                    )
                    continue
                camera = CameraConfig(
                    args.camera_id,
                    str(row.get("name", "")),
                    source_path,
                    points[0],
                    points[1],
                    roi,
                    direction,
                )
                try:
                    validate_geometry(camera, (frame.shape[1], frame.shape[0]))
                except ValueError as exc:
                    print(f"[{args.camera_id}] {exc}")
                    continue
                row["stop_line"] = {
                    "point1": list(points[0]),
                    "point2": list(points[1]),
                }
                row["traffic_light_roi"] = list(roi)
                row["crossing_direction"] = direction
                temporary = args.cameras.with_suffix(".yaml.tmp")
                temporary.write_text(
                    yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                temporary.replace(args.cameras)
                print(f"[{args.camera_id}] Saved {args.cameras.name}")
                return 0
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
