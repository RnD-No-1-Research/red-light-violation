"""Vẽ vùng phát hiện xe dạng polygon riêng cho từng camera."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from main import configure_logging  # noqa: E402
from utils.config_loader import read_yaml, resolve_path  # noqa: E402
from utils.polygon import parse_polygon, validate_polygon_bounds  # noqa: E402
from utils.video_source import VideoSource  # noqa: E402


def main() -> int:
    """Click các đỉnh; S/Enter lưu, R vẽ lại, Backspace bỏ điểm, Esc hủy."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera_id", required=True)
    parser.add_argument("--cameras", type=Path, default=ROOT / "config/cameras.yaml")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Xóa giới hạn polygon của camera, trở về toàn frame",
    )
    args = parser.parse_args()
    document = read_yaml(args.cameras)
    rows = document.get("cameras", [])
    matches = [row for row in rows if row.get("camera_id") == args.camera_id]
    if len(matches) != 1:
        parser.error("camera_id phải tồn tại duy nhất trong YAML")

    def save(points: list[list[int]]) -> None:
        """Đọc lại YAML và cập nhật riêng polygon, giữ cấu hình camera khác."""
        latest = read_yaml(args.cameras)
        selected = [
            row
            for row in latest.get("cameras", [])
            if row.get("camera_id") == args.camera_id
        ]
        if len(selected) != 1:
            raise ValueError("Camera đã bị thay đổi trong lúc vẽ; chưa lưu")
        selected[0]["detection_polygon"] = points
        temporary = args.cameras.with_suffix(".yaml.tmp")
        temporary.write_text(
            yaml.safe_dump(latest, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(args.cameras)
        print(f"[{args.camera_id}] Đã lưu polygon vào {args.cameras.name}")

    if args.clear:
        save([])
        return 0
    row = matches[0]
    raw = os.path.expandvars(str(row.get("source", "")))
    if not raw or "${" in raw:
        parser.error("Chưa cấu hình source hoặc biến môi trường camera")
    source_path = (
        raw
        if raw.lower().startswith(("rtsp://", "rtsps://"))
        else str(resolve_path(raw))
    )
    source = VideoSource(source_path, args.camera_id)
    window = f"Polygon - {args.camera_id}"
    try:
        packet = source.read_frame()
        if packet is None:
            raise OSError("Không đọc được frame đầu camera")
        frame = packet.image
        source.release()
        height, width = frame.shape[:2]
        scale = min(1.0, 1200 / width, 800 / height)
        display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        sx, sy = width / display_size[0], height / display_size[1]
        try:
            polygon = parse_polygon(row.get("detection_polygon"))
            validate_polygon_bounds(polygon, (width, height))
            points = [(round(x), round(y)) for x, y in polygon]
        except ValueError:
            points = []

        def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
            """Quy đổi tọa độ cửa sổ về pixel frame gốc trước khi lưu."""
            if event == cv2.EVENT_LBUTTONDOWN:
                point = (
                    min(width - 1, max(0, round(x * sx))),
                    min(height - 1, max(0, round(y * sy))),
                )
                if not points or point != points[-1]:
                    points.append(point)
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, *display_size)
        cv2.setMouseCallback(window, on_mouse)
        print(f"[{args.camera_id}] Click >=3 đỉnh theo thứ tự quanh vùng cần xét.")
        print("S/Enter: lưu | R: vẽ lại | Backspace/chuột phải: bỏ điểm | Esc: hủy")
        print("Chọn vùng phủ cả trước và sau vạch để theo dõi xe đã vi phạm.")
        while True:
            canvas = frame.copy()
            if len(points) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], (255, 160, 0))
                canvas = cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0)
            if len(points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.array(points, dtype=np.int32)],
                    len(points) >= 3,
                    (255, 160, 0),
                    3,
                )
            for index, point in enumerate(points):
                cv2.circle(canvas, point, 6, (0, 255, 255), -1)
                cv2.putText(
                    canvas,
                    str(index + 1),
                    point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            line = row.get("stop_line", {})
            if isinstance(line, dict) and all(k in line for k in ("point1", "point2")):
                cv2.line(
                    canvas,
                    tuple(map(int, line["point1"])),
                    tuple(map(int, line["point2"])),
                    (0, 255, 255),
                    2,
                )
            preview = cv2.resize(canvas, display_size)
            cv2.rectangle(preview, (0, 0), (display_size[0], 52), (0, 0, 0), -1)
            cv2.putText(
                preview,
                "Click vertices | S/Enter: save | R: reset | Esc: cancel",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                preview,
                "Right-click/Backspace: undo | Include BOTH sides of stop line",
                (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return 0
            if key in (ord("r"), ord("R")):
                points.clear()
            elif key in (8, 127) and points:
                points.pop()
            elif key in (ord("s"), ord("S"), 10, 13):
                try:
                    if len(points) < 3:
                        raise ValueError("Cần ít nhất 3 đỉnh trước khi lưu")
                    polygon = parse_polygon(points)
                    validate_polygon_bounds(polygon, (width, height))
                except ValueError as exc:
                    print(f"[{args.camera_id}] {exc}")
                    continue
                save([[round(x), round(y)] for x, y in polygon])
                return 0
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
