"""Vẽ kết quả và ghi video, ảnh, CSV; không quyết định vi phạm."""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from models.types import Detection
from utils.config_loader import CameraConfig

COLUMNS = [
    "id",
    "camera_id",
    "track_id",
    "timestamp",
    "plate_text",
    "plate_confidence",
    "vehicle_class",
    "evidence_path",
]


def write_image(path: Path, image: np.ndarray) -> None:
    """Ghi JPEG hỗ trợ đường dẫn Unicode trên Windows, kiểm tra lỗi encoder."""
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise OSError(f"Không encode được ảnh {path.name}")
    temporary = path.with_suffix(".jpg.part")
    encoded.tofile(str(temporary))
    temporary.replace(path)


class Display:
    """Một writer và một khóa CSV riêng cho mỗi camera."""

    def __init__(self, camera: CameraConfig, base_dir: Path, run_id: str) -> None:
        self.camera = camera
        self.run_id = run_id
        self.directory = base_dir / camera.camera_id
        self.videos = self.directory / "videos"
        self.evidence = self.directory / "evidence"
        self.videos.mkdir(parents=True, exist_ok=True)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.csv_lock = threading.Lock()
        self.csv_path = self.directory / "violations.csv"
        with self.csv_lock:
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow(COLUMNS)
        self.writer: cv2.VideoWriter | None = None
        self._signature: tuple[tuple[int, int], float, int] | None = None
        self._part = 0

    def ensure_video(self, size: tuple[int, int], fps: float, epoch: int) -> None:
        """Mở file MP4 mới khi thay đổi kích thước, FPS hoặc phiên kết nối."""
        signature = (size, fps, epoch)
        if signature == self._signature:
            return
        self.close()
        self._part += 1
        filename = self.videos / f"{self.run_id}_{self._part:03d}.mp4"
        self.writer = cv2.VideoWriter(
            str(filename), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        if not self.writer.isOpened():
            raise OSError("Không tạo được MP4; kiểm tra codec, quyền ghi và ổ đĩa")
        self._signature = signature

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[Detection],
        light: str,
        timestamp: datetime,
        source_seconds: float | None,
        highlight: set[int] | None = None,
    ) -> np.ndarray:
        """Vẽ vạch, ROI, bbox, đèn và thời gian lên bản sao của frame."""
        canvas = frame.copy()
        marked = highlight or set()
        if self.camera.detection_polygon:
            polygon = np.array(self.camera.detection_polygon, dtype=np.int32)
            cv2.polylines(canvas, [polygon], True, (255, 160, 0), 2)
        p1 = tuple(int(v) for v in self.camera.point1)
        p2 = tuple(int(v) for v in self.camera.point2)
        cv2.line(canvas, p1, p2, (0, 255, 255), 2)
        x, y, w, h = self.camera.roi
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = (0, 0, 255) if track.track_id in marked else (0, 200, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.circle(canvas, ((x1 + x2) // 2, y2), 4, color, -1)
            cv2.putText(
                canvas,
                f"{track.class_name} #{track.track_id}"
                + (" VI PHAM" if track.track_id in marked else ""),
                (max(0, x1), max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        lines = [
            f"{self.camera.camera_id} | LIGHT: {light}",
            timestamp.isoformat(timespec="milliseconds"),
        ]
        if source_seconds is not None:
            lines.append(f"Video time: {source_seconds:.3f}s")
        for index, text in enumerate(lines):
            position = (10, 22 + index * 22)
            cv2.putText(
                canvas,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def write_frame(self, frame: np.ndarray) -> None:
        """Ghi một frame đã được controller chọn và annotate."""
        if self.writer is None:
            raise RuntimeError("Chưa mở VideoWriter")
        self.writer.write(frame)

    def save_evidence(
        self,
        event_id: str,
        track_id: int,
        timestamp: datetime,
        plate_text: str,
        plate_confidence: float,
        vehicle_class: str,
        full_frame: np.ndarray,
        vehicle_crop: np.ndarray,
        plate_crop: np.ndarray | None,
    ) -> None:
        """Ghi ảnh trước, chỉ append CSV khi các ảnh đã ghi thành công."""
        full_path = self.evidence / f"{event_id}_frame.jpg"
        write_image(full_path, full_frame)
        if vehicle_crop.size:
            write_image(self.evidence / f"{event_id}_vehicle.jpg", vehicle_crop)
        if plate_crop is not None and plate_crop.size:
            write_image(self.evidence / f"{event_id}_plate.jpg", plate_crop)
        with self.csv_lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        event_id,
                        self.camera.camera_id,
                        track_id,
                        timestamp.isoformat(timespec="milliseconds"),
                        plate_text,
                        f"{plate_confidence:.4f}",
                        vehicle_class,
                        full_path.relative_to(self.directory).as_posix(),
                    ]
                )

    def close(self) -> None:
        """Hoàn tất container MP4 và giải phóng writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
