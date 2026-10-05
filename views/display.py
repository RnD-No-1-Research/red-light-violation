"""Vẽ kết quả và ghi video, ảnh, CSV; không quyết định vi phạm."""

from __future__ import annotations

import csv
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from models.types import Detection
from utils.config_loader import CameraConfig
from utils.occlusion_review import ReviewEvent
from utils.late_detection import LateSuspicion
from utils.video_source import FramePacket
from utils.vehicle_anchor import vehicle_anchor

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
        self.started_at = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.log = logging.LoggerAdapter(
            logging.getLogger(__name__), {"camera_id": camera.camera_id}
        )
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
        self._video_path: Path | None = None
        self._frames_written = 0

    def ensure_video(self, size: tuple[int, int], fps: float, epoch: int) -> None:
        """Mở file MP4 mới khi thay đổi kích thước, FPS hoặc phiên kết nối."""
        signature = (size, fps, epoch)
        if signature == self._signature:
            return
        self.close()
        self._part += 1
        filename = self.videos / (
            f"{self.camera.camera_id}_{self.started_at}_"
            f"{self.run_id}_{self._part:03d}.mp4"
        )
        self.writer = cv2.VideoWriter(
            str(filename), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        if not self.writer.isOpened():
            raise OSError("Không tạo được MP4; kiểm tra codec, quyền ghi và ổ đĩa")
        self._signature = signature
        self._video_path = filename
        self._frames_written = 0
        self.log.info("Đang ghi video kết quả: %s", filename.resolve())

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[Detection],
        light: str,
        timestamp: datetime,
        source_seconds: float | None,
        highlight: set[int] | None = None,
        suspected: set[int] | None = None,
        direction_unknown: set[int] | None = None,
    ) -> np.ndarray:
        """Vẽ vạch, ROI, bbox, đèn và thời gian lên bản sao của frame."""
        canvas = frame.copy()
        marked = highlight or set()
        suspected = suspected or set()
        direction_unknown = direction_unknown or set()
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
            suffix = " VI PHAM" if track.track_id in marked else ""
            if track.track_id in suspected and track.track_id not in marked:
                color = (0, 165, 255)
                suffix = " NGHI VAN VUOT DEN DO"
            if track.track_id in direction_unknown and track.track_id not in marked:
                color = (0, 165, 255)
                suffix = " CHUA RO HUONG"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.circle(canvas, tuple(map(int, vehicle_anchor(track.bbox))), 4, color, -1)
            cv2.putText(
                canvas,
                f"{track.class_name} #{track.track_id}"
                + suffix,
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
        self._frames_written += 1

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
        *, decision: dict | None = None,
    ) -> None:
        """Ghi ảnh trước, chỉ append CSV khi các ảnh đã ghi thành công."""
        full_path = self.evidence / f"{event_id}_frame.jpg"
        write_image(full_path, full_frame)
        if vehicle_crop.size:
            write_image(self.evidence / f"{event_id}_vehicle.jpg", vehicle_crop)
        if plate_crop is not None and plate_crop.size:
            write_image(self.evidence / f"{event_id}_plate.jpg", plate_crop)
        if decision is not None:
            metadata = self.evidence / f"{event_id}_decision.json"
            metadata.write_text(json.dumps({
                "id": event_id, "camera_id": self.camera.camera_id,
                "track_id": track_id, "timestamp": timestamp.isoformat(),
                **decision,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
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
            if self._video_path is not None:
                self.log.info(
                    "Đã lưu video kết quả: %s (%d frame)",
                    self._video_path.resolve(),
                    self._frames_written,
                )
        self._video_path = None
        self._signature = None

    def save_late_review(
        self, event_id: str, event: LateSuspicion, packet: FramePacket,
        annotated: np.ndarray,
    ) -> None:
        """Save current-frame uncertainty evidence, without OCR or a violation row."""
        directory = self.directory / "review_evidence"
        directory.mkdir(exist_ok=True)
        image_path = directory / f"{event_id}_direction.jpg"
        write_image(image_path, annotated)
        (directory / f"{event_id}_decision.json").write_text(
            json.dumps({"camera_id": self.camera.camera_id,
                        "track_id": event.vehicle.track_id,
                        "source_seconds": packet.source_seconds,
                        "sequence": packet.sequence, **event.details}, indent=2),
            encoding="utf-8",
        )
        path = self.directory / "late_review_events.csv"
        with self.csv_lock:
            header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if header:
                    writer.writerow(["id", "camera_id", "track_id", "status",
                                     "timestamp", "source_seconds", "vehicle_class", "evidence_path"])
                writer.writerow([event_id, self.camera.camera_id, event.vehicle.track_id,
                                 event.details["reason"], packet.timestamp.isoformat(),
                                 packet.source_seconds, event.vehicle.class_name,
                                 image_path.relative_to(self.directory).as_posix()])

    def save_review(
        self, event_id: str, event: ReviewEvent, packet: FramePacket,
        before_image: np.ndarray, after_image: np.ndarray,
    ) -> None:
        """Separate review CSV; no OCR and no confirmed-violation row."""
        directory = self.directory / "review_evidence"
        directory.mkdir(exist_ok=True)
        before_path = directory / f"{event_id}_before.jpg"
        after_path = directory / f"{event_id}_after.jpg"
        write_image(before_path, before_image)
        write_image(after_path, after_image)
        path = self.directory / "review_events.csv"
        with self.csv_lock:
            header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if header:
                    writer.writerow([
                        "id", "camera_id", "track_id", "status", "timestamp",
                        "vehicle_class", "before_seconds", "after_seconds",
                        "gap_seconds", "before_path", "after_path",
                    ])
                writer.writerow([
                    event_id, self.camera.camera_id, event.after.track_id,
                    "SUSPECTED_OCCLUDED_CROSSING", packet.timestamp.isoformat(),
                    event.after.class_name, event.before.packet.source_seconds,
                    packet.source_seconds, f"{event.gap_seconds:.6f}",
                    before_path.relative_to(self.directory).as_posix(),
                    after_path.relative_to(self.directory).as_posix(),
                ])
