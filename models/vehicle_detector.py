"""YOLO11 COCO dùng chung kết hợp ByteTrack riêng từng camera."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from models._yolo import LockedYOLO
from models.types import Detection
from utils.polygon import Polygon, box_in_polygon


class CameraTracker(BYTETracker):
    """Giữ bộ đếm ID toàn cục tăng dần khi camera khác reset tracker."""

    @staticmethod
    def reset_id() -> None:
        """Không reset bộ đếm STrack dùng chung của Ultralytics."""
        return None


class VehicleDetector(LockedYOLO):
    """Một YOLO, nhiều ByteTrack; khóa bảo vệ inference và bộ đếm ID."""

    def __init__(self, path: Path, device: str, imgsz: int) -> None:
        super().__init__(path, device, imgsz)
        self.trackers: dict[str, CameraTracker] = {}
        self.classes = [
            i
            for i, name in self.model.names.items()
            if name in {"bicycle", "car", "motorcycle", "bus", "truck"}
        ]
        if len(self.classes) != 5:
            raise ValueError("vehicle_model cần weights COCO với đủ 5 nhóm xe")

    def reset_camera(self, camera_id: str, confidence: float) -> None:
        """Tạo tracker rỗng cho camera mà không tải lại weights."""
        with self.lock:
            args = SimpleNamespace(
                track_high_thresh=confidence,
                track_low_thresh=min(0.1, confidence),
                new_track_thresh=confidence,
                track_buffer=30,
                match_thresh=0.8,
                fuse_score=True,
                frame_rate=30,
            )
            try:
                self.trackers[camera_id] = CameraTracker(args, frame_rate=30)
            except TypeError:
                self.trackers[camera_id] = CameraTracker(args)

    def drop_camera(self, camera_id: str) -> None:
        """Giải phóng trạng thái tracking của camera đã dừng."""
        with self.lock:
            self.trackers.pop(camera_id, None)

    def track(
        self,
        frame: np.ndarray,
        camera_id: str,
        confidence: float,
        polygon: Polygon = (),
    ) -> list[Detection]:
        """Suy luận xe và cập nhật ByteTrack tích hợp trong Ultralytics."""
        with self.lock:
            if camera_id not in self.trackers:
                self.reset_camera(camera_id, confidence)
            result = self.model.predict(
                frame,
                conf=min(0.1, confidence),
                classes=self.classes,
                device=self.device,
                imgsz=self.imgsz,
                verbose=False,
            )[0]
            boxes = result.boxes.cpu().numpy()
            if polygon:
                mask = np.array(
                    [box_in_polygon(tuple(box), polygon) for box in boxes.xyxy],
                    dtype=bool,
                )
                boxes = boxes[mask]
            rows = self.trackers[camera_id].update(boxes, frame)
            return [
                Detection(
                    tuple(int(round(v)) for v in row[:4]),
                    float(row[5]),
                    result.names[int(row[6])],
                    int(row[4]),
                )
                for row in rows
                if box_in_polygon(tuple(row[:4]), polygon)
            ]
