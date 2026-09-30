"""Phát hiện vùng biển số trong ảnh xe."""

import numpy as np

from models._yolo import LockedYOLO
from models.types import Detection


class PlateDetector(LockedYOLO):
    """YOLO11 biển số dùng chung; trả biển có confidence cao nhất."""

    def detect(self, vehicle: np.ndarray, confidence: float) -> Detection | None:
        """Trả bbox biển theo tọa độ crop xe, hoặc None khi không thấy."""
        with self.lock:
            result = self.model.predict(
                vehicle,
                conf=confidence,
                device=self.device,
                imgsz=self.imgsz,
                verbose=False,
            )[0]
            boxes = result.boxes
            if len(boxes) == 0:
                return None
            index = int(boxes.conf.argmax().item())
            return Detection(
                tuple(int(round(v)) for v in boxes.xyxy[index].cpu().tolist()),
                float(boxes.conf[index].item()),
                result.names[int(boxes.cls[index].item())],
            )
