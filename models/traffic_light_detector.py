"""Suy luận màu đèn trong ROI, không thực hiện bỏ phiếu hoặc nghiệp vụ."""

from pathlib import Path

import numpy as np

from models._yolo import LockedYOLO


class TrafficLightDetector(LockedYOLO):
    """Phát hiện RED/GREEN/YELLOW/UNKNOWN từ nhãn của checkpoint."""

    def __init__(
        self,
        path: Path,
        device: str,
        imgsz: int,
        labels: dict[str, str],
    ) -> None:
        super().__init__(path, device, imgsz)
        self.labels = {key.casefold(): value for key, value in labels.items()}
        available = {self.labels.get(n.casefold()) for n in self.model.names.values()}
        if not {"RED", "GREEN", "YELLOW"}.issubset(available):
            raise ValueError(
                "Weights đèn/traffic_light_labels thiếu RED, GREEN, YELLOW"
            )

    def detect(
        self,
        roi: np.ndarray,
        confidence: float,
        *,
        canvas_size: int = 0,
        fallback_roi: np.ndarray | None = None,
    ) -> str:
        """Thử ROI dự phòng khi UNKNOWN; không ghi đè màu đã nhận ở ROI chính."""
        if roi.size == 0:
            return "UNKNOWN"
        with self.lock:
            color = self._predict_color(roi, confidence)
            if color != "UNKNOWN" or (not canvas_size and fallback_roi is None):
                return color
            candidate = roi if fallback_roi is None else fallback_roi
            if candidate.size == 0:
                return "UNKNOWN"
            if canvas_size:
                # Đệm đen, không kéo giãn ROI; hiệu chỉnh riêng từng camera/model.
                height, width = candidate.shape[:2]
                side = max(canvas_size, height, width)
                canvas = np.zeros(
                    (side, side, *candidate.shape[2:]), dtype=candidate.dtype
                )
                x, y = (side - width) // 2, (side - height) // 2
                canvas[y : y + height, x : x + width] = candidate
                candidate = canvas
            return self._predict_color(candidate, confidence)

    def _predict_color(self, roi: np.ndarray, confidence: float) -> str:
        """Suy luận một crop, chỉ gọi khi đã giữ khóa model."""
        max_dim = max(roi.shape[:2])
        img_size = self.imgsz if max_dim >= 300 else (320 if max_dim >= 100 else 160)
        result = self.model.predict(
            roi, conf=confidence, device=self.device, imgsz=img_size, verbose=False
        )[0]
        best_color, best_conf = "UNKNOWN", 0.0
        for box in result.boxes:
            color = self.labels.get(
                result.names[int(box.cls.item())].casefold(), "UNKNOWN"
            )
            score = float(box.conf.item())
            if color in ("RED", "GREEN", "YELLOW") and score > best_conf:
                best_conf, best_color = score, color
        return best_color
