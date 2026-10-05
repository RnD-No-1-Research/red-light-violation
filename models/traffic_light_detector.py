"""Suy luận màu đèn trong ROI, không thực hiện bỏ phiếu hoặc nghiệp vụ."""

from pathlib import Path

import cv2
import numpy as np

from models._yolo import LockedYOLO


def count_hsv_colors(crop: np.ndarray) -> tuple[int, int, int]:
    """Đếm số pixel thuộc dải màu RED, YELLOW, GREEN trong không gian màu HSV."""
    if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
        return 0, 0, 0
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:
        return 0, 0, 0
    # Red: dải 0-10 và 170-180
    r_mask = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255)) | cv2.inRange(
        hsv, (170, 70, 70), (180, 255, 255)
    )
    # Green: dải 40-95
    g_mask = cv2.inRange(hsv, (40, 70, 70), (95, 255, 255))
    # Yellow / Amber: dải 15-38
    y_mask = cv2.inRange(hsv, (15, 70, 70), (38, 255, 255))

    return (
        int(np.count_nonzero(r_mask)),
        int(np.count_nonzero(y_mask)),
        int(np.count_nonzero(g_mask)),
    )


def verify_color_with_hsv(crop: np.ndarray, yolo_color: str) -> str:
    """Xác thực hoặc hiệu chỉnh màu đèn từ YOLO bằng lọc pixel HSV.

    Vẫn đảm bảo duy trì chính xác 4 lớp: 'RED', 'GREEN', 'YELLOW', 'UNKNOWN'.
    """
    red_px, yellow_px, green_px = count_hsv_colors(crop)

    # 1. Nếu YOLO nhận là RED nhưng thực tế đèn đang XANH (hoặc VÀNG)
    if yolo_color == "RED":
        if red_px < 10 and green_px >= 20:
            return "GREEN"
        if red_px < 10 and yellow_px >= 20:
            return "YELLOW"
        return "RED"

    # 2. Nếu YOLO nhận là GREEN nhưng thực tế là ĐỎ (hoặc VÀNG)
    if yolo_color == "GREEN":
        if green_px < 10 and red_px >= 20:
            return "RED"
        if green_px < 10 and yellow_px >= 20:
            return "YELLOW"
        return "GREEN"

    # 3. Nếu YOLO nhận là YELLOW nhưng thực tế là ĐỎ hoặc XANH
    if yolo_color == "YELLOW":
        if yellow_px < 10 and red_px >= 20 and red_px > green_px * 2:
            return "RED"
        if yellow_px < 10 and green_px >= 20 and green_px > red_px * 2:
            return "GREEN"
        return "YELLOW"

    # 4. Nếu YOLO trả về UNKNOWN: khôi phục màu nếu có 1 màu thực sự sáng vượt trội
    if yolo_color == "UNKNOWN":
        if red_px >= 25 and red_px > green_px * 2 and red_px > yellow_px * 2:
            return "RED"
        if green_px >= 25 and green_px > red_px * 2 and green_px > yellow_px * 2:
            return "GREEN"
        if yellow_px >= 20 and yellow_px > red_px * 2 and yellow_px > green_px * 2:
            return "YELLOW"

    return yolo_color


class TrafficLightDetector(LockedYOLO):
    """Phát hiện RED/GREEN/YELLOW/UNKNOWN từ nhãn của checkpoint kết hợp xác thực HSV."""

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
        """Suy luận một crop kết hợp kiểm tra HSV, chỉ gọi khi đã giữ khóa model."""
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
        return verify_color_with_hsv(roi, best_color)

