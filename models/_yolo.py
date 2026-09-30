"""Nạp YOLO một lần và bảo vệ toàn bộ truy cập predictor."""

from pathlib import Path
from threading import RLock

from ultralytics import YOLO


class LockedYOLO:
    """Model YOLO11 dùng chung giữa các camera, có khóa riêng."""

    def __init__(self, path: Path, device: str, imgsz: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Thiếu {path.name}. Chạy python scripts/download_weights.py"
            )
        self.lock = RLock()
        self.model = YOLO(str(path), task="detect")
        if self.model.task != "detect":
            raise ValueError(f"{path.name} phải là model detection")
        modules = {type(m).__name__ for m in self.model.model.modules()}
        if not {"C3k2", "C2PSA"}.issubset(modules):
            raise ValueError(f"{path.name} không có cấu trúc YOLO11 dự kiến")
        self.device = device
        self.imgsz = imgsz
