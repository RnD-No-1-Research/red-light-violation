"""Khởi tạo AI ở tầng Model và chọn GPU/CPU."""

import logging
from dataclasses import dataclass
from typing import Any

import torch

from models.plate_detector import PlateDetector
from models.plate_ocr import PlateOCR
from models.traffic_light_detector import TrafficLightDetector
from models.vehicle_detector import VehicleDetector


@dataclass
class ModelBundle:
    """Bốn instance model duy nhất dùng cho mọi camera."""

    traffic: TrafficLightDetector
    vehicle: VehicleDetector
    plate: PlateDetector
    ocr: PlateOCR


def select_device(requested: str) -> str:
    """Ưu tiên CUDA nếu khả dụng; fallback CPU khi kiểm tra CUDA thất bại."""
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            probe = torch.ones((2, 2), device="cuda:0")
            (probe @ probe).cpu()
            return "cuda:0"
        except RuntimeError:
            logging.warning("CUDA không hoạt động; chuyển sang CPU")
    else:
        logging.warning("Không có CUDA khả dụng; chạy CPU")
    return "cpu"


def create_models(settings: dict[str, Any]) -> ModelBundle:
    """Nạp ba YOLO và một EasyOCR; dừng với hướng dẫn nếu thiếu weights."""
    for key in ("traffic_light_model", "vehicle_model", "plate_model"):
        if not settings[key].is_file():
            raise FileNotFoundError(
                f"Thiếu {settings[key].name}. Chạy python scripts/download_weights.py"
            )
    device = select_device(settings["device"])
    torch.set_num_threads(2)
    logging.info("Nạp model một lần trên %s", device)
    size = settings["imgsz"]
    return ModelBundle(
        TrafficLightDetector(
            settings["traffic_light_model"],
            device,
            size,
            settings["traffic_light_labels"],
        ),
        VehicleDetector(settings["vehicle_model"], device, size),
        PlateDetector(settings["plate_model"], device, size),
        PlateOCR(settings["ocr_model_dir"], device, settings["ocr_download_enabled"]),
    )
