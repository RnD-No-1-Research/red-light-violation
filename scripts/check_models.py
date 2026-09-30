"""Kiểm tra weights thật và OCR trên CPU/GPU với ảnh rỗng."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    """Nạp model, kiểm tra nhãn và gọi thử bốn nhánh suy luận."""
    config_dir = ROOT / "output/.ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_VERBOSE", "false")
    import numpy as np

    from main import configure_logging
    from models.runtime import create_models
    from utils.config_loader import load_settings

    configure_logging()
    settings = load_settings(ROOT / "config/settings.yaml")
    models = create_models(settings)
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    print("[system] Light labels:", models.traffic.model.names)
    print("[system] Plate labels:", models.plate.model.names)
    print("[system] Light:", models.traffic.detect(frame, 0.5))
    print("[system] Vehicles:", models.vehicle.track(frame, "check_a", 0.4))
    print("[system] Vehicles camera B:", models.vehicle.track(frame, "check_b", 0.4))
    print("[system] Plate:", models.plate.detect(frame, 0.5))
    print("[system] OCR:", models.ocr.read(frame))
    print("[system] PASS: model load/inference; không phải kiểm định độ chính xác")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
