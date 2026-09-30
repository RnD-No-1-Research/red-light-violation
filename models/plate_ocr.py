"""EasyOCR với thứ tự đọc theo hàng cho biển một hoặc hai dòng."""

from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import easyocr
import numpy as np

from models.types import OCRResult


def order_fragments(fragments: list[Any]) -> list[Any]:
    """Gom các hộp OCR gần cùng hàng, rồi đọc từ trái sang phải."""
    rows: list[list[Any]] = []
    for fragment in sorted(fragments, key=lambda f: np.mean(np.array(f[0])[:, 1])):
        box = np.array(fragment[0])
        center = float(box[:, 1].mean())
        height = max(1.0, float(np.ptp(box[:, 1])))
        for row in rows:
            ref = np.array(row[0][0])
            if abs(center - float(ref[:, 1].mean())) <= 0.5 * max(
                height, np.ptp(ref[:, 1])
            ):
                row.append(fragment)
                break
        else:
            rows.append([fragment])
    return [
        f for row in rows for f in sorted(row, key=lambda f: min(p[0] for p in f[0]))
    ]


class PlateOCR:
    """Khởi tạo EasyOCR một lần; chỉ đọc chữ Latin và chữ số trên biển."""

    def __init__(self, directory: Path, device: str, download_enabled: bool) -> None:
        self.lock = Lock()
        directory.mkdir(parents=True, exist_ok=True)
        self.reader = easyocr.Reader(
            ["en"],
            gpu=device if device != "cpu" else False,
            model_storage_directory=str(directory),
            user_network_directory=str(directory / "user_network"),
            download_enabled=download_enabled,
            verbose=False,
        )

    def read(self, plate: np.ndarray) -> OCRResult:
        """Đọc biển theo thứ tự hàng; không sửa O thành 0 hoặc đoán ký tự."""
        if plate.size == 0:
            return OCRResult("", 0.0)
        scale = max(1.0, 100.0 / plate.shape[0])
        enlarged = cv2.resize(
            plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        with self.lock:
            fragments = self.reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.",
                workers=0,
            )
        ordered = order_fragments(fragments)
        text = " ".join(str(f[1]) for f in ordered)
        length = sum(len(str(f[1])) for f in ordered)
        score = sum(float(f[2]) * len(str(f[1])) for f in ordered) / max(length, 1)
        return OCRResult(text, score)
