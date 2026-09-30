"""Kiểu kết quả độc lập với đối tượng tensor của thư viện AI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Bounding box và độ tin cậy của một phát hiện."""

    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    track_id: int = -1


@dataclass(frozen=True)
class OCRResult:
    """Văn bản OCR thô và độ tin cậy trung bình có trọng số ký tự."""

    text: str
    confidence: float
