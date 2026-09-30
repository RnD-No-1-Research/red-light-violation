"""Chuẩn hóa định dạng biển dân sự phổ biến; không xác thực đăng ký xe."""

import re

PLATE_PATTERN = re.compile(r"^([1-9][0-9][A-Z][0-9]?)([0-9]{5})$")


def normalize_plate(text: str) -> str:
    """Nhận dạng 29A-12345, 30E1-12345; loại khoảng trắng và dấu phân cách."""
    compact = re.sub(r"[\s.\-]", "", text.upper())
    match = PLATE_PATTERN.fullmatch(compact)
    return f"{match[1]}-{match[2]}" if match else "UNKNOWN"
