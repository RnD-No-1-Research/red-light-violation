"""Kiểm tra chuẩn hóa biển dân sự một/hai dòng."""

import pytest

from utils.plate_format import normalize_plate


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("29A-12345", "29A-12345"),
        ("30E1\n123.45", "30E1-12345"),
        ("29a 12345", "29A-12345"),
        ("", "UNKNOWN"),
        ("29A-12O45", "UNKNOWN"),
        ("garbage", "UNKNOWN"),
        ("29A1234567", "UNKNOWN"),
        ("29A/12345", "UNKNOWN"),
    ],
)
def test_format(raw: str, expected: str) -> None:
    """Không đoán ký tự hoặc chấp nhận chuỗi thừa."""
    assert normalize_plate(raw) == expected
