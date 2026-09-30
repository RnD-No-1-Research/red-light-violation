"""Các tình huống cắt vạch và loại trừ vi phạm."""

import pytest

from utils.line_crossing import LineCrossing, crosses_segment


@pytest.mark.parametrize(
    "light,expected",
    [("RED", True), ("GREEN", False), ("YELLOW", False), ("UNKNOWN", False)],
)
def test_light_condition(light: str, expected: bool) -> None:
    """Chỉ cắt vạch đúng chiều khi đỏ mới là vi phạm."""
    detector = LineCrossing((0, 10), (100, 10), "negative_to_positive")
    assert not detector.update(1, (20, 8), 1, light)
    assert detector.update(1, (20, 12), 2, light) == expected


def test_no_retroactive_or_duplicate() -> None:
    """Không báo xe đã vượt từ trước hoặc ghi lặp cùng track."""
    detector = LineCrossing((0, 10), (100, 10), "negative_to_positive")
    assert not detector.update(1, (20, 12), 1, "GREEN")
    assert not detector.update(1, (20, 14), 2, "RED")
    assert not detector.update(2, (20, 8), 1, "RED")
    assert detector.update(2, (20, 12), 2, "RED")
    assert not detector.update(2, (20, 8), 3, "RED")
    assert not detector.update(2, (20, 12), 4, "RED")


def test_segment_direction_touch_and_gap() -> None:
    """Loại kéo dài ngoài đoạn, ngược chiều và khoảng mất track."""
    assert not crosses_segment((110, 8), (110, 12), (0, 10), (100, 10))
    assert not crosses_segment((20, 12), (20, 8), (0, 10), (100, 10))
    assert crosses_segment(
        (20, 12), (20, 8), (0, 10), (100, 10), "positive_to_negative"
    )
    detector = LineCrossing((0, 10), (100, 10), "negative_to_positive")
    detector.update(1, (20, 8), 1, "RED")
    assert not detector.update(1, (20, 10), 2, "RED")
    assert detector.update(1, (20, 12), 3, "RED")
    detector.update(2, (20, 8), 1, "RED")
    assert not detector.update(2, (20, 12), 3, "RED")


def test_diagonal_and_reset() -> None:
    """Hỗ trợ vạch chéo và xóa lịch sử sau reconnect."""
    assert crosses_segment((5, 2), (5, 8), (0, 0), (10, 10))
    detector = LineCrossing((0, 10), (100, 10), "negative_to_positive")
    detector.update(1, (20, 8), 1, "RED")
    detector.reset()
    assert not detector.update(1, (20, 12), 2, "RED")


def test_touch_beyond_segment_is_not_crossing() -> None:
    """Điểm chạm ngoài đoạn không được nối tắt thành một lần cắt vạch."""
    detector = LineCrossing((0, 10), (100, 10), "negative_to_positive")
    detector.update(1, (20, 8), 1, "RED")
    detector.update(1, (120, 10), 2, "RED")
    assert not detector.update(1, (120, 12), 3, "RED")
