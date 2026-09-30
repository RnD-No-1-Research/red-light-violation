"""Kiểm tra thứ tự OCR mà không khởi tạo EasyOCR Reader."""

from models.plate_ocr import order_fragments


def test_two_rows_sorted() -> None:
    """Đọc hàng trên trước, trong từng hàng đọc trái sang phải."""
    fragments = [
        ([[0, 30], [50, 30], [50, 45], [0, 45]], "12345", 0.9),
        ([[30, 2], [50, 2], [50, 17], [30, 17]], "E1", 0.9),
        ([[0, 0], [25, 0], [25, 15], [0, 15]], "30", 0.9),
    ]
    assert [f[1] for f in order_fragments(fragments)] == ["30", "E1", "12345"]
