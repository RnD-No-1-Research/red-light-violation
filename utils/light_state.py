"""Bỏ phiếu trạng thái đèn, độc lập với suy luận AI."""

from collections import Counter, deque


class LightSmoother:
    """Đòi hỏi đa số tuyệt đối trong cửa sổ đầy đủ; hòa trả UNKNOWN."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window phải >= 1")
        self.values: deque[str] = deque(maxlen=window)

    def update(self, state: str) -> str:
        """Thêm quan sát; không phát RED khi cửa sổ chưa đủ frame."""
        self.values.append(state)
        if len(self.values) < self.values.maxlen:
            return "UNKNOWN"
        winner, count = Counter(self.values).most_common(1)[0]
        return winner if count > len(self.values) / 2 else "UNKNOWN"

    def reset(self) -> None:
        """Xóa cửa sổ sau mất kết nối để không dùng màu đèn cũ."""
        self.values.clear()
