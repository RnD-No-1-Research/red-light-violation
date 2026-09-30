"""Hình học đoạn vạch và trạng thái chuyển động riêng từng camera."""

from dataclasses import dataclass

Point = tuple[float, float]


def signed_side(point: Point, start: Point, end: Point) -> float:
    """Trả tích có hướng: dương là bên phải hướng start tới end trên ảnh."""
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )


def crosses_segment(
    previous: Point,
    current: Point,
    start: Point,
    end: Point,
    direction: str = "negative_to_positive",
) -> bool:
    """Kiểm tra đi qua đoạn vạch hữu hạn theo đúng chiều, không chỉ đường kéo dài."""
    before = signed_side(previous, start, end)
    after = signed_side(current, start, end)
    if direction == "negative_to_positive":
        directed = before < 0 < after
    elif direction == "positive_to_negative":
        directed = before > 0 > after
    else:
        raise ValueError("Hướng cắt vạch không hợp lệ")
    if not directed:
        return False
    ratio = before / (before - after)
    intersection = (
        previous[0] + ratio * (current[0] - previous[0]),
        previous[1] + ratio * (current[1] - previous[1]),
    )
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = dx * dx + dy * dy
    if norm == 0:
        return False
    projection = (
        (intersection[0] - start[0]) * dx + (intersection[1] - start[1]) * dy
    ) / norm
    return 0 <= projection <= 1


@dataclass
class Observation:
    """Lần nhìn thấy gần nhất và điểm không nằm đúng trên vạch."""

    point: Point
    frame_index: int
    contact: Point | None = None


class LineCrossing:
    """Ghi nhận tối đa một vi phạm cho mỗi track trong phiên camera."""

    def __init__(self, start: Point, end: Point, direction: str) -> None:
        self.start = start
        self.end = end
        self.direction = direction
        self.previous: dict[int, Observation] = {}
        self.recorded: set[int] = set()

    def update(self, track_id: int, point: Point, frame_index: int, light: str) -> bool:
        """Xét cắt vạch ở hai lần quan sát liên tiếp và đèn đỏ hiện tại."""
        old = self.previous.get(track_id)
        consecutive = old is not None and old.frame_index == frame_index - 1
        crossed = bool(
            consecutive
            and crosses_segment(old.point, point, self.start, self.end, self.direction)
        )
        if consecutive and old.contact is not None:
            before = signed_side(old.point, self.start, self.end)
            after = signed_side(point, self.start, self.end)
            directed = (
                before < 0 < after
                if self.direction == "negative_to_positive"
                else before > 0 > after
            )
            # Dùng điểm chạm thực tế, tránh nối tắt qua phần kéo dài của vạch.
            x, y = old.contact
            inside = min(self.start[0], self.end[0]) <= x <= max(
                self.start[0], self.end[0]
            ) and min(self.start[1], self.end[1]) <= y <= max(
                self.start[1], self.end[1]
            )
            crossed = directed and inside
        # Giữ điểm trước vạch nếu xe chạm đúng vạch, nhưng không nối qua mất track.
        anchor = (
            old.point
            if consecutive and signed_side(point, self.start, self.end) == 0
            else point
        )
        contact = point if signed_side(point, self.start, self.end) == 0 else None
        self.previous[track_id] = Observation(anchor, frame_index, contact)
        if crossed and light == "RED" and track_id not in self.recorded:
            self.recorded.add(track_id)
            return True
        return False

    def prune(self, frame_index: int) -> None:
        """Xóa vị trí track mất dấu; vẫn giữ tập ID đã ghi để chống trùng."""
        self.previous = {
            key: value
            for key, value in self.previous.items()
            if value.frame_index >= frame_index - 1
        }

    def reset(self) -> None:
        """Xóa lịch sử vị trí và ID khi tạo phiên RTSP mới."""
        self.previous.clear()
        self.recorded.clear()
