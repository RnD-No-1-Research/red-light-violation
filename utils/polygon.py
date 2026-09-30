"""Kiểm tra vùng đa giác và điểm tham chiếu xe, không phụ thuộc AI."""

from __future__ import annotations

import math

Point = tuple[float, float]
Polygon = tuple[Point, ...]
EPSILON = 1e-6


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _on_segment(p: Point, a: Point, b: Point) -> bool:
    return (
        abs(_cross(a, b, p)) <= EPSILON
        and min(a[0], b[0]) - EPSILON <= p[0] <= max(a[0], b[0]) + EPSILON
        and min(a[1], b[1]) - EPSILON <= p[1] <= max(a[1], b[1]) + EPSILON
    )


def _intersects(a: Point, b: Point, c: Point, d: Point) -> bool:
    sides = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    if sides[0] * sides[1] < 0 and sides[2] * sides[3] < 0:
        return True
    return any(
        (
            _on_segment(c, a, b),
            _on_segment(d, a, b),
            _on_segment(a, c, d),
            _on_segment(b, c, d),
        )
    )


def parse_polygon(value: object) -> Polygon:
    """Validate polygon đơn; None hoặc danh sách rỗng nghĩa là toàn frame."""
    if value is None or (isinstance(value, (list, tuple)) and not value):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("detection_polygon phải là danh sách các điểm [x, y]")
    points: list[Point] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Mỗi đỉnh polygon phải là [x, y]")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in point
        ):
            raise ValueError("Tọa độ polygon phải là số hữu hạn không âm")
        points.append((float(point[0]), float(point[1])))
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError("Polygon cần ít nhất 3 đỉnh khác nhau, không lặp đỉnh")
    n = len(points)
    area = sum(
        points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
        for i in range(n)
    )
    if abs(area) <= EPSILON:
        raise ValueError("Polygon phải có diện tích lớn hơn 0")
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _intersects(
                points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]
            ):
                raise ValueError("Các cạnh polygon không được tự cắt nhau")
    return tuple(points)


def validate_polygon_bounds(polygon: Polygon, size: tuple[int, int]) -> None:
    """Kiểm tra mọi đỉnh nằm trong frame gốc."""
    width, height = size
    if any(not (0 <= x < width and 0 <= y < height) for x, y in polygon):
        raise ValueError("Polygon nằm ngoài frame; hãy vẽ lại")


def contains_point(point: Point, polygon: Polygon) -> bool:
    """Trả True cho điểm trong hoặc trên biên; vùng rỗng nhận mọi điểm."""
    if not polygon:
        return True
    x, y = point
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if _on_segment(point, a, b):
            return True
        if (a[1] > y) != (b[1] > y):
            crossing_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < crossing_x:
                inside = not inside
    return inside


def box_in_polygon(bbox: tuple[float, float, float, float], polygon: Polygon) -> bool:
    """Xét xe bằng trung điểm cạnh dưới bbox, cùng điểm dùng để cắt vạch."""
    x1, _, x2, y2 = bbox
    return contains_point(((x1 + x2) / 2, y2), polygon)
