"""Reject obvious box deformation before interpreting an anchor crossing."""

from math import hypot

Box = tuple[int, int, int, int]


def box_motion_issue(
    previous: Box,
    current: Box,
    start,
    end,
    direction: str,
) -> str | None:
    """Check dimensions and motion of both edge midpoints along the line normal.

    One pixel tolerance accommodates rounded tracker coordinates. This is a
    geometric heuristic, not proof of physical identity or motion.
    """
    px1, py1, px2, py2 = previous
    x1, y1, x2, y2 = current
    pw, ph, width, height = px2 - px1, py2 - py1, x2 - x1, y2 - y1
    if min(pw, ph, width, height) <= 0:
        return "invalid_box"
    if not (0.5 <= width / pw <= 2 and 0.5 <= height / ph <= 2):
        return "box_size_jump"
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if not length:
        return "invalid_line"
    sign = 1 if direction == "negative_to_positive" else -1
    old_x, new_x = (px1 + px2) / 2, (x1 + x2) / 2
    top_motion = sign * (dx * (y1 - py1) - dy * (new_x - old_x)) / length
    bottom_motion = sign * (dx * (y2 - py2) - dy * (new_x - old_x)) / length
    if bottom_motion > 0 and top_motion + 1.0 < 0.25 * bottom_motion:
        return "box_deformation"
    return None
