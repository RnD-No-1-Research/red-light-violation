"""Shared point for stop-line decisions and the displayed vehicle marker."""


def vehicle_anchor(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Use the user's chosen point: horizontal center, 20 pixels above the bottom."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2, float(y2 - 20))
