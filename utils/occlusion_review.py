"""Bounded evidence for suspected crossings during a detection gap.

This never confirms a violation or predicts an invisible bounding box.
Packets share their image arrays; at most one second of near-line history is held.
"""

from dataclasses import dataclass
from math import hypot

from models.types import Detection
from utils.box_motion import box_motion_issue
from utils.line_crossing import crosses_segment, signed_side
from utils.video_source import FramePacket
from utils.vehicle_anchor import vehicle_anchor


@dataclass(frozen=True)
class ReviewObservation:
    packet: FramePacket
    vehicle: Detection
    seconds: float


@dataclass(frozen=True)
class ReviewEvent:
    before: ReviewObservation
    after: Detection
    gap_seconds: float


def anchor(vehicle: Detection) -> tuple[float, float]:
    return vehicle_anchor(vehicle.bbox)


class OcclusionReview:
    """Keep only red, approaching tracks near the finite stop line."""

    def __init__(self, start, end, direction: str, max_gap_seconds: float = 1.0):
        self.start, self.end, self.direction = start, end, direction
        self.max_gap_seconds = max_gap_seconds
        self.previous: dict[int, ReviewObservation] = {}
        self.suspected: set[int] = set()
        self.last_seconds: float | None = None
        self.epoch: int | None = None

    def reset(self) -> None:
        self.previous.clear()
        self.suspected.clear()
        self.last_seconds = None
        self.epoch = None

    def update(
        self,
        packet: FramePacket,
        tracks: list[Detection],
        light: str,
        raw_light: str,
        confirmed: set[int],
    ) -> list[ReviewEvent]:
        seconds = (
            packet.source_seconds
            if packet.source_seconds is not None
            else packet.timestamp.timestamp()
        )
        if self.epoch != packet.epoch:
            self.reset()
            self.epoch = packet.epoch
        if self.last_seconds is not None and seconds <= self.last_seconds:
            self.previous.clear()
        self.last_seconds = seconds
        if self.max_gap_seconds <= 0 or light != "RED" or raw_light != "RED":
            self.previous.clear()
            return []
        self.previous = {
            key: value
            for key, value in self.previous.items()
            if 0 < seconds - value.seconds <= self.max_gap_seconds
            and key not in confirmed
            and key not in self.suspected
        }
        events = []
        length = hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])
        if not length:
            return events
        for vehicle in tracks:
            tid = vehicle.track_id
            old = self.previous.pop(tid, None)
            if tid < 0 or tid in confirmed or tid in self.suspected:
                continue
            point = anchor(vehicle)
            x1, y1, x2, y2 = vehicle.bbox
            width, height = x2 - x1, y2 - y1
            if min(width, height) <= 0:
                continue
            if old is not None and packet.sequence - old.packet.sequence > 1:
                ox1, oy1, ox2, oy2 = old.vehicle.bbox
                ow, oh = ox2 - ox1, oy2 - oy1
                prior = anchor(old.vehicle)
                plausible = (
                    vehicle.class_name == old.vehicle.class_name
                    and box_motion_issue(
                        old.vehicle.bbox,
                        vehicle.bbox,
                        self.start,
                        self.end,
                        self.direction,
                    )
                    is None
                    and 0.5 <= width / ow <= 2.0
                    and 0.5 <= height / oh <= 2.0
                    and hypot(point[0] - prior[0], point[1] - prior[1])
                    <= 2 * hypot(ow, oh)
                )
                if plausible and crosses_segment(
                    prior, point, self.start, self.end, self.direction
                ):
                    self.suspected.add(tid)
                    events.append(ReviewEvent(old, vehicle, seconds - old.seconds))
                    continue
            side = signed_side(point, self.start, self.end)
            approaching = (
                side < 0 if self.direction == "negative_to_positive" else side > 0
            )
            dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
            projection = (
                (point[0] - self.start[0]) * dx + (point[1] - self.start[1]) * dy
            ) / length**2
            if approaching and abs(side) / length <= height and 0 <= projection <= 1:
                self.previous[tid] = ReviewObservation(packet, vehicle, seconds)
        return events

    def retained_image_bytes(self) -> int:
        """Count shared images once, for diagnostics (not called per frame)."""
        images = {id(o.packet.image): o.packet.image for o in self.previous.values()}
        return sum(image.nbytes for image in images.values())
