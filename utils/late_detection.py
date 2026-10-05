"""Optional policy: count vehicles first tracked just beyond the red stop line.

This infers a violation from stable presence or subsequent motion; it does not claim
that a before/after crossing was observed. Controllers resolve shared defaults
and per-camera overrides before constructing this independent state machine.
"""

from dataclasses import dataclass
from math import hypot

from models.types import Detection
from utils.box_motion import box_motion_issue
from utils.line_crossing import signed_side
from utils.video_source import FramePacket
from utils.vehicle_anchor import vehicle_anchor


@dataclass
class Candidate:
    first: Detection
    last: Detection
    first_seconds: float
    last_sequence: int
    first_sequence: int
    last_seconds: float
    count: int = 1


@dataclass(frozen=True)
class LateViolation:
    vehicle: Detection
    details: dict


@dataclass(frozen=True)
class LateSuspicion:
    vehicle: Detection
    details: dict


class LateDetection:
    def __init__(
        self,
        start,
        end,
        direction,
        *,
        enabled=False,
        band_height_ratio=1.0,
        min_red_seconds=1.0,
        require_forward_motion=True,
    ):
        self.start, self.end, self.direction = start, end, direction
        self.enabled = enabled
        self.band_height_ratio = band_height_ratio
        self.min_red_seconds = min_red_seconds
        self.require_forward_motion = require_forward_motion
        self.reset()

    def reset(self):
        self.seen: set[int] = set()
        self.pending: dict[int, Candidate] = {}
        self.eligible: dict[int, str] = {}
        # A rejected crossing needs fresh coherent motion, never stationary presence.
        self.motion_recovery: set[int] = set()
        self.suspected: set[int] = set()
        self.new_suspicions: list[LateSuspicion] = []
        self.last_decisions: dict[int, str] = {}
        self.red_since: float | None = None
        self.last_seconds: float | None = None
        self.epoch: int | None = None

    def coordinates(self, vehicle):
        point = vehicle_anchor(vehicle.bbox)
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        length = hypot(dx, dy)
        sign = 1 if self.direction == "negative_to_positive" else -1
        normal = sign * signed_side(point, self.start, self.end) / length
        tangent = (
            (point[0] - self.start[0]) * dx + (point[1] - self.start[1]) * dy
        ) / length
        return normal, tangent, length

    def update(
        self,
        packet: FramePacket,
        tracks: list[Detection],
        light: str,
        raw_light: str,
        recorded: set[int],
        *,
        rejected_ids: set[int] | None = None,
        rejection_reasons: dict[int, str] | None = None,
    ) -> list[LateViolation]:
        self.new_suspicions = []
        if not self.enabled:
            return []
        seconds = (
            packet.source_seconds
            if packet.source_seconds is not None
            else packet.timestamp.timestamp()
        )
        starting_epoch = packet.epoch != self.epoch
        if starting_epoch:
            self.reset()
            self.epoch = packet.epoch
        if self.last_seconds is not None and seconds <= self.last_seconds:
            self.pending.clear()
            self.eligible.clear()
            self.motion_recovery.clear()
            self.red_since = None
        self.last_seconds = seconds
        self.last_decisions = {}
        self.pending = {
            tid: c
            for tid, c in self.pending.items()
            if tid not in recorded
            and 0 < seconds - c.first_seconds <= 2.0
            and 0 < seconds - c.last_seconds <= 0.75
        }
        if raw_light != "RED" or light in ("GREEN", "YELLOW"):
            self.red_since = None
            self.pending.clear()
        elif self.red_since is None:
            self.red_since = seconds
        if light != "RED":
            self.pending.clear()
        # A missing track must not carry red-phase eligibility across green/yellow.
        if light in ("GREEN", "YELLOW") or raw_light in ("GREEN", "YELLOW"):
            self.eligible.clear()
            self.motion_recovery.clear()
        red_ready = (
            light == "RED" and raw_light == "RED" and self.red_since is not None
            and seconds - self.red_since >= self.min_red_seconds
        )
        events = []
        for vehicle in tracks:
            tid = vehicle.track_id
            is_new = tid not in self.seen
            self.seen.add(tid)
            if tid < 0 or tid in recorded:
                self.motion_recovery.discard(tid)
                self.suspected.discard(tid)
                continue
            x1, y1, x2, y2 = vehicle.bbox
            if min(x2 - x1, y2 - y1) <= 0:
                continue
            distance, along, length = self.coordinates(vehicle)
            if distance <= 0:
                self.eligible[tid] = "PREVIOUSLY_BEFORE_LINE"
                self.pending.pop(tid, None)
                self.last_decisions[tid] = "before_line"
                continue
            if light in ("GREEN", "YELLOW") or raw_light in ("GREEN", "YELLOW"):
                self.eligible.pop(tid, None)
            elif is_new and not starting_epoch and raw_light == "RED":
                self.eligible[tid] = "FIRST_SEEN_RED"
            if tid in (rejected_ids or set()):
                self.pending.pop(tid, None)
                self.last_decisions[tid] = "direct_box_rejected"
                # Only a deformation rejection may recover. Class/size jumps and
                # callers without a reason keep the conservative hard rejection.
                if (
                    (rejection_reasons or {}).get(tid) == "box_deformation"
                    and tid in self.eligible
                ):
                    self.motion_recovery.add(tid)
                else:
                    self.eligible.pop(tid, None)
                    self.motion_recovery.discard(tid)
                    continue
            if tid not in self.eligible:
                self.last_decisions[tid] = "no_red_entry_history"
                continue
            if not red_ready:
                self.last_decisions[tid] = "waiting_red"
                continue
            candidate = self.pending.get(tid)
            in_band = (
                2 <= distance <= (y2 - y1) * self.band_height_ratio + 2
                and 0 <= along <= length
            )
            if candidate is None:
                if in_band:
                    self.pending[tid] = Candidate(
                        vehicle, vehicle, seconds, packet.sequence, packet.sequence, seconds
                    )
                    self.last_decisions[tid] = (
                        "recovering_motion" if tid in self.motion_recovery else "collecting"
                    )
                else:
                    self.last_decisions[tid] = "outside_band"
                continue
            old_distance, _, _ = self.coordinates(candidate.last)
            first_distance, first_along, _ = self.coordinates(candidate.first)
            progress = distance - first_distance
            height = candidate.first.bbox[3] - candidate.first.bbox[1]
            min_progress = max(2.0, 0.08 * height)
            stationary = hypot(progress, along - first_along) <= min_progress
            issues = [
                box_motion_issue(
                    box, vehicle.bbox, self.start, self.end, self.direction
                )
                for box in (candidate.last.bbox, candidate.first.bbox)
            ]
            # A stationary vehicle does not need evidence of coherent translation.
            # Keep class/size checks; small edge jitter is not a crossing here.
            needs_motion = self.require_forward_motion or tid in self.motion_recovery
            if not needs_motion and stationary:
                issues = [issue for issue in issues if issue != "box_deformation"]
            hard_invalid = (
                packet.sequence <= candidate.last_sequence
                or candidate.first.class_name != vehicle.class_name
                or distance <= 0
                or not 0 <= along <= length
                or any(issue and issue != "box_deformation" for issue in issues)
            )
            if hard_invalid:
                self.pending.pop(tid, None)
                self.eligible.pop(tid, None)
                self.motion_recovery.discard(tid)
                self.last_decisions[tid] = "box_or_direction_rejected"
                continue
            if distance < old_distance - max(2.0, 0.02 * height) or any(issues):
                # Rebase on the current box. The rejected displacement contributes
                # no progress/observations to a future event; fresh motion is required.
                self.pending.pop(tid, None)
                self.motion_recovery.add(tid)
                if in_band:
                    self.pending[tid] = Candidate(
                        vehicle, vehicle, seconds, packet.sequence, packet.sequence, seconds
                    )
                self.last_decisions[tid] = "recovering_motion"
                continue
            candidate.last, candidate.last_sequence = vehicle, packet.sequence
            candidate.last_seconds = seconds
            candidate.count += 1
            forward = progress >= min_progress and progress >= 0.25 * abs(
                along - first_along
            )
            self.last_decisions[tid] = "waiting_motion_or_observations"
            if (candidate.count >= 2 and forward) or (
                candidate.count >= 3 and not needs_motion and stationary
            ):
                events.append(
                    LateViolation(
                        vehicle,
                        {
                            "observed_crossing": False,
                            "first_seconds": candidate.first_seconds,
                            "decision_seconds": seconds,
                            "first_sequence": candidate.first_sequence,
                            "decision_sequence": packet.sequence,
                            "first_bbox": candidate.first.bbox,
                            "first_distance_after_line_px": first_distance,
                            "forward_progress_px": progress,
                            "observations": candidate.count,
                            "red_since_seconds": self.red_since,
                            "motion_basis": "FORWARD"
                            if forward
                            else "STATIONARY_AFTER_LINE",
                            "candidate_basis": self.eligible[tid],
                            "recovered_after_box_rejection": tid in self.motion_recovery,
                        },
                    )
                )
                self.pending.pop(tid, None)
                self.motion_recovery.discard(tid)
                self.suspected.discard(tid)
                self.last_decisions[tid] = "event"
            elif (
                candidate.count >= 3 and stationary and needs_motion
                and tid not in self.motion_recovery and tid not in self.suspected
            ):
                self.suspected.add(tid)
                self.new_suspicions.append(LateSuspicion(vehicle, {
                    "reason": "SUSPECTED_LATE_DIRECTION_UNKNOWN",
                    "observed_crossing": False,
                    "first_seconds": candidate.first_seconds,
                    "decision_seconds": seconds,
                    "observations": candidate.count,
                    "candidate_basis": self.eligible[tid],
                }))
                self.last_decisions[tid] = "direction_unknown_review"
        return events
