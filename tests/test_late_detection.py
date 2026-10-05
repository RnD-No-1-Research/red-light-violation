"""New IDs behind the line follow an explicit per-camera inference policy."""

from dataclasses import replace
import csv
import json
import threading

import pytest

from controllers.violation_controller import ViolationController
from models.types import Detection
from tests.test_multi_camera import bundle, FakeVehicle
from tests.test_occlusion_review import packet
from utils.late_detection import LateDetection


def vehicle(bottom=55, x=70, tid=10):
    # Argument denotes the decision point, which is 20 px above the bbox bottom.
    bottom += 20
    return Detection((x - 15, bottom - 40, x + 15, bottom), 0.9, "motorcycle", tid)


def ready(enabled=True):
    d = LateDetection((0, 60), (160, 60), "positive_to_negative", enabled=enabled)
    d.update(packet(1, 0), [], "RED", "RED", set())
    d.update(packet(2, 1), [], "RED", "RED", set())
    return d


def test_late_appearance_red_and_once():
    d = ready()
    assert not d.update(packet(3, 1.1), [vehicle()], "RED", "RED", set())
    events = d.update(packet(4, 1.2), [vehicle(51)], "RED", "RED", set())
    assert len(events) == 1 and events[0].details["observed_crossing"] is False
    assert events[0].details["observations"] == 2
    assert not d.update(packet(6, 1.4), [vehicle(43)], "RED", "RED", {10})


@pytest.mark.parametrize(
    "first,next_box,last",
    [
        (vehicle(55), vehicle(55), vehicle(55)),  # stationary
        (vehicle(55), vehicle(57), vehicle(59)),  # wrong way
        (vehicle(55), vehicle(54, x=100), vehicle(53, x=130)),  # lateral traffic
        (vehicle(55, x=180), vehicle(51, x=180), vehicle(47, x=180)),
        (vehicle(10), vehicle(6), vehicle(2)),  # outside first-appearance band
    ],
)
def test_no_event_for_exclusions(first, next_box, last):
    d = ready()
    assert not d.update(packet(3, 1.1), [first], "RED", "RED", set())
    assert not d.update(packet(4, 1.2), [next_box], "RED", "RED", set())
    assert not d.update(packet(5, 1.3), [last], "RED", "RED", set())


@pytest.mark.parametrize(
    "light,raw",
    [("UNKNOWN", "RED"), ("RED", "UNKNOWN"), ("GREEN", "GREEN"), ("YELLOW", "YELLOW")],
)
def test_light_change_cancels_without_retroactive_event(light, raw):
    d = ready()
    d.update(packet(3, 1.1), [vehicle()], "RED", "RED", set())
    assert not d.update(packet(4, 1.2), [vehicle(51)], light, raw, set())
    assert not d.update(packet(5, 1.3), [vehicle(47)], "RED", "RED", set())
    # UNKNOWN discards the old candidate; recovery may start a fresh observation.
    assert all(c.count == 1 for c in d.pending.values())
    if light in ("GREEN", "YELLOW"):
        assert not d.pending and 10 not in d.eligible


def test_startup_reconnect_disabled():
    for enabled, epoch, sequence, seconds in [
        (False, 0, 4, 1.2),
        (True, 1, 4, 1.2),
    ]:
        d = ready(enabled)
        d.update(packet(3, 1.1), [vehicle()], "RED", "RED", set())
        assert not d.update(
            packet(sequence, seconds, epoch), [vehicle(50)], "RED", "RED", set()
        )
        assert not d.pending
    d = LateDetection((0, 60), (160, 60), "positive_to_negative", enabled=True)
    for i in range(5):
        assert not d.update(
            packet(i + 1, i * 0.4), [vehicle(55 - i * 3)], "RED", "RED", set()
        )


def test_missing_source_frames_allow_short_gap_but_restart_long_gap():
    d = ready()
    d.update(packet(3, 1.1), [vehicle()], "RED", "RED", set())
    d.update(packet(4, 1.4), [], "RED", "RED", set())
    events = d.update(packet(5, 1.7), [vehicle(50)], "RED", "RED", set())
    assert len(events) == 1 and events[0].details["observations"] == 2
    d = ready()
    d.update(packet(3, 1.1), [vehicle()], "RED", "RED", set())
    assert not d.update(packet(6, 2.2), [vehicle(50)], "RED", "RED", set())
    assert d.pending[10].count == 1
    assert d.pending[10].first_seconds == 2.2


def test_recovered_before_line_track_needs_new_after_line_observations():
    d = ready()
    d.update(packet(3, 1.1), [vehicle(70)], "RED", "RED", set())
    assert not d.update(packet(5, 1.4), [vehicle(55)], "RED", "RED", set())
    events = d.update(packet(6, 1.6), [vehicle(50)], "RED", "RED", set())
    assert len(events) == 1
    assert events[0].details["candidate_basis"] == "PREVIOUSLY_BEFORE_LINE"
    assert events[0].details["observed_crossing"] is False


def test_raw_red_warmup_does_not_blacklist_new_id():
    d = LateDetection((0, 60), (160, 60), "positive_to_negative", enabled=True)
    d.update(packet(1, 0), [], "UNKNOWN", "RED", set())
    assert not d.update(packet(2, 0.4), [vehicle()], "UNKNOWN", "RED", set())
    assert not d.update(packet(3, 1.1), [vehicle(52)], "RED", "RED", set())
    events = d.update(packet(4, 1.4), [vehicle(47)], "RED", "RED", set())
    assert len(events) == 1
    assert events[0].details["red_since_seconds"] == 0


def test_known_green_while_missing_prevents_carrying_eligibility_into_next_red():
    d = ready()
    d.update(packet(3, 1.1), [vehicle()], "RED", "RED", set())
    d.update(packet(4, 1.2), [], "GREEN", "GREEN", set())
    d.update(packet(5, 1.3), [], "RED", "RED", set())
    for i in range(5):
        assert not d.update(packet(6 + i, 2.4 + i / 10), [vehicle(50 - i * 4)], "RED", "RED", set())
    assert 10 not in d.eligible


def test_rejected_direct_crossing_cannot_be_promoted_by_late_policy():
    d = ready()
    d.require_forward_motion = False
    d.update(packet(3, 1.1), [vehicle(70)], "RED", "RED", set())
    assert not d.update(packet(4, 1.2), [vehicle(55)], "RED", "RED", set(), rejected_ids={10})
    for i in range(5):
        assert not d.update(packet(5 + i, 1.3 + i / 10), [vehicle(55)], "RED", "RED", set())
    assert 10 not in d.eligible


def test_no_bbox_deformation_or_changed_class():
    for after in (
        Detection((55, 35, 85, 67), 0.9, "motorcycle", 10),
        Detection((55, 27, 85, 67), 0.9, "car", 10),
    ):
        d = ready()
        d.update(packet(3, 1.1), [vehicle()], "RED", "RED", set())
        assert not d.update(packet(4, 1.2), [after], "RED", "RED", set())
        if after.class_name == "motorcycle":
            assert d.pending[10].count == 1 and 10 in d.motion_recovery
        else:
            assert not d.pending and 10 not in d.eligible


def test_stationary_policy_matches_blue_helmet_but_not_startup_vehicle():
    d = LateDetection(
        (34, 335),
        (542, 362),
        "positive_to_negative",
        enabled=True,
        require_forward_motion=False,
    )
    d.update(packet(1, 0), [], "RED", "RED", set())
    d.update(packet(2, 1), [], "RED", "RED", set())
    boxes = [(460, 249, 496, 328), (460, 250, 496, 328), (460, 252, 496, 327)]
    events = []
    for i, box in enumerate(boxes):
        events = d.update(
            packet(103 + i, 16.95 + i / 6),
            [Detection(box, 0.6, "motorcycle", 61)],
            "RED",
            "RED",
            set(),
        )
    assert len(events) == 1
    assert events[0].details["motion_basis"] == "STATIONARY_AFTER_LINE"
    assert events[0].vehicle.track_id == 61
    d.reset()
    for i in range(20):
        assert not d.update(
            packet(i + 1, i / 6),
            [Detection(boxes[0], 0.6, "motorcycle", 61)],
            "RED",
            "RED",
            set(),
        )


def test_stationary_policy_still_rejects_lateral_motion():
    d = ready()
    d.require_forward_motion = False
    for i, x in enumerate((70, 95, 120)):
        assert not d.update(
            packet(3 + i, 1.1 + i / 10), [vehicle(55, x=x)], "RED", "RED", set()
        )


def test_controller_saves_red_violation_with_reason(camera, settings):
    class LateVehicle(FakeVehicle):
        def track(self, frame, camera_id, confidence):
            index = self.counts[camera_id]
            self.counts[camera_id] += 1
            return [] if index < 2 else [vehicle(70 + (index - 2) * 5)]

    models = bundle()
    models.vehicle = LateVehicle()
    config = replace(
        camera, late_detection_enabled=True, late_detection_min_red_seconds=0.1
    )
    frames = []
    controller = ViolationController(
        config, settings, models, threading.Event(), lambda cid, f: frames.append(f)
    )
    assert controller.run()
    folder = settings["output_base_dir"] / camera.camera_id
    with (folder / "violations.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["track_id"] == "10"
    metadata = json.loads(
        (folder / "evidence" / f"{rows[0]['id']}_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["reason"] == "LATE_DETECTION_AFTER_LINE"
    assert metadata["observed_crossing"] is False
    assert not (folder / "review_events.csv").exists()
    assert 10 in controller.crossing.recorded
    assert tuple(frames[-1][75, 55]) == (0, 0, 255)
