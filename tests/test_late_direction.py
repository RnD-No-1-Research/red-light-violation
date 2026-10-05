"""Unknown or lateral direction must not be promoted by stationary presence."""

import csv
import threading
from dataclasses import replace

from models.types import Detection
from controllers.violation_controller import ViolationController
from tests.test_multi_camera import FakeVehicle, bundle
from tests.test_occlusion_review import packet
from utils.late_detection import LateDetection


def test_cam01_stationary_wrong_direction_is_only_review():
    d = LateDetection((1453, 804), (5, 919), "negative_to_positive", enabled=True)
    d.update(packet(1, 0), [], "RED", "RED", set())
    boxes = [(1158, 518, 1303, 729), (1157, 516, 1303, 728), (1157, 515, 1303, 728)]
    reviews = []
    for i, box in enumerate(boxes):
        assert not d.update(
            packet(172 + i, 56.7 + i / 3),
            [Detection(box, 0.9, "motorcycle", 201)],
            "RED",
            "RED",
            set(),
        )
        reviews.extend(d.new_suspicions)
    assert len(reviews) == 1 and 201 in d.suspected
    assert reviews[0].details["observed_crossing"] is False


def test_review_once_then_promote_only_on_forward_motion():
    d = LateDetection((0, 60), (160, 60), "positive_to_negative", enabled=True)
    d.update(packet(1, 0), [], "RED", "RED", set())
    reviews = []
    for i in range(8):
        assert not d.update(
            packet(2 + i, 2 + i / 6),
            [Detection((50, 30, 80, 70), 0.9, "motorcycle", 1)],
            "RED",
            "RED",
            set(),
        )
        reviews.extend(d.new_suspicions)
    assert len(reviews) == 1
    events = d.update(
        packet(10, 3.4),
        [Detection((50, 20, 80, 60), 0.9, "motorcycle", 1)],
        "RED",
        "RED",
        set(),
    )
    assert len(events) == 1 and events[0].details["motion_basis"] == "FORWARD"
    assert 1 not in d.suspected
    d.reset()
    assert not d.suspected and not d.new_suspicions


def test_lateral_vehicle_stops_or_returns_with_new_id_never_confirms():
    d = LateDetection((0, 60), (160, 60), "positive_to_negative", enabled=True)
    d.update(packet(1, 0), [], "RED", "RED", set())
    for i, x in enumerate((40, 60, 80, 80, 80, 80, 80, 80)):
        tid = 1 if i < 4 else 2
        assert not d.update(
            packet(i + 2, 2 + i / 6),
            [Detection((x - 10, 30, x + 10, 70), 0.9, "motorcycle", tid)],
            "RED",
            "RED",
            set(),
        )
    assert 2 in d.suspected


def test_controller_saves_direction_review_without_ocr_or_violation(camera, settings):
    class Stationary(FakeVehicle):
        def track(self, frame, camera_id, confidence):
            index = self.counts[camera_id]
            self.counts[camera_id] += 1
            return (
                []
                if index < 2
                else [Detection((50, 50, 80, 95), 0.9, "motorcycle", 12)]
            )

    class NoPlate:
        def detect(self, *args):
            raise AssertionError("Direction-only review must not run plate inference")

    models = bundle()
    models.vehicle = Stationary()
    models.plate = NoPlate()
    config = replace(camera, late_detection_min_red_seconds=0.1)
    frames = []
    controller = ViolationController(
        config, settings, models, threading.Event(), lambda cid, f: frames.append(f)
    )
    assert controller.run()
    folder = settings["output_base_dir"] / camera.camera_id
    assert (
        list(csv.DictReader((folder / "violations.csv").open(encoding="utf-8"))) == []
    )
    rows = list(
        csv.DictReader((folder / "late_review_events.csv").open(encoding="utf-8"))
    )
    assert len(rows) == 1 and rows[0]["status"] == "SUSPECTED_LATE_DIRECTION_UNKNOWN"
    assert (folder / rows[0]["evidence_path"]).is_file()
    assert tuple(frames[-1][90, 50]) == (0, 165, 255)
