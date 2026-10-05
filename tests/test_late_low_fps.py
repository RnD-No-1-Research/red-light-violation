"""Regression trajectories from cam_01; no model or camera-specific code paths."""

import pytest

from models.types import Detection
from tests.test_occlusion_review import packet
from utils.late_detection import LateDetection
from utils.vehicle_anchor import vehicle_anchor


@pytest.mark.parametrize(
    "tid,boxes",
    [
        (213, [(905, 605, 1029, 785), (872, 581, 1009, 781), (847, 557, 989, 766)]),
        (198, [(1232, 601, 1367, 792), (1216, 598, 1350, 787),
               (1193, 587, 1331, 783), (1171, 582, 1314, 786),
               (1154, 579, 1296, 782), (1112, 541, 1274, 772)]),
    ],
)
def test_diagonal_low_fps_trajectory(tid, boxes):
    d = LateDetection((1421, 788), (3, 916), "negative_to_positive",
                      enabled=True, require_forward_motion=False)
    d.update(packet(1, 0), [], "RED", "RED", set())
    recorded, events = set(), []
    for i, box in enumerate(boxes):
        found = d.update(packet(i + 2, 2 + i / 3.015846),
                         [Detection(box, .8, "motorcycle", tid)], "RED", "RED", recorded)
        events.extend(found)
        recorded.update(e.vehicle.track_id for e in found)
    assert len(events) == 1
    assert events[0].details["motion_basis"] == "FORWARD"
    assert events[0].details["observations"] == len(boxes)
    if tid == 198:
        assert events[0].details["decision_seconds"] - events[0].details["first_seconds"] > 1


def test_selected_anchor_is_shared_by_late_and_review():
    from utils.occlusion_review import anchor
    v = Detection((10, 20, 30, 70), .9, "car", 1)
    assert vehicle_anchor(v.bbox) == anchor(v) == (20, 50)
    d = LateDetection((0, 60), (160, 60), "positive_to_negative")
    assert d.coordinates(v)[0] == 10
