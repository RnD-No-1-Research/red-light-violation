"""A noisy crossing can recover only from fresh, coherent forward motion."""

import pytest

from models.types import Detection
from tests.test_occlusion_review import packet
from utils.late_detection import LateDetection
from utils.line_crossing import LineCrossing
from utils.vehicle_anchor import vehicle_anchor


@pytest.mark.parametrize(
    "tid,boxes,decision_index",
    [
        # User's 300: merged box shrinks at crossing, then the vehicle moves.
        (63, [(368, 247, 428, 375), (367, 245, 426, 375),
              (365, 243, 424, 373), (368, 241, 417, 348),
              (369, 238, 409, 324), (368, 235, 405, 314)], 5),
        # User's 417: a transient expansion must not blacklist the ID forever.
        (77, [(14, 230, 60, 316), (18, 228, 66, 317), (25, 226, 73, 314),
              (27, 223, 85, 331), (29, 220, 94, 341), (40, 219, 95, 321),
              (50, 217, 97, 305), (57, 215, 101, 296)], 7),
        # User's 431: still rejected at 33.075; confirms after further observations.
        (84, [(31, 237, 89, 361), (44, 238, 97, 349), (58, 236, 111, 347),
              (77, 236, 118, 321), (87, 231, 126, 311)], 4),
    ],
)
def test_real_cam03_box_recovery(tid, boxes, decision_index):
    start, end, direction = (27, 335), (545, 359), "positive_to_negative"
    late = LateDetection(start, end, direction, enabled=True, require_forward_motion=False)
    direct = LineCrossing(start, end, direction)
    late.update(packet(1, 0), [], "RED", "RED", set())
    events = []
    for i, box in enumerate(boxes):
        v = Detection(box, .8, "motorcycle", tid)
        assert not direct.update(tid, vehicle_anchor(box), i + 2, "RED", bbox=box, class_name=v.class_name)
        rejected = {tid: direct.last_rejection} if direct.last_rejection else {}
        found = late.update(packet(i + 2, 2 + i / 6), [v], "RED", "RED", direct.recorded,
                            rejected_ids=set(rejected), rejection_reasons=rejected)
        if i < decision_index:
            assert not found
        for e in found:
            events.append(e)
            direct.recorded.add(tid)
    assert len(events) == 1
    assert events[0].details["recovered_after_box_rejection"]
    assert events[0].details["motion_basis"] == "FORWARD"
    assert events[0].details["observed_crossing"] is False
    assert events[0].details["decision_sequence"] == decision_index + 2


def test_stationary_shrunken_car_never_uses_stationary_recovery():
    late = LateDetection((0, 60), (160, 60), "positive_to_negative",
                         enabled=True, require_forward_motion=False)
    late.update(packet(1, 0), [], "RED", "RED", set())
    before = Detection((50, 10, 100, 85), .9, "car", 1)
    shrunken = Detection((50, 10, 100, 75), .9, "car", 1)
    late.update(packet(2, 1.1), [before], "RED", "RED", set())
    assert not late.update(packet(3, 1.2), [shrunken], "RED", "RED", set(),
                           rejected_ids={1}, rejection_reasons={1:"box_deformation"})
    # Includes expiry/reacquisition and small bbox jitter; no coherent translation.
    for i in range(40):
        v = Detection((50, 10, 100, 75 + i % 2), .9, "car", 1)
        assert not late.update(packet(4+i, 1.3+i/6), [v], "RED", "RED", set())
    assert 1 in late.motion_recovery


@pytest.mark.parametrize("reason", ["class_changed", "box_size_jump"])
def test_hard_rejections_do_not_gain_recovery(reason):
    late = LateDetection((0, 60), (160, 60), "positive_to_negative",
                         enabled=True, require_forward_motion=False)
    late.update(packet(1, 0), [], "RED", "RED", set())
    late.update(packet(2, 1.1), [Detection((50, 40, 100, 85), .9, "car", 1)], "RED", "RED", set())
    late.update(packet(3, 1.2), [Detection((50, 10, 100, 75), .9, "car", 1)], "RED", "RED", set(),
                rejected_ids={1}, rejection_reasons={1:reason})
    for i in range(3):
        assert not late.update(packet(4+i, 1.3+i/6),
                               [Detection((50, 5-i*5, 100, 70-i*5), .9, "car", 1)], "RED", "RED", set())
    assert 1 not in late.eligible and 1 not in late.motion_recovery
