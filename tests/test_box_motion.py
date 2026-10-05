"""Regression for car #3 swallowing a motorcycle in cam_04."""

import pytest

from utils.box_motion import box_motion_issue
from utils.line_crossing import LineCrossing
from utils.occlusion_review import OcclusionReview
from tests.test_occlusion_review import packet
from models.types import Detection


LINE = ((193, 369), (638, 374))


def point(box):
    return ((box[0] + box[2]) / 2, float(box[3]))


@pytest.mark.parametrize(
    "tid,before,after,expected",
    [
        (3, (510, 270, 612, 376), (515, 270, 609, 367), False),
        (56, (439, 304, 476, 385), (433, 292, 468, 368), True),
        (84, (417, 297, 455, 375), (419, 295, 456, 370), True),
        (85, (245, 291, 282, 371), (247, 287, 282, 363), True),
        (86, (210, 296, 245, 371), (212, 292, 245, 366), True),
        (83, (286, 296, 322, 373), (288, 294, 323, 370), True),
        (91, (229, 295, 265, 374), (235, 292, 271, 369), True),
        (113, (313, 286, 415, 376), (318, 281, 418, 369), True),
    ],
)
def test_observed_cam04_crossings(tid, before, after, expected):
    detector = LineCrossing(*LINE, "positive_to_negative")
    detector.update(tid, point(before), 1, "RED", bbox=before, class_name="car")
    assert (
        detector.update(tid, point(after), 2, "RED", bbox=after, class_name="car")
        == expected
    )
    assert (tid in detector.recorded) == expected
    if not expected:
        assert detector.last_rejection == "box_deformation"
        assert not detector.update(
            tid, point(after), 3, "RED", bbox=after, class_name="car"
        )


def test_identity_and_size_jump_rejected():
    d = LineCrossing((0, 60), (200, 60), "positive_to_negative")
    before, after = (30, 30, 70, 80), (30, 0, 70, 50)
    d.update(1, point(before), 1, "RED", bbox=before, class_name="car")
    assert not d.update(1, point(after), 2, "RED", bbox=after, class_name="motorcycle")
    assert d.last_rejection == "class_changed"
    assert (
        box_motion_issue(
            before, (20, 0, 160, 50), (0, 60), (200, 60), "positive_to_negative"
        )
        == "box_size_jump"
    )


def test_towards_camera_and_diagonal_motion_allowed():
    assert (
        box_motion_issue(
            (30, 0, 70, 50), (25, 5, 75, 65), (0, 60), (200, 60), "negative_to_positive"
        )
        is None
    )
    # Horizontal translation across a diagonal line also moves edge midpoints
    # along its normal, even with unchanged y coordinates.
    assert (
        box_motion_issue(
            (50, 0, 70, 40), (30, 0, 50, 40), (0, 0), (100, 100), "negative_to_positive"
        )
        is None
    )


def test_gap_review_cannot_reintroduce_deformed_car():
    r = OcclusionReview(*LINE, "positive_to_negative")
    before = Detection((510, 270, 612, 376), 0.8, "car", 3)
    after = Detection((515, 270, 609, 367), 0.8, "car", 3)
    r.update(packet(1, 0), [before], "RED", "RED", set())
    assert not r.update(packet(5, 0.7), [after], "RED", "RED", set())
    assert not r.suspected


def test_touch_uses_box_before_contact():
    d = LineCrossing((0, 60), (200, 60), "positive_to_negative")
    for index, box in enumerate([(30, 20, 70, 70), (30, 10, 70, 60)], 1):
        assert not d.update(1, point(box), index, "RED", bbox=box, class_name="car")
    box = (30, 5, 70, 55)
    assert d.update(1, point(box), 3, "RED", bbox=box, class_name="car")
