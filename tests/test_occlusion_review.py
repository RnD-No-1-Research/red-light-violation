"""Gap recovery must produce review evidence, never a confirmed violation."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from models.types import Detection
from utils.occlusion_review import OcclusionReview
from utils.video_source import FramePacket


def packet(index, seconds, epoch=0, rtsp=False):
    return FramePacket(
        np.zeros((120, 160, 3), np.uint8),
        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
        index,
        epoch,
        None if rtsp else seconds,
    )


def vehicle(bottom, tid=82, x=50, name="motorcycle", width=30):
    # Preserve the test's decision-point coordinates with the 20 px anchor offset.
    bottom += 20
    return Detection(
        (x - width // 2, bottom - 40, x + width // 2, bottom), 0.9, name, tid
    )


def review():
    return OcclusionReview((0, 60), (159, 60), "positive_to_negative")


@pytest.mark.parametrize("rtsp", [False, True])
def test_gap_recovers_as_suspect_once(rtsp):
    r = review()
    assert not r.update(packet(1, 0, rtsp=rtsp), [vehicle(80)], "RED", "RED", set())
    for i in range(2, 5):
        assert not r.update(packet(i, (i - 1) / 6, rtsp=rtsp), [], "RED", "RED", set())
    events = r.update(packet(5, 4 / 6, rtsp=rtsp), [vehicle(50)], "RED", "RED", set())
    assert len(events) == 1
    assert events[0].before.packet.sequence == 1
    assert events[0].gap_seconds == pytest.approx(4 / 6, abs=1e-6)
    assert r.suspected == {82}
    assert not r.update(packet(6, 0.9, rtsp=rtsp), [vehicle(40)], "RED", "RED", set())


@pytest.mark.parametrize(
    "light,raw",
    [("UNKNOWN", "RED"), ("RED", "UNKNOWN"), ("GREEN", "GREEN"), ("RED", "YELLOW")],
)
def test_any_non_red_during_gap_blocks(light, raw):
    r = review()
    r.update(packet(1, 0), [vehicle(80)], "RED", "RED", set())
    r.update(packet(2, 0.2), [], light, raw, set())
    assert not r.update(packet(5, 0.7), [vehicle(50)], "RED", "RED", set())


@pytest.mark.parametrize(
    "after,index,seconds,epoch",
    [
        (vehicle(50), 2, 0.2, 0),  # no gap: normal crossing owns this
        (vehicle(50), 5, 1.01, 0),
        (vehicle(50), 5, 0.7, 1),
        (vehicle(50), 5, -0.1, 0),
        (vehicle(50, tid=83), 5, 0.7, 0),
        (vehicle(50, name="car"), 5, 0.7, 0),
        (vehicle(50, width=100), 5, 0.7, 0),
        (vehicle(50, x=200), 5, 0.7, 0),
        (vehicle(90), 5, 0.7, 0),  # retreat, still before
    ],
)
def test_reject_invalid_recovery(after, index, seconds, epoch):
    r = review()
    r.update(packet(1, 0), [vehicle(80)], "RED", "RED", set())
    assert not r.update(packet(index, seconds, epoch), [after], "RED", "RED", set())


def test_finite_segment_and_reverse_direction():
    r = review()
    r.update(packet(1, 0), [vehicle(80, x=155)], "RED", "RED", set())
    assert not r.update(packet(5, 0.7), [vehicle(50, x=180)], "RED", "RED", set())
    r = review()
    r.update(packet(1, 0), [vehicle(50)], "RED", "RED", set())
    assert not r.update(packet(5, 0.7), [vehicle(80)], "RED", "RED", set())


def test_history_shared_pruned_and_disabled():
    r = review()
    p = packet(1, 0)
    r.update(p, [vehicle(80), vehicle(85, tid=83)], "RED", "RED", set())
    assert r.retained_image_bytes() == p.image.nbytes
    r.update(packet(10, 1.1), [], "RED", "RED", set())
    assert r.retained_image_bytes() == 0
    r.max_gap_seconds = 0
    r.update(packet(11, 1.2), [vehicle(80)], "RED", "RED", set())
    assert not r.previous


def test_confirmed_track_excluded_and_reset_clears_suspect():
    r = review()
    r.update(packet(1, 0), [vehicle(80)], "RED", "RED", set())
    assert not r.update(packet(5, 0.7), [vehicle(50)], "RED", "RED", {82})
    r.suspected.add(82)
    r.reset()
    assert not r.suspected and not r.previous
