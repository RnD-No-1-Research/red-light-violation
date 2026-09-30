"""Đọc file thật và mô phỏng lỗi RTSP không cần camera mạng."""

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from utils.video_source import VideoSource


def test_file_eof_and_release(video: Path) -> None:
    """Đọc đủ frame theo thứ tự, EOF không gây reconnect."""
    source = VideoSource(str(video), "file")
    frames = []
    try:
        while (packet := source.read_frame()) is not None:
            frames.append(packet)
        assert len(frames) == 6
        assert frames[0].source_seconds == 0
        assert frames[-1].source_seconds == pytest.approx(0.5)
        assert source.error is None
        assert not source.is_opened()
    finally:
        source.release()
        source.release()


def test_missing_file(tmp_path: Path) -> None:
    """Nguồn local không tồn tại báo lỗi trước khi gọi decoder."""
    with pytest.raises(FileNotFoundError):
        VideoSource(str(tmp_path / "missing.mp4"), "missing")


class DeadCapture:
    """Capture giả luôn không kết nối được."""

    def isOpened(self) -> bool:
        """Trả trạng thái kết nối thất bại."""
        return False

    def release(self) -> None:
        """Không có tài nguyên native cần đóng."""
        return None


def test_rtsp_retry_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Một lần mở đầu và ba retry rồi kết thúc."""
    calls = []

    def open_fake(self: VideoSource) -> DeadCapture:
        """Đếm số lần cố kết nối."""
        calls.append(1)
        return DeadCapture()

    monkeypatch.setattr(VideoSource, "_open_rtsp", open_fake)
    source = VideoSource("rtsp://example.invalid/stream", "rtsp", retry_delay=0.001)
    try:
        assert source.read_frame() is None
        assert len(calls) == 4
        assert source.error is not None
    finally:
        source.release()


def test_latest_buffer_no_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buffer chỉ giữ frame mới nhất, không trả lặp lại một frame."""
    produced = threading.Event()
    allow_exit = threading.Event()

    class LiveCapture(DeadCapture):
        """Sinh ba frame rồi chờ test đóng nguồn."""

        def __init__(self) -> None:
            self.count = 0

        def isOpened(self) -> bool:
            """Mô phỏng kết nối thành công."""
            return True

        def get(self, prop: int) -> float:
            """Trả metadata giả phù hợp cho test."""
            return 25.0

        def read(self) -> tuple[bool, Any]:
            """Xuất frame hữu hạn để kiểm tra producer-consumer."""
            self.count += 1
            if self.count <= 3:
                return True, np.full((10, 10, 3), self.count, dtype=np.uint8)
            produced.set()
            allow_exit.wait(timeout=3)
            return False, None

    monkeypatch.setattr(VideoSource, "_open_rtsp", lambda self: LiveCapture())
    source = VideoSource("rtsp://example.invalid/stream", "buffer", retries=0)
    try:
        assert produced.wait(timeout=2)
        assert source.read_frame().sequence == 3
        allow_exit.set()
        assert source.read_frame() is None
    finally:
        allow_exit.set()
        source.release()


def test_release_interrupts_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop đánh thức consumer và hủy khoảng chờ reconnect 5 giây."""
    attempted = threading.Event()

    def dead_open(self: VideoSource) -> DeadCapture:
        """Báo đã thử mở để test đồng bộ mà không sleep cố định."""
        attempted.set()
        return DeadCapture()

    monkeypatch.setattr(VideoSource, "_open_rtsp", dead_open)
    source = VideoSource("rtsp://example.invalid/stream", "stop", retry_delay=5)
    result = []
    consumer = threading.Thread(target=lambda: result.append(source.read_frame()))
    consumer.start()
    assert attempted.wait(timeout=2)
    start = time.monotonic()
    source.release()
    consumer.join(timeout=2)
    assert not consumer.is_alive()
    assert time.monotonic() - start < 2
    assert result == [None]
