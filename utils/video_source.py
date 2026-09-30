"""Nguồn file tuần tự hoặc RTSP có buffer mới nhất và reconnect."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """Frame cùng thời điểm thu nhận và phiên kết nối."""

    image: np.ndarray
    timestamp: datetime
    sequence: int
    epoch: int
    source_seconds: float | None = None


class VideoSource:
    """Đọc file hoặc RTSP; chỉ luồng producer được thao tác capture RTSP."""

    def __init__(
        self,
        source: str,
        camera_id: str,
        retries: int = 3,
        retry_delay: float = 5.0,
        timeout_ms: int = 5000,
        fallback_fps: float = 25.0,
    ) -> None:
        self.source = source
        self.is_rtsp = source.lower().startswith(("rtsp://", "rtsps://"))
        self.log = logging.LoggerAdapter(
            logging.getLogger(__name__), {"camera_id": camera_id}
        )
        self.retries = retries
        self.retry_delay = retry_delay
        self.timeout_ms = timeout_ms
        self.fallback_fps = fallback_fps
        self._fps = fallback_fps
        self._size = (0, 0)
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._finished = False
        self._sequence = 0
        self._epoch = 0
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        if self.is_rtsp:
            self._thread = threading.Thread(
                target=self._produce,
                name=f"capture-{camera_id}",
                daemon=True,
            )
            self._thread.start()
        else:
            if not Path(source).is_file():
                raise FileNotFoundError(
                    "Không có file video; kiểm tra source trong YAML"
                )
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                self._cap.release()
                raise OSError("Không mở được file video")
            self._read_metadata(self._cap)

    def _read_metadata(self, cap: cv2.VideoCapture) -> None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if math.isfinite(fps) and 0 < fps <= 240 else self.fallback_fps
        self._size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _open_rtsp(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(
            self.source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                self.timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                self.timeout_ms,
            ],
        )

    def _produce(self) -> None:
        failures = 0
        try:
            while not self._stop.is_set():
                cap = self._open_rtsp()
                received = False
                try:
                    if cap.isOpened():
                        self._read_metadata(cap)
                        while not self._stop.is_set():
                            ok, frame = cap.read()
                            if not ok:
                                break
                            self._size = (frame.shape[1], frame.shape[0])
                            if not received:
                                received = True
                                failures = 0
                                self.log.info("RTSP đã kết nối, phiên %d", self._epoch)
                            self._sequence += 1
                            packet = FramePacket(
                                frame,
                                datetime.now(timezone.utc),
                                self._sequence,
                                self._epoch,
                            )
                            with self._condition:
                                self._latest = packet
                                self._condition.notify_all()
                finally:
                    cap.release()
                if self._stop.is_set():
                    break
                with self._condition:
                    self._latest = None
                if failures >= self.retries:
                    self.error = "RTSP không phục hồi sau số lần retry cho phép"
                    self.log.error(self.error)
                    break
                failures += 1
                self._epoch += 1
                self.log.warning(
                    "RTSP gián đoạn; retry %d/%d sau %.1fs",
                    failures,
                    self.retries,
                    self.retry_delay,
                )
                if self._stop.wait(self.retry_delay):
                    break
        except Exception as exc:
            self.error = f"Lỗi đọc RTSP ({type(exc).__name__})"
            self.log.error(self.error)
        finally:
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    def read_frame(self) -> FramePacket | None:
        """Chờ frame mới; trả None khi EOF, lỗi cuối cùng hoặc đã đóng."""
        if self._stop.is_set():
            return None
        if self.is_rtsp:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._latest is not None
                    or self._finished
                    or self._stop.is_set()
                )
                packet, self._latest = self._latest, None
                return packet
        if self._cap is None or self._finished:
            return None
        ok, frame = self._cap.read()
        if not ok:
            self._finished = True
            expected = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if self._sequence == 0 or self._sequence + 2 < expected:
                self.error = "Video rỗng hoặc lỗi giải mã trước EOF"
                self.log.error(self.error)
            return None
        self._sequence += 1
        self._size = (frame.shape[1], frame.shape[0])
        return FramePacket(
            frame,
            datetime.now(timezone.utc),
            self._sequence,
            0,
            (self._sequence - 1) / self._fps,
        )

    def release(self) -> None:
        """Dừng đọc và đóng capture; có thể gọi nhiều lần an toàn."""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.timeout_ms / 1000 + 2)
            if self._thread.is_alive():
                self.log.warning("Backend chưa kết thúc capture sau timeout")
        elif self._cap is not None:
            self._cap.release()

    def is_opened(self) -> bool:
        """Cho biết nguồn vẫn đang hoạt động hoặc đang retry."""
        return not self._stop.is_set() and not self._finished

    def get_fps(self) -> float:
        """Trả FPS từ nguồn, dùng giá trị dự phòng nếu metadata không hợp lệ."""
        return self._fps

    def get_frame_size(self) -> tuple[int, int]:
        """Trả kích thước nguồn theo thứ tự rộng, cao."""
        return self._size
