"""Khởi tạo model một lần và quản lý một worker cho mỗi camera."""

from __future__ import annotations

import logging
import threading
from typing import Any

from controllers.violation_controller import ViolationController
from utils.config_loader import CameraConfig


class CameraManager:
    """Cô lập lỗi camera, chờ EOF hoặc xử lý Ctrl+C và đóng tài nguyên."""

    def __init__(
        self,
        settings: dict[str, Any],
        cameras: list[CameraConfig],
        models: Any = None,
        show_video: bool | None = None,
    ) -> None:
        self.settings = settings
        self.cameras = cameras
        self.models = models
        if show_video is None:
            self.show_video = bool(settings.get("show_video", True))
        else:
            self.show_video = show_video
        self.stop_event = threading.Event()
        self.failures: list[str] = []
        self._lock = threading.Lock()
        self._latest_frames: dict[str, Any] = {}
        self._frame_lock = threading.Lock()

    def _on_frame(self, camera_id: str, frame: Any) -> None:
        with self._frame_lock:
            self._latest_frames[camera_id] = frame

    def _worker(self, controller: ViolationController) -> None:
        cid = controller.camera.camera_id
        try:
            success = controller.run()
            if not success:
                with self._lock:
                    self.failures.append(cid)
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Dừng camera: %s",
                str(exc),
                extra={"camera_id": cid},
            )
            with self._lock:
                self.failures.append(cid)
        finally:
            logging.getLogger(__name__).info(
                "Camera đã kết thúc", extra={"camera_id": cid}
            )

    def run(self) -> int:
        """Chạy các worker; mã 0 thành công, 1 có lỗi camera, 130 khi Ctrl+C."""
        if self.models is None:
            from models.runtime import create_models

            self.models = create_models(self.settings)
        controllers = [
            ViolationController(
                c,
                self.settings,
                self.models,
                self.stop_event,
                frame_callback=self._on_frame if self.show_video else None,
            )
            for c in self.cameras
        ]
        threads = [
            threading.Thread(
                target=self._worker, args=(c,), name=f"camera-{c.camera.camera_id}"
            )
            for c in controllers
        ]
        for thread in threads:
            thread.start()
        interrupted = False
        displayed_windows: set[str] = set()
        try:
            if self.show_video:
                import cv2

                logging.getLogger(__name__).info("Đã bật hiển thị video trực tiếp (nhấn 'q' hoặc 'Esc' trên cửa sổ video để dừng)")
                for c in self.cameras:
                    cv2.namedWindow(f"Camera - {c.camera_id}", cv2.WINDOW_NORMAL)
            while any(thread.is_alive() for thread in threads):
                if self.show_video:
                    import cv2

                    frames_to_show: dict[str, Any] = {}
                    with self._frame_lock:
                        frames_to_show = dict(self._latest_frames)

                    for cid, frame in frames_to_show.items():
                        win_name = f"Camera - {cid}"
                        disp = frame
                        h, w = disp.shape[:2]
                        if w > 1024:
                            scale = 1024 / w
                            disp = cv2.resize(
                                disp,
                                (1024, int(h * scale)),
                                interpolation=cv2.INTER_AREA,
                            )
                        cv2.imshow(win_name, disp)
                        displayed_windows.add(win_name)

                    if displayed_windows and any(
                        cv2.getWindowProperty(w_name, cv2.WND_PROP_VISIBLE) < 1
                        for w_name in displayed_windows
                    ):
                        logging.info("Cửa sổ camera đã đóng; đang dừng hệ thống")
                        self.stop_event.set()
                        for controller in controllers:
                            controller.request_stop()
                        break

                    key = cv2.waitKey(20) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        logging.info(
                            "Nhận phím thoát (q/Esc); đang đóng camera và video"
                        )
                        interrupted = True
                        self.stop_event.set()
                        for controller in controllers:
                            controller.request_stop()
                        break
                else:
                    for thread in threads:
                        thread.join(timeout=0.2)
        except KeyboardInterrupt:
            interrupted = True
            logging.info("Nhận Ctrl+C; đang đóng camera và video")
            self.stop_event.set()
            for controller in controllers:
                controller.request_stop()
        finally:
            if self.show_video:
                import cv2

                cv2.destroyAllWindows()
            for thread in threads:
                thread.join()
        return 130 if interrupted else int(bool(self.failures))
