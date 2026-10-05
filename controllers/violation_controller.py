"""Pipeline nghiệp vụ cho một camera, không import thư viện AI trực tiếp."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from models.types import Detection
from utils.config_loader import CameraConfig, late_detection_settings, validate_geometry
from utils.light_state import LightSmoother
from utils.line_crossing import LineCrossing
from utils.late_detection import LateDetection
from utils.occlusion_review import OcclusionReview, ReviewEvent
from utils.plate_format import normalize_plate
from utils.polygon import box_in_polygon
from utils.video_source import FramePacket, VideoSource
from utils.vehicle_anchor import vehicle_anchor
from views.display import Display


def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Cắt bbox vào biên ảnh, trả crop rỗng nếu bbox không hợp lệ."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    return image[
        max(0, y1) : min(height, max(0, y2)), max(0, x1) : min(width, max(0, x2))
    ].copy()


class ViolationController:
    """Điều phối nguồn video, model chung, trạng thái riêng và đầu ra camera."""

    def __init__(
        self,
        camera: CameraConfig,
        settings: dict[str, Any],
        models: Any,
        stop_event: threading.Event,
        frame_callback: Callable[[str, np.ndarray], None] | None = None,
    ) -> None:
        self.camera = camera
        self.settings = {**settings, **camera.overrides}
        self.models = models
        self.stop_event = stop_event
        self.frame_callback = frame_callback
        self.log = logging.LoggerAdapter(
            logging.getLogger(__name__), {"camera_id": camera.camera_id}
        )
        self.smoother = LightSmoother(settings["light_smoothing_window"])
        self.crossing = LineCrossing(camera.point1, camera.point2, camera.direction)
        self.review = OcclusionReview(
            camera.point1, camera.point2, camera.direction,
            self.settings.get("occlusion_review_seconds", 1.0),
        )
        late = late_detection_settings(camera, self.settings)
        self.late = LateDetection(
            camera.point1, camera.point2, camera.direction,
            enabled=late["late_detection_enabled"],
            band_height_ratio=late["late_detection_band_height_ratio"],
            min_red_seconds=late["late_detection_min_red_seconds"],
            require_forward_motion=late["late_detection_require_forward_motion"],
        )
        self.source: VideoSource | None = None
        self.view: Display | None = None
        self.run_id = uuid4().hex[:12]
        self._source_lock = threading.Lock()

    def request_stop(self) -> None:
        """Yêu cầu dừng và đánh thức read_frame đang chờ RTSP."""
        self.stop_event.set()
        with self._source_lock:
            source = self.source
        if source is not None and source.is_rtsp:
            source.release()

    def _save_violation(
        self,
        packet: FramePacket,
        tracks: list[Detection],
        vehicle: Detection,
        state: str,
        reason: str = "DIRECT_LINE_CROSSING",
        details: dict | None = None,
    ) -> None:
        vehicle_crop = crop_image(packet.image, vehicle.bbox)
        plate_crop = None
        plate_text, score = "UNKNOWN", 0.0
        try:
            if vehicle_crop.size:
                plate = self.models.plate.detect(
                    vehicle_crop, self.settings["plate_conf"]
                )
                if plate is not None:
                    candidate = crop_image(vehicle_crop, plate.bbox)
                    if candidate.size:
                        plate_crop = candidate
                        ocr = self.models.ocr.read(candidate)
                        plate_text = normalize_plate(ocr.text)
                        score = ocr.confidence if plate_text != "UNKNOWN" else 0.0
        except Exception as exc:
            self.log.warning(
                "OCR/biển số lỗi (%s); vẫn lưu bằng chứng", type(exc).__name__
            )
        event_id = uuid4().hex
        annotated = self.view.annotate(
            packet.image,
            tracks,
            state,
            packet.timestamp,
            packet.source_seconds,
            highlight=set(self.crossing.recorded),
            suspected=self.review.suspected | self.late.suspected,
            direction_unknown=self.late.suspected,
        )
        self.view.save_evidence(
            event_id,
            vehicle.track_id,
            packet.timestamp,
            plate_text,
            score,
            vehicle.class_name,
            annotated,
            vehicle_crop,
            plate_crop,
            decision={"reason": reason, "source_seconds": packet.source_seconds,
                      "sequence": packet.sequence, **(details or {})},
        )
        self.log.info("Vi phạm track=%d, biển=%s, reason=%s", vehicle.track_id, plate_text, reason)

    def run(self) -> bool:
        """Chạy đến EOF/stop; trả False nếu nguồn video gặp lỗi cuối cùng."""
        camera = self.camera
        settings = self.settings
        try:
            source = VideoSource(
                camera.source,
                camera.camera_id,
                settings["rtsp_retries"],
                settings["rtsp_retry_delay"],
                settings["rtsp_timeout_ms"],
                settings["fallback_fps"],
            )
            with self._source_lock:
                self.source = source
            epoch = -1
            last_size = (0, 0)
            while not self.stop_event.is_set():
                packet = source.read_frame()
                if packet is None:
                    break
                frame = packet.image
                height, width = frame.shape[:2]
                if packet.epoch != epoch or (width, height) != last_size:
                    validate_geometry(camera, (width, height))
                    self.smoother.reset()
                    self.crossing.reset()
                    self.review.reset()
                    self.late.reset()
                    self.models.vehicle.reset_camera(
                        camera.camera_id, settings["vehicle_conf"]
                    )
                    epoch = packet.epoch
                    last_size = (width, height)
                    self.log.info("Bắt đầu phiên %d, frame %dx%d", epoch, width, height)
                if self.view is None:
                    self.view = Display(
                        camera, settings["output_base_dir"], self.run_id
                    )
                self.view.ensure_video((width, height), source.get_fps(), epoch)
                x, y, w, h = camera.roi
                fallback_roi = None
                if camera.traffic_light_fallback_roi is not None:
                    fx, fy, fw, fh = camera.traffic_light_fallback_roi
                    fallback_roi = frame[fy : fy + fh, fx : fx + fw]
                raw_light = self.models.traffic.detect(
                    frame[y : y + h, x : x + w],
                    settings["traffic_light_conf"],
                    canvas_size=camera.traffic_light_canvas_size,
                    fallback_roi=fallback_roi,
                )
                state = self.smoother.update(raw_light)
                if camera.detection_polygon:
                    tracks = self.models.vehicle.track(
                        frame,
                        camera.camera_id,
                        settings["vehicle_conf"],
                        polygon=camera.detection_polygon,
                    )
                else:
                    tracks = self.models.vehicle.track(
                        frame, camera.camera_id, settings["vehicle_conf"]
                    )
                tracks = [
                    track
                    for track in tracks
                    if box_in_polygon(track.bbox, camera.detection_polygon)
                ]
                frame_index = packet.sequence
                rejected_ids = set()
                rejection_reasons = {}
                for vehicle in tracks:
                    point = vehicle_anchor(vehicle.bbox)
                    if self.crossing.update(
                        vehicle.track_id, point, frame_index, state,
                        bbox=vehicle.bbox, class_name=vehicle.class_name,
                    ):
                        self._save_violation(packet, tracks, vehicle, state)
                    elif self.crossing.last_rejection:
                        rejected_ids.add(vehicle.track_id)
                        rejection_reasons[vehicle.track_id] = self.crossing.last_rejection
                        self.log.info(
                            "Bỏ lần cắt vạch không đáng tin track=%d: %s",
                            vehicle.track_id, self.crossing.last_rejection,
                        )
                self.crossing.prune(frame_index)
                for event in self.late.update(
                    packet, tracks, state, raw_light, self.crossing.recorded,
                    rejected_ids=rejected_ids,
                    rejection_reasons=rejection_reasons,
                ):
                    self.crossing.recorded.add(event.vehicle.track_id)
                    self._save_violation(
                        packet, tracks, event.vehicle, state,
                        reason="LATE_DETECTION_AFTER_LINE", details=event.details,
                    )
                for event in self.late.new_suspicions:
                    annotated_review = self.view.annotate(
                        frame, tracks, state, packet.timestamp, packet.source_seconds,
                        highlight=self.crossing.recorded,
                        suspected=self.review.suspected | self.late.suspected,
                        direction_unknown=self.late.suspected,
                    )
                    self.view.save_late_review(uuid4().hex, event, packet, annotated_review)
                    self.log.info(
                        "NGHI VAN track=%d: chưa có bằng chứng hướng đi; không ghi vi phạm",
                        event.vehicle.track_id,
                    )
                self.log.debug(
                    "Late detection source_seconds=%s decisions=%s",
                    packet.source_seconds, self.late.last_decisions,
                )
                for event in self.review.update(
                    packet, tracks, state, raw_light, self.crossing.recorded
                ):
                    self._save_review(packet, tracks, state, event)
                annotated = self.view.annotate(
                    frame,
                    tracks,
                    state,
                    packet.timestamp,
                    packet.source_seconds,
                    set(self.crossing.recorded),
                    suspected=self.review.suspected | self.late.suspected,
                    direction_unknown=self.late.suspected,
                )
                self.view.write_frame(annotated)
                if self.frame_callback is not None:
                    self.frame_callback(camera.camera_id, annotated)
            return source.error is None
        finally:
            if self.source is not None:
                self.source.release()
            if self.view is not None:
                self.view.close()
            self.models.vehicle.drop_camera(camera.camera_id)
            self.review.previous.clear()
            self.late.pending.clear()

    def _save_review(
        self, packet: FramePacket, tracks: list[Detection], state: str,
        event: ReviewEvent,
    ) -> None:
        before = event.before
        before_image = self.view.annotate(
            before.packet.image, [before.vehicle], "RED", before.packet.timestamp,
            before.packet.source_seconds,
        )
        after_image = self.view.annotate(
            packet.image, tracks, state, packet.timestamp, packet.source_seconds,
            highlight=self.crossing.recorded,
            suspected=self.review.suspected | self.late.suspected,
            direction_unknown=self.late.suspected,
        )
        self.view.save_review(
            uuid4().hex, event, packet, before_image, after_image
        )
        self.log.info(
            "NGHI VAN VUOT DEN DO track=%d, gap=%.3fs; cần xem ảnh trước/sau",
            event.after.track_id, event.gap_seconds,
        )
