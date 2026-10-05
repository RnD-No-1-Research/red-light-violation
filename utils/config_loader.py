"""Đọc YAML và kiểm tra cấu hình độc lập với các thư viện AI."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils.polygon import Polygon, parse_polygon, validate_polygon_bounds

ROOT = Path(__file__).resolve().parents[1]
Point = tuple[float, float]
LATE_DETECTION_DEFAULTS = {
    "late_detection_enabled": True,
    "late_detection_band_height_ratio": 1.0,
    "late_detection_min_red_seconds": 1.0,
    "late_detection_require_forward_motion": True,
}


@dataclass(frozen=True)
class CameraConfig:
    """Cấu hình riêng của một camera, tọa độ theo frame gốc."""

    camera_id: str
    name: str
    source: str
    point1: Point
    point2: Point
    roi: tuple[int, int, int, int]
    direction: str = "negative_to_positive"
    enabled: bool = True
    overrides: dict[str, float] = field(default_factory=dict)
    detection_polygon: Polygon = ()
    traffic_light_canvas_size: int = 0
    traffic_light_fallback_roi: tuple[int, int, int, int] | None = None
    # None means inherit the common setting; explicit false still overrides true.
    late_detection_enabled: bool | None = None
    late_detection_band_height_ratio: float | None = None
    late_detection_min_red_seconds: float | None = None
    late_detection_require_forward_motion: bool | None = None


def read_yaml(path: Path) -> dict[str, Any]:
    """Đọc một tài liệu YAML dạng mapping, không thực thi đối tượng."""
    try:
        with path.open(encoding="utf-8-sig") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Sai cú pháp YAML: {path.name}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"YAML phải là mapping: {path.name}")
    return data


def resolve_path(value: str, root: Path = ROOT) -> Path:
    """Giải đường dẫn tương đối theo thư mục project, không theo cwd."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _number(value: Any, label: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} phải là số")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} phải hữu hạn và >= {minimum}")
    return float(value)


def _confidence(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0 < result <= 1:
        raise ValueError(f"{label} phải nằm trong (0, 1]")
    return result


def _late_options(data: dict, *, use_defaults: bool = False) -> dict:
    values = dict(LATE_DETECTION_DEFAULTS) if use_defaults else {}
    values.update({key: data[key] for key in LATE_DETECTION_DEFAULTS if key in data})
    for key in ("late_detection_enabled", "late_detection_require_forward_motion"):
        if key in values and type(values[key]) is not bool:
            raise ValueError(f"{key} phải là boolean")
    for key, minimum in (("late_detection_band_height_ratio", 0.01),
                         ("late_detection_min_red_seconds", 0.1)):
        if key in values:
            values[key] = _number(values[key], key, minimum)
    if values.get("late_detection_band_height_ratio", 1.0) > 2:
        raise ValueError("late_detection_band_height_ratio tối đa 2")
    return values


def late_detection_settings(camera: CameraConfig, settings: dict) -> dict:
    """Resolve camera overrides over common settings, including explicit false."""
    return {
        key: (getattr(camera, key) if getattr(camera, key) is not None
              else settings.get(key, default))
        for key, default in LATE_DETECTION_DEFAULTS.items()
    }


def load_settings(path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Đọc ngưỡng và đường dẫn chung; chưa khởi tạo AI."""
    data = read_yaml(path)
    data.update(_late_options(data, use_defaults=True))
    for key in ("traffic_light_model", "vehicle_model", "plate_model"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Thiếu đường dẫn {key}")
    for key, default in {
        "output_base_dir": "output",
        "ocr_model_dir": "weights/easyocr",
    }.items():
        data.setdefault(key, default)
    for key in (
        "traffic_light_model",
        "vehicle_model",
        "plate_model",
        "output_base_dir",
        "ocr_model_dir",
    ):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{key} phải là đường dẫn không rỗng")
        data[key] = resolve_path(data[key], root)
    for key, default in {
        "traffic_light_conf": 0.5,
        "vehicle_conf": 0.4,
        "plate_conf": 0.5,
    }.items():
        data[key] = _confidence(data.get(key, default), key)
    for key, default in {
        "light_smoothing_window": 5,
        "imgsz": 640,
        "rtsp_retries": 3,
        "rtsp_timeout_ms": 5000,
    }.items():
        value = data.get(key, default)
        if type(value) is not int or value < (0 if key == "rtsp_retries" else 1):
            raise ValueError(f"{key} phải là số nguyên hợp lệ")
        data[key] = value
    if data["rtsp_retries"] > 3:
        raise ValueError("rtsp_retries tối đa 3")
    for key, default in {"rtsp_retry_delay": 5.0, "fallback_fps": 25.0}.items():
        data[key] = _number(data.get(key, default), key, 0.001)
    data.setdefault("device", "auto")
    data["occlusion_review_seconds"] = _number(
        data.get("occlusion_review_seconds", 1.0), "occlusion_review_seconds"
    )
    if data["occlusion_review_seconds"] > 1.0:
        raise ValueError("occlusion_review_seconds tối đa 1 giây; 0 để tắt")
    if data["device"] not in ("auto", "cpu", "cuda", "cuda:0"):
        raise ValueError("device phải là auto, cpu, cuda hoặc cuda:0")
    data.setdefault("ocr_download_enabled", True)
    data.setdefault("show_video", True)
    if type(data["ocr_download_enabled"]) is not bool:
        raise ValueError("ocr_download_enabled phải là boolean")
    data.setdefault(
        "traffic_light_labels", {"red": "RED", "green": "GREEN", "yellow": "YELLOW"}
    )
    labels = data["traffic_light_labels"]
    if not isinstance(labels, dict) or not labels:
        raise ValueError("traffic_light_labels phải là mapping")
    if any(
        not isinstance(k, str) or v not in ("RED", "GREEN", "YELLOW", "UNKNOWN")
        for k, v in labels.items()
    ):
        raise ValueError("traffic_light_labels không hợp lệ")
    return data


def load_cameras(
    path: Path,
    root: Path = ROOT,
    camera_id: str | None = None,
) -> list[CameraConfig]:
    """Validate camera bật; cho phép placeholder trong camera đã tắt."""
    rows = read_yaml(path).get("cameras")
    if not isinstance(rows, list):
        raise ValueError("cameras phải là danh sách")
    ids: set[str] = set()
    result: list[CameraConfig] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Mỗi camera phải là mapping")
        cid = row.get("camera_id")
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", cid):
            raise ValueError("camera_id chỉ được chứa chữ, số, _ và -")
        if cid in ids:
            raise ValueError(f"Trùng camera_id: {cid}")
        ids.add(cid)
        enabled = row.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError(f"[{cid}] enabled phải là true/false")
        if not enabled or (camera_id is not None and cid != camera_id):
            continue
        source = row.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"[{cid}] source phải là chuỗi không rỗng")
        source = os.path.expandvars(source)
        if re.search(r"\$\{[^}]+\}", source):
            raise ValueError(f"[{cid}] Chưa đặt biến môi trường source")
        if not source.lower().startswith(("rtsp://", "rtsps://")):
            if "://" in source:
                raise ValueError(f"[{cid}] Chỉ hỗ trợ file local và RTSP")
            source = str(resolve_path(source, root))
        line = row.get("stop_line", {})
        if not isinstance(line, dict):
            raise ValueError(f"[{cid}] stop_line phải là mapping")
        points: list[Point] = []
        for key in ("point1", "point2"):
            point = line.get(key)
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"[{cid}] {key} cần [x, y]")
            points.append(tuple(_number(v, f"[{cid}] {key}") for v in point))
        if points[0] == points[1]:
            raise ValueError(f"[{cid}] Hai điểm vạch không được trùng nhau")
        roi = row.get("traffic_light_roi")
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError(f"[{cid}] ROI cần [x, y, w, h]")
        if any(type(v) is not int for v in roi):
            raise ValueError(f"[{cid}] ROI phải chứa số nguyên")
        if min(roi[:2]) < 0 or min(roi[2:]) <= 0:
            raise ValueError(f"[{cid}] ROI không hợp lệ")
        canvas_size = row.get("traffic_light_canvas_size", 0)
        if type(canvas_size) is not int or not 0 <= canvas_size <= 4096:
            raise ValueError(f"[{cid}] traffic_light_canvas_size cần số nguyên 0..4096")
        fallback_roi = row.get("traffic_light_fallback_roi")
        if fallback_roi is not None:
            if (
                not isinstance(fallback_roi, list)
                or len(fallback_roi) != 4
                or any(type(v) is not int for v in fallback_roi)
                or min(fallback_roi[:2]) < 0
                or min(fallback_roi[2:]) <= 0
            ):
                raise ValueError(f"[{cid}] traffic_light_fallback_roi không hợp lệ")
        direction = row.get("crossing_direction", "negative_to_positive")
        try:
            late = _late_options(row)
        except ValueError as exc:
            raise ValueError(f"[{cid}] {exc}") from exc
        if direction not in ("negative_to_positive", "positive_to_negative"):
            raise ValueError(f"[{cid}] crossing_direction không hợp lệ")
        overrides = {
            k: _confidence(row[k], f"[{cid}] {k}")
            for k in ("traffic_light_conf", "vehicle_conf", "plate_conf")
            if k in row
        }
        try:
            polygon = parse_polygon(row.get("detection_polygon"))
        except ValueError as exc:
            raise ValueError(f"[{cid}] {exc}") from exc
        result.append(
            CameraConfig(
                cid,
                str(row.get("name", cid)),
                source,
                points[0],
                points[1],
                tuple(roi),
                direction,
                True,
                overrides,
                polygon,
                canvas_size,
                tuple(fallback_roi) if fallback_roi is not None else None,
                late.get("late_detection_enabled"),
                late.get("late_detection_band_height_ratio"),
                late.get("late_detection_min_red_seconds"),
                late.get("late_detection_require_forward_motion"),
            )
        )
    if camera_id is not None and not result:
        raise ValueError(f"[{camera_id}] Camera không tồn tại hoặc đã tắt")
    if not result:
        raise ValueError("Không có camera enabled")
    return result


def validate_geometry(camera: CameraConfig, size: tuple[int, int]) -> None:
    """Kiểm tra vạch và ROI nằm trong frame thực tế của camera."""
    width, height = size
    validate_polygon_bounds(camera.detection_polygon, size)
    for x, y in (camera.point1, camera.point2):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Vạch dừng nằm ngoài frame; hãy vẽ lại")
    x, y, w, h = camera.roi
    if x + w > width or y + h > height:
        raise ValueError("ROI đèn nằm ngoài frame; hãy cấu hình lại")
    if camera.traffic_light_fallback_roi is not None:
        x, y, w, h = camera.traffic_light_fallback_roi
        if x + w > width or y + h > height:
            raise ValueError("ROI đèn dự phòng nằm ngoài frame; hãy cấu hình lại")
