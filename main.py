"""Điểm vào CLI của hệ thống multi-camera."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from utils.config_loader import ROOT, load_cameras, load_settings


class CameraLogFilter(logging.Filter):
    """Bổ sung camera_id và che URL RTSP trong log Python."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Chuẩn hóa bản ghi trước khi xuất ra console."""
        if not hasattr(record, "camera_id"):
            record.camera_id = "system"
        record.msg = re.sub(r"rtsps?://\S+", "rtsp://***", record.getMessage())
        record.args = ()
        return True


def configure_logging() -> None:
    """Thiết lập định dạng log có camera_id cho toàn project."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    handler = logging.StreamHandler()
    handler.addFilter(CameraLogFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(camera_id)s] %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def main() -> int:
    """Đọc CLI, validate cấu hình và chạy các camera được chọn."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Phát hiện vượt đèn đỏ đa camera")
    parser.add_argument("--camera_id")
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.yaml")
    parser.add_argument("--cameras", type=Path, default=ROOT / "config/cameras.yaml")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Kiểm tra YAML, không mở nguồn hoặc tải model",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hiển thị cửa sổ video trực tiếp (mặc định: bật, dùng --no-show để tắt)",
    )
    args = parser.parse_args()
    (ROOT / "output/.ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "output/.ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_VERBOSE", "false")
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    try:
        settings = load_settings(args.settings)
        cameras = load_cameras(args.cameras, camera_id=args.camera_id)
        if args.check_config:
            logging.info("YAML hợp lệ: %s", ", ".join(c.camera_id for c in cameras))
            return 0
        from controllers.camera_manager import CameraManager

        return CameraManager(settings, cameras, show_video=args.show).run()
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.info("Đã hủy khởi tạo")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
