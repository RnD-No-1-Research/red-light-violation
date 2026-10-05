"""Đo pipeline thật trên file video; không suy ra accuracy khi chưa có ground truth."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    """Warm-up model rồi đo camera riêng và đồng thời, xuất số liệu thô JSON."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ultralytics-source", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    config_dir = ROOT / "output/.benchmark_ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    if args.ultralytics_source:
        source = args.ultralytics_source.resolve()
        spec = importlib.util.spec_from_file_location(
            "ultralytics",
            source / "__init__.py",
            submodule_search_locations=[str(source)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["ultralytics"] = module
        spec.loader.exec_module(module)

    import cv2
    import numpy as np
    import psutil
    import torch
    import ultralytics
    from controllers.camera_manager import CameraManager
    from controllers.violation_controller import ViolationController
    from models.runtime import create_models
    from utils.config_loader import load_cameras, load_settings
    from utils.video_source import VideoSource
    from views.display import Display

    def digest(path: Path) -> str:
        """Hash tài nguyên để tái lập lần đo."""
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    settings = load_settings(ROOT / "config/settings.yaml")
    settings.update(show_video=False, ocr_download_enabled=False)
    cameras = load_cameras(ROOT / "config/cameras.yaml")
    if any(c.source.lower().startswith(("rtsp://", "rtsps://")) for c in cameras):
        raise ValueError("Benchmark hữu hạn này chỉ nhận file video")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "output" / ("benchmark_" + stamp)
    output.mkdir(parents=True)
    report = {
        "created_utc": stamp,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "ultralytics_source_override": bool(args.ultralytics_source),
        },
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in settings.items()
        },
        "config_text": (ROOT / "config/cameras.yaml").read_text(encoding="utf-8"),
        "settings_text": (ROOT / "config/settings.yaml").read_text(encoding="utf-8"),
        "weights_sha256": {
            k: digest(settings[k])
            for k in ("vehicle_model", "traffic_light_model", "plate_model")
        },
        "videos": {},
        "runs": [],
        "notes": [
            "No ground truth: event counts are predictions, not accuracy.",
            "CPU/GPU auto selection; no GUI; real MP4/JPEG/CSV writes enabled.",
            "Stage timing includes lock wait; do not sum stage percentiles.",
            "Frame latency excludes read/decode, includes pipeline and output write.",
            "Run wall time includes file open/decode, worker startup and writer close.",
            "Peak RSS sampled every 100ms; not system-wide RAM or an exact peak.",
            "Warm-up excluded; per-camera first-frame tracking setup remains included.",
        ],
    }
    initial = {}
    for c in cameras:
        cap = cv2.VideoCapture(c.source)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read {c.camera_id}")
        initial[c.camera_id] = frame
        report["videos"][c.camera_id] = {
            "file": Path(c.source).name,
            "sha256": digest(Path(c.source)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": frame.shape[1],
            "height": frame.shape[0],
        }
        cap.release()
    start = time.perf_counter()
    models = create_models(settings)
    report["load_seconds"] = time.perf_counter() - start
    report["environment"].update(
        device=models.vehicle.device, torch_threads=torch.get_num_threads()
    )

    def sync() -> None:
        """Đợi GPU hoàn tất trước khi đọc đồng hồ."""
        if models.vehicle.device.startswith("cuda"):
            torch.cuda.synchronize()

    # Warm-up trên frame thật; tracker sẽ reset ở đầu mỗi lần chạy controller.
    start = time.perf_counter()
    for c in cameras:
        f = initial[c.camera_id]
        x, y, w, h = c.roi
        for _ in range(3):
            models.traffic.detect(
                f[y : y + h, x : x + w], settings["traffic_light_conf"]
            )
            models.vehicle.track(
                f, c.camera_id, settings["vehicle_conf"], c.detection_polygon
            )
        models.vehicle.drop_camera(c.camera_id)
    models.plate.detect(
        initial[cameras[0].camera_id][:160, :320], settings["plate_conf"]
    )
    models.ocr.read(np.zeros((64, 192, 3), np.uint8))
    sync()
    report["warmup_seconds"] = time.perf_counter() - start

    timings = defaultdict(list)
    events = []
    local = threading.local()

    def timed_method(original: object, name: str) -> object:
        """Bọc lời gọi model, không sửa kết quả hoặc thuật toán."""

        def call(*a: object, **kw: object) -> object:
            sync()
            begin = time.perf_counter()
            result = original(*a, **kw)
            sync()
            timings[(threading.current_thread().name, name)].append(
                time.perf_counter() - begin
            )
            return result

        return call

    for obj, method, name in (
        (models.traffic, "detect", "light"),
        (models.vehicle, "track", "vehicle_tracking"),
        (models.plate, "detect", "plate"),
        (models.ocr, "read", "ocr"),
    ):
        setattr(obj, method, timed_method(getattr(obj, method), name))
    original_read, original_write = VideoSource.read_frame, Display.write_frame
    original_event = ViolationController._save_violation

    def read_frame(source: VideoSource) -> object:
        """Bắt đầu đo frame sau khi decode xong."""
        packet = original_read(source)
        if packet is not None:
            local.frame_start = time.perf_counter()
        return packet

    def write_frame(view: Display, frame: np.ndarray) -> None:
        """Kết thúc đo sau khi ghi frame vào writer."""
        original_write(view, frame)
        sync()
        timings[(threading.current_thread().name, "frame")].append(
            time.perf_counter() - local.frame_start
        )

    def event(
        controller: ViolationController,
        packet: object,
        tracks: list,
        vehicle: object,
        state: str,
    ) -> None:
        """Ghi thời gian nguồn để người đánh giá đối chiếu thủ công."""
        events.append(
            {
                "camera_id": controller.camera.camera_id,
                "source_seconds": packet.source_seconds,
                "track_id": vehicle.track_id,
                "bbox": list(vehicle.bbox),
                "light": state,
            }
        )
        original_event(controller, packet, tracks, vehicle, state)

    VideoSource.read_frame, Display.write_frame = read_frame, write_frame
    ViolationController._save_violation = event

    def stats(values: list) -> dict:
        """Thống kê thời gian bằng mili giây."""
        return {
            "calls": len(values),
            "mean_ms": float(np.mean(values) * 1000),
            "p50_ms": float(np.percentile(values, 50) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
        }

    scenarios = [[c] for c in cameras] + ([cameras] if len(cameras) > 1 else [])
    process = psutil.Process()
    try:
        for group in scenarios:
            label = "+".join(c.camera_id for c in group)
            for repeat in range(1, args.repeats + 1):
                timings.clear()
                events.clear()
                peak = [process.memory_info().rss]
                done = threading.Event()

                def monitor() -> None:
                    """Lấy mẫu RSS của tiến trình benchmark."""
                    while not done.wait(0.1):
                        peak[0] = max(peak[0], process.memory_info().rss)

                sampler = threading.Thread(target=monitor)
                sampler.start()
                run_settings = {
                    **settings,
                    "output_base_dir": output / label / str(repeat),
                }
                print(f"START {label} repeat={repeat}", flush=True)
                begin = time.perf_counter()
                try:
                    code = CameraManager(
                        run_settings, group, models, show_video=False
                    ).run()
                finally:
                    elapsed = time.perf_counter() - begin
                    done.set()
                    sampler.join()
                counts = {
                    c.camera_id: len(timings[("camera-" + c.camera_id, "frame")])
                    for c in group
                }
                row = {
                    "scenario": label,
                    "repeat": repeat,
                    "exit_code": code,
                    "wall_seconds": elapsed,
                    "frames": counts,
                    "fps_total": sum(counts.values()) / elapsed,
                    "fps_per_camera_over_run": {
                        k: v / elapsed for k, v in counts.items()
                    },
                    "peak_rss_bytes": peak[0],
                    "events": list(events),
                    "timings": {
                        cid + "/" + stage: stats(v)
                        for (cid, stage), v in timings.items()
                        if v
                    },
                    "raw_seconds": {
                        cid + "/" + stage: v[:] for (cid, stage), v in timings.items()
                    },
                }
                report["runs"].append(row)
                (output / "results.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(
                    f"DONE {label} repeat={repeat} frames={counts} fps={row['fps_total']:.2f} events={len(events)} code={code}",
                    flush=True,
                )
                if code or any(
                    counts[c.camera_id] != report["videos"][c.camera_id]["frames"]
                    for c in group
                ):
                    raise RuntimeError(
                        "Incomplete run; results are not a successful benchmark"
                    )
    finally:
        VideoSource.read_frame, Display.write_frame = original_read, original_write
        ViolationController._save_violation = original_event
    print(f"REPORT {output / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
