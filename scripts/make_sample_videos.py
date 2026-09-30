"""Sinh hai video đồ họa để kiểm tra I/O; không dùng đánh giá AI."""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Tạo clip hữu hạn để thử hai camera bằng model thật."""
    directory = ROOT / "data/videos"
    directory.mkdir(parents=True, exist_ok=True)
    for camera_id in ("smoke_01", "smoke_02"):
        path = directory / f"{camera_id}.mp4"
        if path.exists():
            print(f"[{camera_id}] Already exists; keep {path.name}")
            continue
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240)
        )
        if not writer.isOpened():
            raise OSError("Không mở được MP4 writer")
        try:
            for index in range(20):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "SYNTHETIC I/O TEST",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                cv2.rectangle(
                    frame,
                    (100, 40 + index * 5),
                    (150, 70 + index * 5),
                    (160, 160, 160),
                    -1,
                )
                writer.write(frame)
        finally:
            writer.release()
        print(f"[{camera_id}] Created {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
