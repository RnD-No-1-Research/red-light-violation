"""Tải checkpoint đã ghim revision và kiểm tra SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = {
    "vehicle": {
        "file": "yolo26s.pt",
        "repo": "Ultralytics/YOLO11",
        "revision": "8b8ac7d1fae7468f85dbf89670dd66f41f485aab",
        "remote": "yolo26s.pt",
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    },
    "traffic": {
        "file": "traffic_light.pt",
        "repo": "Sawanparuthipqr/trafficlights_detection_models",
        "revision": "dd820f77ea31d47574589391b190b6f7fa20296d",
        "remote": "collected_checkpoints/tl-yolo11n.pt",
        "sha256": "a90c121b6a1e2ed5417f46edba2e6245738cdafc698ca7c5cdc395002b58cd4e",
    },
    "plate": {
        "file": "plate_detect.pt",
        "repo": "morsetechlab/yolov11-license-plate-detection",
        "revision": "251a30d7daedca065f56e04b0af04052c907c68f",
        "remote": "license-plate-finetune-v1n.pt",
        "sha256": "0aec75976c56eb6f26dfb274c430620ec65137915ff1ae47c3a48c7af8afb7b2",
    },
}


def sha256(path: Path) -> str:
    """Tính SHA-256 theo từng khối, không nạp toàn bộ file vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(key: str, directory: Path, force: bool = False) -> Path:
    """Tải một weights; không ghi đè weights tự train nếu chưa bật force."""
    item = MANIFEST[key]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / item["file"]
    if target.exists():
        if sha256(target) == item["sha256"]:
            print(f"[system] OK {target.name} (SHA-256)")
            return target
        if not force:
            raise ValueError(
                f"{target.name} đã tồn tại với hash khác; dùng --force để thay"
            )
    url = (
        f"https://huggingface.co/{item['repo']}/resolve/"
        f"{item['revision']}/{item['remote']}"
    )
    partial = target.with_suffix(".pt.part")
    print(f"[system] Download {item['repo']} -> {target.name}", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rlv-demo/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if sha256(partial) != item["sha256"]:
            raise ValueError(f"Sai checksum: {target.name}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print(f"[system] Verified {target.name}: {target.stat().st_size} bytes")
    return target


def main() -> int:
    """CLI tải weights; chỉ cần Python chuẩn, không cần cài torch trước."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        for key in [args.only] if args.only else MANIFEST:
            download(key, ROOT / "weights", args.force)
        (ROOT / "weights/download_manifest.json").write_text(
            json.dumps(MANIFEST, indent=2), encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"[system] Download failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
