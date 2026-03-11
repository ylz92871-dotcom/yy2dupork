#!/usr/bin/env python3
"""YOLO PT inference test script for competition submission."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLO sign recognition with a .pt model on image/folder/video."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="D:/smartcar-competition/runs/detect/sign_v2_fixed_100/weights/best.pt",
        help="Path to YOLO .pt file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to one image, one image folder, or one video file.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="D:/smartcar-competition/baord/test/results",
        help="Directory to save inference outputs.",
    )
    return parser.parse_args()


def infer_single_image(
    model: YOLO, image_path: Path, save_dir: Path, imgsz: int, conf: float
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    result = model.predict(source=image, imgsz=imgsz, conf=conf, verbose=False)[0]
    drawn = result.plot()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{image_path.stem}_result{image_path.suffix}"
    ok = cv2.imwrite(str(out_path), drawn)
    if not ok:
        raise RuntimeError(f"Failed to save image result: {out_path}")
    return out_path


def infer_video(
    model: YOLO, video_path: Path, save_dir: Path, imgsz: int, conf: float
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{video_path.stem}_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {out_path}")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            drawn = result.plot()
            writer.write(drawn)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frame processed from video: {video_path}")

    return out_path


def run_inference(
    model: YOLO, source_path: Path, save_dir: Path, imgsz: int, conf: float
) -> Tuple[Path, Dict[str, int]]:
    stats = {"images": 0, "videos": 0, "failed": 0}

    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix in IMAGE_EXTS:
            stats["images"] += 1
            try:
                out_path = infer_single_image(model, source_path, save_dir, imgsz, conf)
                print(f"[SUCCESS] Image: {source_path} -> {out_path}")
            except Exception as exc:
                stats["failed"] += 1
                print(f"[FAILED] {source_path}: {exc}")
            return save_dir, stats
        if suffix in VIDEO_EXTS:
            stats["videos"] += 1
            try:
                out_path = infer_video(model, source_path, save_dir, imgsz, conf)
                print(f"[SUCCESS] Video: {source_path} -> {out_path}")
            except Exception as exc:
                stats["failed"] += 1
                print(f"[FAILED] {source_path}: {exc}")
            return save_dir, stats
        raise RuntimeError(f"Unsupported file type: {source_path}")

    if source_path.is_dir():
        task_save_dir = save_dir / f"{source_path.name}_result"
        task_save_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in source_path.iterdir() if p.is_file())
        supported_files = [
            p for p in files if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in VIDEO_EXTS
        ]
        if not supported_files:
            raise RuntimeError(f"No supported image/video found in folder: {source_path}")

        for file_path in supported_files:
            suffix = file_path.suffix.lower()
            try:
                if suffix in IMAGE_EXTS:
                    stats["images"] += 1
                    out_path = infer_single_image(model, file_path, task_save_dir, imgsz, conf)
                    print(f"[SUCCESS] Image: {file_path} -> {out_path}")
                elif suffix in VIDEO_EXTS:
                    stats["videos"] += 1
                    out_path = infer_video(model, file_path, task_save_dir, imgsz, conf)
                    print(f"[SUCCESS] Video: {file_path} -> {out_path}")
            except Exception as exc:
                stats["failed"] += 1
                print(f"[FAILED] {file_path}: {exc}")

        return task_save_dir, stats

    raise RuntimeError(f"Unsupported source path: {source_path}")


def main() -> int:
    args = parse_args()

    weights_path = Path(args.weights)
    source_path = Path(args.source)
    save_dir = Path(args.save_dir)
    task_save_dir = save_dir / f"{source_path.name}_result" if source_path.is_dir() else save_dir

    print(f"Input path: {source_path}")
    print(f"Model path: {weights_path}")
    print(f"Output dir: {task_save_dir}")

    if not weights_path.exists():
        print("Inference failed: model file does not exist.")
        return 1
    if not source_path.exists():
        print("Inference failed: source path does not exist.")
        return 1

    try:
        model = YOLO(str(weights_path))
    except Exception as exc:
        print(f"Inference failed: cannot load model. {exc}")
        return 1

    try:
        output_dir, stats = run_inference(model, source_path, save_dir, args.imgsz, args.conf)
    except Exception as exc:
        print(f"Inference failed: {exc}")
        return 1

    print("Inference finished.")
    print(f"Output saved to: {output_dir}")
    print(
        f"Summary -> images: {stats['images']}, videos: {stats['videos']}, failed: {stats['failed']}"
    )

    success_count = stats["images"] + stats["videos"] - stats["failed"]
    if success_count <= 0:
        print("Inference failed: no file processed successfully.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
