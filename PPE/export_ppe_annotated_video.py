#!/usr/bin/env python3
"""
Export an annotated video using the same two-stage PPE pipeline as ppe_inference.py
(person detector + PPE on each person crop). No OpenCV playback UI.

Run from anywhere:
    python3 PPE/export_ppe_annotated_video.py
    cd PPE && python3 export_ppe_annotated_video.py
"""

from __future__ import annotations

import os
import sys
import time

import cv2
from ultralytics import YOLO

# Allow imports when launched from repo root
_PPE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PPE_DIR not in sys.path:
    sys.path.insert(0, _PPE_DIR)

import ppe_inference as pe  # noqa: E402


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path.strip()))


def prompt_input_video() -> str:
    while True:
        raw = input("Enter path to input video file: ").strip()
        if not raw:
            print("Path cannot be empty.")
            continue
        path = _expand(raw)
        if not os.path.isfile(path):
            print(f"Not a file: {path}")
            continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Could not open video with OpenCV.")
            cap.release()
            continue
        ok, _ = cap.read()
        cap.release()
        if not ok:
            print("Could not read the first frame.")
            continue
        return path


def prompt_output_path(input_video: str) -> str:
    default = os.path.splitext(input_video)[0] + "_ppe_annotated.mp4"
    raw = input(f"Enter output annotated video path [{default}]: ").strip()
    path = _expand(raw) if raw else default
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            print(f"Cannot create output directory: {exc}")
            return prompt_output_path(input_video)
    return path


def prompt_model_path(label: str, default_path: str) -> str:
    raw = input(f"{label} (Enter for default):\n  [{default_path}]\n> ").strip()
    return _expand(raw) if raw else default_path


def prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Invalid number, using default.")
        return default


def draw_export_overlay(
    frame,
    frame_idx: int,
    total_frames: int,
    fps: float,
    person_conf: float,
    ppe_conf: float,
    person_count: int,
    ppe_count: int,
) -> None:
    time_sec = frame_idx / fps if fps > 0 else 0.0
    text = (
        f"Frame {frame_idx}/{max(total_frames - 1, 0)} | {time_sec:.2f}s | "
        f"PersonConf {person_conf:.2f} | PPEConf {ppe_conf:.2f} | "
        f"Persons {person_count} | PPE {ppe_count}"
    )
    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )


def export_annotated_video(
    source: str,
    output_path: str,
    person_model_path: str,
    ppe_model_path: str,
    person_conf: float,
    ppe_conf: float,
) -> bool:
    if not os.path.isfile(source):
        print(f"Input not found: {source}")
        return False
    if not os.path.exists(ppe_model_path):
        print(f"PPE model not found: {ppe_model_path}")
        return False

    print("Loading person model...")
    person_model = pe.load_person_model(person_model_path)
    print("Loading PPE model...")
    ppe_model = YOLO(ppe_model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: could not open input video.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-6 else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"Error: could not create output writer: {output_path}")
        return False

    print(f"Writing annotated video -> {output_path}")
    t0 = time.perf_counter()
    frame_idx = 0
    written = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = pe.run_two_stage_inference(
                frame,
                person_model,
                ppe_model,
                person_conf,
                ppe_conf,
                manual_person_boxes=None,
            )
            if len(result) == 4:
                annotated, person_count, ppe_count, _ = result
            else:
                annotated, person_count, ppe_count = result

            draw_export_overlay(
                annotated,
                frame_idx,
                total_frames,
                fps,
                person_conf,
                ppe_conf,
                person_count,
                ppe_count,
            )
            writer.write(annotated)
            written += 1
            frame_idx += 1

            if frame_idx % 30 == 0 or frame_idx == 1:
                elapsed = time.perf_counter() - t0
                rate = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"  Processed {frame_idx} frames ({rate:.2f} fps)")

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t0
    print(f"Done. Wrote {written} frames in {elapsed:.1f}s -> {output_path}")
    return written > 0


def main() -> None:
    print("PPE annotated video export (batch, no GUI)\n")

    source = prompt_input_video()
    output_path = prompt_output_path(source)

    person_path = prompt_model_path("Person detector (.pt)", pe.DEFAULT_PERSON_MODEL)
    ppe_path = prompt_model_path("PPE model (.pt)", pe.DEFAULT_MODEL_PATH)

    person_conf = prompt_float("Person confidence (0–1)", 0.4)
    ppe_conf = prompt_float("PPE confidence (0–1)", 0.4)

    if not (0.0 <= person_conf <= 1.0 and 0.0 <= ppe_conf <= 1.0):
        print("Error: confidences must be between 0 and 1.")
        sys.exit(1)

    ok = export_annotated_video(
        source=source,
        output_path=output_path,
        person_model_path=person_path,
        ppe_model_path=ppe_path,
        person_conf=person_conf,
        ppe_conf=ppe_conf,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
