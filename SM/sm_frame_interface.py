#!/usr/bin/env python3
"""
Image interface for SM detection.

Features:
1) Prompts for input/output image folders (with remembered history).
2) Navigate images in OpenCV UI.
3) Show annotations from SM detection.
4) Manual box selection with corner handles (M).
5) On C, copy the ENTIRE frame to output (not crop) after selection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SINGLE_MODEL = SCRIPT_DIR / "single-model-detection-0.2.5" / "single-model-detection-0.2.5.pt"
WINDOW_NAME = "SM Frame Saver"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PATH_HISTORY_FILE = SCRIPT_DIR / "sm_frame_paths.json"


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def load_path_history() -> dict:
    if not os.path.exists(PATH_HISTORY_FILE):
        return {"input_paths": [], "output_paths": []}
    try:
        with open(PATH_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"input_paths": [], "output_paths": []}
        input_paths = data.get("input_paths", [])
        output_paths = data.get("output_paths", [])
        if not isinstance(input_paths, list):
            input_paths = []
        if not isinstance(output_paths, list):
            output_paths = []
        return {"input_paths": input_paths, "output_paths": output_paths}
    except (json.JSONDecodeError, OSError):
        return {"input_paths": [], "output_paths": []}


def save_path_history(history: dict) -> None:
    with open(PATH_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def prompt_folder_with_history(kind: str, history_key: str, must_exist: bool, allow_clear: bool = False) -> Path:
    history = load_path_history()
    entries = history.get(history_key, [])

    while True:
        print(f"\nSelect {kind} folder path:")
        for i, p in enumerate(entries, start=1):
            print(f"{i}. {p}")
        print("N. Enter new folder")
        if allow_clear:
            print("C. Clear saved history")

        choice = input("Choice: ").strip()
        if allow_clear and choice.lower() == "c":
            history[history_key] = []
            save_path_history(history)
            entries = []
            print(f"Cleared saved {kind} folder history.")
            continue

        if choice.lower() == "n" or not choice.isdigit():
            new_path = input(f"Enter {kind} folder path: ").strip()
            folder = Path(new_path).expanduser().resolve()
            if must_exist and not folder.is_dir():
                print(f"{kind.capitalize()} folder does not exist: {folder}")
                continue
            if not must_exist:
                folder.mkdir(parents=True, exist_ok=True)

            folder_str = str(folder)
            if folder_str not in entries:
                entries.append(folder_str)
                history[history_key] = entries
                save_path_history(history)
            return folder

        idx = int(choice) - 1
        if 0 <= idx < len(entries):
            folder = Path(entries[idx]).expanduser().resolve()
            if must_exist and not folder.is_dir():
                print(f"Saved folder no longer exists: {folder}")
                continue
            if not must_exist:
                folder.mkdir(parents=True, exist_ok=True)
            return folder

        print("Invalid selection.")


def save_full_frame(image, image_path: Path, output_dir: Path, tag: str) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    save_name = f"{image_path.stem}_{tag}_{stamp}.jpg"
    save_path = output_dir / save_name
    cv2.imwrite(str(save_path), image)
    return str(save_path)


def draw_pending_manual_boxes(frame, pending_manual_boxes):
    if not pending_manual_boxes:
        return frame
    for box in pending_manual_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 7, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y1), 7, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 7, (0, 255, 0), -1)
        cv2.circle(frame, (x1, y2), 7, (0, 255, 0), -1)
    cv2.putText(
        frame,
        f"{len(pending_manual_boxes)} manual box(es) active",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0),
        2,
    )
    return frame


def select_manual_boxes_with_handles(base_frame, initial_boxes=None):
    h, w = base_frame.shape[:2]
    handle_radius = 7
    hit_radius = 14
    state = {"rects": [], "drawing": False, "resizing": False, "active_box": None, "active_corner": None, "start": (0, 0), "temp_rect": None}

    for box in initial_boxes or []:
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 > x1 and y2 > y1:
            state["rects"].append([x1, y1, x2, y2])

    def corners(rect):
        x1, y1, x2, y2 = rect
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def nearest_corner(px, py):
        best = None
        best_dist = 10**9
        for b_idx, rect in enumerate(state["rects"]):
            for c_idx, (cx, cy) in enumerate(corners(rect)):
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                if d2 < best_dist:
                    best_dist = d2
                    best = (b_idx, c_idx)
        if best is not None and best_dist <= hit_radius * hit_radius:
            return best
        return None

    def normalize_rect(xa, ya, xb, yb):
        x1, x2 = sorted((xa, xb))
        y1, y2 = sorted((ya, yb))
        return list(clamp_box(x1, y1, x2, y2, w, h))

    def on_mouse(event, x, y, flags, param):
        x = min(max(0, x), w - 1)
        y = min(max(0, y), h - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = nearest_corner(x, y)
            if hit is not None:
                state["resizing"] = True
                state["active_box"], state["active_corner"] = hit
                return
            state["drawing"] = True
            state["start"] = (x, y)
            state["temp_rect"] = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["drawing"]:
                sx, sy = state["start"]
                state["temp_rect"] = normalize_rect(sx, sy, x, y)
            elif state["resizing"] and state["active_box"] is not None:
                x1, y1, x2, y2 = state["rects"][state["active_box"]]
                corner = state["active_corner"]
                if corner == 0:
                    x1, y1 = x, y
                elif corner == 1:
                    x2, y1 = x, y
                elif corner == 2:
                    x2, y2 = x, y
                elif corner == 3:
                    x1, y2 = x, y
                state["rects"][state["active_box"]] = normalize_rect(x1, y1, x2, y2)
        elif event == cv2.EVENT_LBUTTONUP:
            if state["drawing"] and state["temp_rect"] is not None:
                x1, y1, x2, y2 = state["temp_rect"]
                if x2 > x1 and y2 > y1:
                    state["rects"].append([x1, y1, x2, y2])
            state["drawing"] = False
            state["resizing"] = False
            state["active_box"] = None
            state["active_corner"] = None
            state["temp_rect"] = None

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            view = base_frame.copy()
            for rect in state["rects"]:
                x1, y1, x2, y2 = map(int, rect)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for cx, cy in corners(rect):
                    cv2.circle(view, (int(cx), int(cy)), handle_radius, (0, 255, 0), -1)
            if state["temp_rect"] is not None:
                x1, y1, x2, y2 = map(int, state["temp_rect"])
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                view,
                "Draw boxes. Drag corner dots to resize. Enter: confirm | Backspace: undo | Esc: cancel",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKeyEx(20)
            if key in (13, 10, 32):
                if not state["rects"]:
                    return None
                return [(x1, y1, x2, y2) for (x1, y1, x2, y2) in state["rects"]]
            if key in (8, 255) and state["rects"]:
                state["rects"].pop()
            if key in (27, ord("q"), ord("Q")):
                return None
    finally:
        cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)


def infer_annotated(image, model: YOLO, conf: float, manual_regions=None):
    h, w = image.shape[:2]
    annotated = image.copy()
    det_boxes = []
    manual_regions = manual_regions or []

    result = model.predict(image, conf=conf, verbose=False)[0]
    if result.boxes is not None:
        for box in result.boxes:
            dconf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            det_boxes.append((x1, y1, x2, y2, str(label), dconf))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{label} {dconf:.2f}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Manual fallback regions: run model only inside selected box(es) and map back.
    for i, region in enumerate(manual_regions, start=1):
        rx1, ry1, rx2, ry2 = map(int, region[:4])
        rx1, ry1, rx2, ry2 = clamp_box(rx1, ry1, rx2, ry2, w, h)
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 180, 0), 2)
        cv2.putText(
            annotated,
            f"manual-{i}",
            (rx1, max(18, ry1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )
        crop = image[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue
        crop_result = model.predict(crop, conf=conf, verbose=False)[0]
        if crop_result.boxes is None:
            continue
        for box in crop_result.boxes:
            dconf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = crop_result.names.get(cls_id, str(cls_id))
            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
            gx1, gy1 = rx1 + cx1, ry1 + cy1
            gx2, gy2 = rx1 + cx2, ry1 + cy2
            gx1, gy1, gx2, gy2 = clamp_box(gx1, gy1, gx2, gy2, w, h)
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{label} {dconf:.2f}",
                (gx1, max(18, gy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return annotated, det_boxes


def main() -> None:
    input_folder = prompt_folder_with_history("input", "input_paths", must_exist=True, allow_clear=True)
    output_folder = prompt_folder_with_history("output", "output_paths", must_exist=False, allow_clear=False)

    images = list_images(input_folder)
    if not images:
        print("No images found in input folder.")
        sys.exit(1)

    if not Path(DEFAULT_SINGLE_MODEL).exists():
        print(f"SM model not found: {DEFAULT_SINGLE_MODEL}")
        sys.exit(1)

    print("Loading SM model...")
    model = YOLO(DEFAULT_SINGLE_MODEL)
    conf = 0.4

    idx = 0
    selected_auto_idx: int | None = None
    manual_annotation_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    display = None
    current_image = None
    current_path = None
    current_det_boxes = []
    needs_refresh = True

    def refresh() -> None:
        nonlocal display, current_image, current_path, current_det_boxes, selected_auto_idx
        current_path = images[idx]
        current_image = cv2.imread(str(current_path))
        if current_image is None:
            display = None
            current_det_boxes = []
            return
        annotated, current_det_boxes = infer_annotated(
            current_image, model, conf, manual_regions=manual_annotation_boxes.get(idx, [])
        )
        if selected_auto_idx is not None and selected_auto_idx >= len(current_det_boxes):
            selected_auto_idx = None
        display = annotated

    def redraw_overlay() -> None:
        if display is None:
            return
        frame = display.copy()
        if selected_auto_idx is not None and 0 <= selected_auto_idx < len(current_det_boxes):
            x1, y1, x2, y2, label, dconf = current_det_boxes[selected_auto_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(
                frame,
                f"SELECTED {label} {dconf:.2f}",
                (x1, max(18, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        frame = draw_pending_manual_boxes(frame, manual_annotation_boxes.get(idx, []))
        cv2.putText(
            frame,
            f"{idx + 1}/{len(images)} | A/D or Arrows: Prev/Next | M: Manual boxes | C: Copy full frame | Q: Quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, frame)

    def on_mouse(event, x, y, flags, param):
        nonlocal selected_auto_idx
        if current_image is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(current_det_boxes):
                x1, y1, x2, y2, _, _ = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_auto_idx = i
                    return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        if needs_refresh:
            refresh()
            needs_refresh = False

        if display is None:
            print(f"Skipping unreadable image: {images[idx]}")
            idx = (idx + 1) % len(images)
            needs_refresh = True
            continue

        redraw_overlay()
        key = cv2.waitKeyEx(30)
        key_char = key & 0xFF
        left_arrow = {2424832, 81}
        right_arrow = {2555904, 83}

        if key_char in (ord("q"), ord("Q"), 27):
            break
        elif key_char in (ord("a"), ord("A")) or key in left_arrow:
            idx = max(0, idx - 1)
            selected_auto_idx = None
            needs_refresh = True
        elif key_char in (ord("d"), ord("D")) or key in right_arrow:
            idx = min(len(images) - 1, idx + 1)
            selected_auto_idx = None
            needs_refresh = True
        elif key_char in (ord("m"), ord("M")):
            selected_auto_idx = None
            existing = manual_annotation_boxes.get(idx, [])
            new_boxes = select_manual_boxes_with_handles(display.copy(), initial_boxes=existing)
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)
            if new_boxes is not None:
                manual_annotation_boxes[idx] = [tuple(map(int, b[:4])) for b in new_boxes]
                needs_refresh = True
            else:
                print("Manual selection cancelled.")
        elif key_char in (ord("c"), ord("C")):
            if current_image is None or current_path is None:
                continue
            has_manual = bool(manual_annotation_boxes.get(idx))
            has_auto = selected_auto_idx is not None and 0 <= selected_auto_idx < len(current_det_boxes)
            if not has_manual and not has_auto:
                print("No selected detection. Click a detection or press M to draw manual boxes.")
                continue
            tag = "manual" if has_manual else "auto"
            save_path = save_full_frame(current_image, current_path, output_folder, tag)
            if save_path:
                print(f"Saved full frame: {save_path}")
            else:
                print("Failed to save full frame.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

