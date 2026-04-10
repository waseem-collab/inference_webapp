#!/usr/bin/env python3
"""
Image crop interface for PPE workflow.

Features:
1) Prompts for input and output image folder paths.
2) Navigates images in an OpenCV interface.
3) Shows person + PPE annotations.
4) Allows manual person selection (draw box with mouse).
5) Press C to copy selected person crop to output folder.
"""

from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PPE_MODEL = SCRIPT_DIR / "pharma-ppe-pt-1.0.0" / "pharma-ppe-pt" / "pharma-ppe-detection.pt"
DEFAULT_PERSON_MODEL = SCRIPT_DIR / "yolo26n.pt"
PERSON_CLASS_ID = 0
PERSON_PAD = 10
WINDOW_NAME = "PPE Cropper"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PATH_HISTORY_FILE = SCRIPT_DIR / "ppe_cropper_paths.json"


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


def save_crop(image, box, image_path: Path, output_dir: Path, idx_hint: int) -> str | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])
    x1 = max(0, x1 - PERSON_PAD)
    y1 = max(0, y1 - PERSON_PAD)
    x2 = min(w - 1, x2 + PERSON_PAD)
    y2 = min(h - 1, y2 + PERSON_PAD)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    save_name = f"{image_path.stem}_person_{idx_hint}_{stamp}.jpg"
    save_path = output_dir / save_name
    cv2.imwrite(str(save_path), crop)
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
        f"{len(pending_manual_boxes)} manual box(es). Press C to save.",
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

    state = {
        "rects": [],
        "drawing": False,
        "resizing": False,
        "active_box": None,
        "active_corner": None,
        "start": (0, 0),
        "temp_rect": None,
    }

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
            pts = corners(rect)
            for c_idx, (cx, cy) in enumerate(pts):
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
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        return [x1, y1, x2, y2]

    def on_mouse(event, x, y, flags, param):
        x = min(max(0, x), w - 1)
        y = min(max(0, y), h - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = nearest_corner(x, y)
            if hit is not None:
                b_idx, c_idx = hit
                state["resizing"] = True
                state["active_box"] = b_idx
                state["active_corner"] = c_idx
                return
            state["drawing"] = True
            state["start"] = (x, y)
            state["temp_rect"] = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if state["drawing"]:
                sx, sy = state["start"]
                state["temp_rect"] = normalize_rect(sx, sy, x, y)
            elif state["resizing"] and state["active_box"] is not None:
                rect = state["rects"][state["active_box"]]
                x1, y1, x2, y2 = rect
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


def infer_annotated(
    image,
    person_model: YOLO,
    ppe_model: YOLO,
    person_conf: float,
    ppe_conf: float,
    manual_person_boxes=None,
):
    h, w = image.shape[:2]
    annotated = image.copy()
    person_boxes: list[tuple[int, int, int, int, float]] = []
    manual_person_boxes = manual_person_boxes or []

    def process_person_region(x1, y1, x2, y2, label_prefix, conf_for_label):
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            return

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            annotated,
            f"{label_prefix} {conf_for_label}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2,
        )

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return
        ppe_result = ppe_model.predict(crop, conf=ppe_conf, verbose=False)[0]
        if ppe_result.boxes is None:
            return
        for pb in ppe_result.boxes:
            pconf = float(pb.conf[0])
            cls_id = int(pb.cls[0])
            label = ppe_result.names.get(cls_id, str(cls_id))
            cx1, cy1, cx2, cy2 = map(int, pb.xyxy[0])
            gx1, gy1 = x1 + cx1, y1 + cy1
            gx2, gy2 = x1 + cx2, y1 + cy2
            gx1, gy1, gx2, gy2 = clamp_box(gx1, gy1, gx2, gy2, w, h)
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{label} {pconf:.2f}",
                (gx1, max(18, gy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    person_result = person_model.predict(image, conf=person_conf, verbose=False)[0]
    if person_result.boxes is not None:
        for pb in person_result.boxes:
            cls_id = int(pb.cls[0])
            if cls_id != PERSON_CLASS_ID:
                continue
            conf = float(pb.conf[0])
            x1, y1, x2, y2 = map(int, pb.xyxy[0])
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            person_boxes.append((x1, y1, x2, y2, conf))
    for i, box in enumerate(person_boxes, start=1):
        x1, y1, x2, y2, conf = box
        process_person_region(x1, y1, x2, y2, f"P{i}", f"{conf:.2f}")

    for box in manual_person_boxes:
        mx1, my1, mx2, my2 = map(int, box[:4])
        process_person_region(mx1, my1, mx2, my2, "P-manual", "1.00")

    return annotated, person_boxes


def main() -> None:
    input_folder = prompt_folder_with_history(
        kind="input",
        history_key="input_paths",
        must_exist=True,
        allow_clear=True,
    )
    output_folder = prompt_folder_with_history(
        kind="output",
        history_key="output_paths",
        must_exist=False,
        allow_clear=False,
    )

    images = list_images(input_folder)
    if not images:
        print("No images found in input folder.")
        sys.exit(1)

    if not Path(DEFAULT_PPE_MODEL).exists():
        print(f"PPE model not found: {DEFAULT_PPE_MODEL}")
        sys.exit(1)

    print("Loading models...")
    person_model = YOLO(DEFAULT_PERSON_MODEL)
    ppe_model = YOLO(DEFAULT_PPE_MODEL)

    person_conf = 0.4
    ppe_conf = 0.4

    idx = 0
    selected_auto_idx: int | None = None
    pending_manual_boxes: list[tuple[int, int, int, int]] = []
    needs_refresh = True
    display = None
    current_image = None
    current_path = None
    current_auto_boxes = []

    def refresh() -> None:
        nonlocal display, current_image, current_path, current_auto_boxes, selected_auto_idx
        current_path = images[idx]
        current_image = cv2.imread(str(current_path))
        if current_image is None:
            display = None
            current_auto_boxes = []
            return
        annotated, current_auto_boxes = infer_annotated(
            current_image,
            person_model,
            ppe_model,
            person_conf,
            ppe_conf,
            manual_person_boxes=pending_manual_boxes,
        )
        if selected_auto_idx is not None and selected_auto_idx >= len(current_auto_boxes):
            selected_auto_idx = None
        display = annotated

    def redraw_overlay() -> None:
        if display is None:
            return
        frame = display.copy()

        if selected_auto_idx is not None and 0 <= selected_auto_idx < len(current_auto_boxes):
            x1, y1, x2, y2, _ = current_auto_boxes[selected_auto_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, "AUTO SELECTED", (x1, max(18, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frame = draw_pending_manual_boxes(frame, pending_manual_boxes)

        cv2.putText(
            frame,
            f"{idx + 1}/{len(images)}  |  A/D or Arrows: Prev/Next  |  M: Manual boxes  |  C: Save crop(s)  |  Q: Quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, frame)

    def on_mouse(event, x, y, flags, param):
        nonlocal selected_auto_idx
        if current_image is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(current_auto_boxes):
                x1, y1, x2, y2, _ = box
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
            pending_manual_boxes = []
            needs_refresh = True
        elif key_char in (ord("d"), ord("D")) or key in right_arrow:
            idx = min(len(images) - 1, idx + 1)
            selected_auto_idx = None
            pending_manual_boxes = []
            needs_refresh = True
        elif key_char in (ord("m"), ord("M")):
            selected_auto_idx = None
            new_boxes = select_manual_boxes_with_handles(display.copy(), initial_boxes=pending_manual_boxes)
            # Manual box tool temporarily owns mouse events; restore click-selection callback.
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)
            if new_boxes is not None:
                pending_manual_boxes = new_boxes
                needs_refresh = True
            else:
                print("Manual selection cancelled.")
        elif key_char in (ord("c"), ord("C")):
            if current_image is None or current_path is None:
                continue
            if pending_manual_boxes:
                saved = 0
                for i, box in enumerate(pending_manual_boxes, start=1):
                    save_path = save_crop(current_image, box, current_path, output_folder, i)
                    if save_path:
                        saved += 1
                        print(f"Saved crop: {save_path}")
                if saved == 0:
                    print("Failed to save manual crops.")
            elif selected_auto_idx is not None and 0 <= selected_auto_idx < len(current_auto_boxes):
                ax1, ay1, ax2, ay2, _ = current_auto_boxes[selected_auto_idx]
                save_path = save_crop(current_image, (ax1, ay1, ax2, ay2), current_path, output_folder, selected_auto_idx + 1)
                if save_path:
                    print(f"Saved crop: {save_path}")
                else:
                    print("Failed to save crop.")
            else:
                print("No selected person. Click auto person or press M to draw manual boxes.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

