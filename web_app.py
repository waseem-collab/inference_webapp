#!/usr/bin/env python3
import html
import json
import logging
import os
import threading
import time
import webbrowser
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

from PPE import ppe_inference
from SM import sm_cropper


APP_TITLE = "Inference Web App (PPE + SM)"
APP_SETTINGS_FILE = Path(__file__).resolve().parent / "web_app_settings.json"
MODELS_DIR = Path(__file__).resolve().parent / "models"
NONE_MODEL_VALUE = "__none__"
PERSON_CROP_PADDING = 10
CROPS_ROOT = Path(__file__).resolve().parent / "crops"


app = Flask(__name__)
state_lock = threading.Lock()
model_cache = {}

state = {
    "person_model_path": ppe_inference.DEFAULT_PERSON_MODEL,
    "ppe_model_path": ppe_inference.DEFAULT_MODEL_PATH,
    "sm_model_path": sm_cropper.DEFAULT_SINGLE_MODEL_PATH,
    "video_folder": str(Path.cwd()),
    "selected_video": "",
    "person_conf": 0.4,
    "ppe_conf": 0.4,
    "sm_conf": 0.7,
    "frame_step": 1,
    "simulate_realtime": True,
    "playing": False,
}

runtime = {
    "cap": None,
    "active_video": "",
    "fps": 25.0,
    "total_frames": 0,
    "frame_idx": -1,
    "displayed_count": 0,
    "started_at": time.time(),
    "last_jpg": None,
    "last_error": "",
    "last_raw_frame": None,
    "last_person_boxes": [],
    "last_action": "",
    "manual_boxes_by_frame": {},
}


def configure_quiet_logging():
    # Keep terminal clean: hide per-request access logs, keep errors.
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.ERROR)
    app.logger.setLevel(logging.ERROR)


def list_videos(folder_path: str) -> list[str]:
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        return []
    videos = []
    for name in os.listdir(folder):
        full_path = folder / name
        if full_path.is_file() and name.lower().endswith(ppe_inference.VIDEO_EXTENSIONS):
            videos.append(str(full_path))
    videos.sort()
    return videos


def discover_pt_models(root_dirs: list[str]) -> list[str]:
    found = set()
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(".pt"):
                    found.add(str(Path(dirpath) / filename))
    return sorted(found)


def _is_generic_yolo_pt(path: Path) -> bool:
    """Basenames like yolo11n.pt — prefer real PPE/SM weights inside the same package folder."""
    name = path.name.lower()
    return name.startswith("yolo") and name.endswith(".pt")


def discover_model_packages(models_dir: Path) -> list[tuple[str, str]]:
    """
    One entry per immediate subfolder of models/ that contains a .pt file.
    Returns (folder_display_name, absolute_path_to_chosen_pt).
    """
    if not models_dir.is_dir():
        return []
    packages: list[tuple[str, str]] = []
    for child in sorted(models_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        pt_files: list[Path] = []
        for dirpath, _, filenames in os.walk(child):
            for fn in filenames:
                if fn.lower().endswith(".pt"):
                    pt_files.append(Path(dirpath) / fn)
        if not pt_files:
            continue
        non_yolo = [p for p in pt_files if not _is_generic_yolo_pt(p)]
        candidates = non_yolo if non_yolo else pt_files
        candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
        chosen = str(candidates[0].resolve())
        packages.append((child.name, chosen))
    return packages


def load_settings():
    if not APP_SETTINGS_FILE.exists():
        return
    try:
        with open(APP_SETTINGS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            return
        for key in list(state.keys()):
            if key in loaded:
                state[key] = loaded[key]
    except (OSError, json.JSONDecodeError):
        return


def save_settings():
    try:
        with open(APP_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


def get_or_load_model(model_path: str):
    if model_path not in model_cache:
        model_cache[model_path] = YOLO(model_path)
    return model_cache[model_path]


def release_capture():
    cap = runtime["cap"]
    if cap is not None:
        cap.release()
    runtime["cap"] = None
    runtime["active_video"] = ""
    runtime["frame_idx"] = -1
    state["playing"] = False


def ensure_capture_open():
    selected_video = state["selected_video"]
    if not selected_video:
        runtime["last_error"] = "Choose a valid video."
        return False

    if runtime["cap"] is not None and runtime["active_video"] == selected_video:
        return True

    release_capture()
    cap = cv2.VideoCapture(selected_video)
    if not cap.isOpened():
        runtime["last_error"] = "Failed to open selected video."
        return False

    runtime["cap"] = cap
    runtime["active_video"] = selected_video
    runtime["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    runtime["fps"] = fps if fps and fps > 1e-6 else 25.0
    runtime["frame_idx"] = -1
    runtime["displayed_count"] = 0
    runtime["started_at"] = time.time()
    runtime["last_error"] = ""
    return True


def run_ppe(frame, person_model, ppe_model, person_conf, ppe_conf, manual_boxes=None):
    annotated = frame.copy()
    h, w = frame.shape[:2]
    person_count = 0
    ppe_count = 0
    person_boxes = []
    manual_boxes = manual_boxes or []

    def process_person_region(x1, y1, x2, y2, person_label):
        nonlocal person_count, ppe_count
        x1, y1, x2, y2 = ppe_inference.clamp_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            return
        person_count += 1
        person_boxes.append((x1, y1, x2, y2))
        person_idx = len(person_boxes)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            annotated,
            f"P{person_idx} {person_label}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        ppe_result = ppe_model.predict(crop, conf=ppe_conf, verbose=False)[0]
        if ppe_result.boxes is None:
            return

        for bbox in ppe_result.boxes:
            bconf = float(bbox.conf[0])
            bcls = int(bbox.cls[0])
            label = ppe_result.names.get(bcls, str(bcls))
            cx1, cy1, cx2, cy2 = map(int, bbox.xyxy[0])
            gx1, gy1 = x1 + cx1, y1 + cy1
            gx2, gy2 = x1 + cx2, y1 + cy2
            gx1, gy1, gx2, gy2 = ppe_inference.clamp_box(gx1, gy1, gx2, gy2, w, h)
            if gx2 <= gx1 or gy2 <= gy1:
                continue

            ppe_count += 1
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{label} {bconf:.2f}",
                (gx1, max(18, gy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    # Force person-model stage to detect only "person" class.
    person_result = person_model.predict(
        frame,
        conf=person_conf,
        classes=[ppe_inference.PERSON_CLASS_ID],
        verbose=False,
    )[0]

    if person_result.boxes is not None:
        for pbox in person_result.boxes:
            cls_id = int(pbox.cls[0])
            if cls_id != ppe_inference.PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, pbox.xyxy[0])
            pconf = float(pbox.conf[0])
            process_person_region(x1, y1, x2, y2, f"person {pconf:.2f}")

    for mb in manual_boxes:
        mx1, my1, mx2, my2 = map(int, mb[:4])
        process_person_region(mx1, my1, mx2, my2, "manual")

    cv2.putText(
        annotated,
        f"PPE: persons={person_count}, ppe_boxes={ppe_count}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return annotated, person_boxes


def run_sm(frame, sm_model, sm_conf):
    annotated, det_count = sm_cropper.draw_detections(frame, sm_model, sm_conf)
    cv2.putText(
        annotated,
        f"SM: detections={det_count}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return annotated


def build_model_options_html(model_paths: list[str], selected_path: str) -> str:
    options = [
        f'<option value="{NONE_MODEL_VALUE}"'
        + (' selected="selected"' if selected_path == NONE_MODEL_VALUE else "")
        + ">None</option>"
    ]
    for path in model_paths:
        selected_attr = ' selected="selected"' if path == selected_path else ""
        label = html.escape(Path(path).name)
        value = html.escape(path, quote=True)
        options.append(f'<option value="{value}"{selected_attr}>{label}</option>')
    return "".join(options)


def build_model_package_options_html(
    packages: list[tuple[str, str]], selected_path: str
) -> str:
    """Dropdown shows top-level folder name; value is the resolved .pt path inside."""
    options = [
        f'<option value="{NONE_MODEL_VALUE}"'
        + (' selected="selected"' if selected_path == NONE_MODEL_VALUE else "")
        + ">None</option>"
    ]
    for folder_name, pt_path in packages:
        selected_attr = ' selected="selected"' if pt_path == selected_path else ""
        label = html.escape(folder_name)
        value = html.escape(pt_path, quote=True)
        tip = html.escape(Path(pt_path).name, quote=True)
        options.append(
            f'<option value="{value}" title="{tip}"{selected_attr}>{label}</option>'
        )
    return "".join(options)


def process_frame_to_jpg(frame):
    try:
        out = frame.copy()
        runtime["last_raw_frame"] = frame.copy()
        runtime["last_person_boxes"] = []
        current_frame_idx = int(runtime.get("frame_idx", -1))
        manual_boxes = runtime["manual_boxes_by_frame"].get(current_frame_idx, [])
        ran_any = False
        if (
            state["person_model_path"] != NONE_MODEL_VALUE
            and state["ppe_model_path"] != NONE_MODEL_VALUE
        ):
            person_model = get_or_load_model(state["person_model_path"])
            ppe_model = get_or_load_model(state["ppe_model_path"])
            out, person_boxes = run_ppe(
                out,
                person_model,
                ppe_model,
                float(state["person_conf"]),
                float(state["ppe_conf"]),
                manual_boxes=manual_boxes,
            )
            runtime["last_person_boxes"] = person_boxes
            ran_any = True

        if state["sm_model_path"] != NONE_MODEL_VALUE:
            sm_model = get_or_load_model(state["sm_model_path"])
            out = run_sm(out, sm_model, float(state["sm_conf"]))
            ran_any = True

        if not ran_any:
            cv2.putText(
                out,
                "No model selected (all set to None)",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
    except Exception as exc:
        runtime["last_error"] = f"Inference error: {exc}"
        state["playing"] = False
        return runtime["last_jpg"]

    ok, buf = cv2.imencode(".jpg", out)
    if not ok:
        return runtime["last_jpg"]

    runtime["displayed_count"] += 1
    runtime["last_jpg"] = buf.tobytes()
    return runtime["last_jpg"]


def save_frame_image(frame, subdir: str, suffix: str = "") -> str:
    target_dir = CROPS_ROOT / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    base = Path(runtime["active_video"]).stem if runtime["active_video"] else "frame"
    frame_idx = max(int(runtime["frame_idx"]), 0)
    stamp = int(time.time() * 1000)
    suffix_part = f"_{suffix}" if suffix else ""
    filename = f"{base}_frame_{frame_idx:06d}{suffix_part}_{stamp}.jpg"
    save_path = target_dir / filename
    cv2.imwrite(str(save_path), frame)
    return str(save_path)


def save_person_crop(person_index: int) -> str:
    frame = runtime["last_raw_frame"]
    if frame is None:
        raise ValueError("No frame available for person crop.")

    boxes = runtime["last_person_boxes"] or []
    if person_index < 1 or person_index > len(boxes):
        raise ValueError("Selected person number is not available in current frame.")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = boxes[person_index - 1]
    x1 = max(0, x1 - PERSON_CROP_PADDING)
    y1 = max(0, y1 - PERSON_CROP_PADDING)
    x2 = min(w - 1, x2 + PERSON_CROP_PADDING)
    y2 = min(h - 1, y2 + PERSON_CROP_PADDING)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid person crop bounds.")

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop generated.")
    return save_frame_image(crop, "person", suffix=f"person_{person_index}")


def save_person_crop_from_box(box, suffix: str = "person_manual") -> str:
    frame = runtime["last_raw_frame"]
    if frame is None:
        raise ValueError("No frame available for person crop.")
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])
    x1 = max(0, x1 - PERSON_CROP_PADDING)
    y1 = max(0, y1 - PERSON_CROP_PADDING)
    x2 = min(w - 1, x2 + PERSON_CROP_PADDING)
    y2 = min(h - 1, y2 + PERSON_CROP_PADDING)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid manual box bounds.")
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop generated.")
    return save_frame_image(crop, "person", suffix=suffix)


def add_manual_box_for_current_frame(x1_ratio: float, y1_ratio: float, x2_ratio: float, y2_ratio: float):
    frame = runtime["last_raw_frame"]
    if frame is None:
        raise ValueError("No frame loaded yet.")
    frame_idx = int(runtime["frame_idx"])
    if frame_idx < 0:
        raise ValueError("No active frame index.")
    h, w = frame.shape[:2]
    x1 = int(max(0.0, min(1.0, x1_ratio)) * (w - 1))
    y1 = int(max(0.0, min(1.0, y1_ratio)) * (h - 1))
    x2 = int(max(0.0, min(1.0, x2_ratio)) * (w - 1))
    y2 = int(max(0.0, min(1.0, y2_ratio)) * (h - 1))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        raise ValueError("Manual box is too small.")
    box = (x1, y1, x2, y2)
    runtime["manual_boxes_by_frame"].setdefault(frame_idx, []).append(box)
    return box, frame_idx


def save_person_crop_from_point(x_ratio: float, y_ratio: float) -> str:
    frame = runtime["last_raw_frame"]
    boxes = runtime["last_person_boxes"] or []
    if frame is None:
        raise ValueError("No frame available for person crop.")
    if not boxes:
        raise ValueError("No person in the current frame.")

    h, w = frame.shape[:2]
    px = int(max(0.0, min(1.0, x_ratio)) * (w - 1))
    py = int(max(0.0, min(1.0, y_ratio)) * (h - 1))

    # Prefer a box that contains the clicked point.
    chosen_idx = None
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        if x1 <= px <= x2 and y1 <= py <= y2:
            chosen_idx = idx
            break

    if chosen_idx is None:
        raise ValueError("Click on a detected person in the video.")
    return save_person_crop(chosen_idx)


def seek_to_frame(target_idx: int):
    if not ensure_capture_open():
        return False
    cap = runtime["cap"]
    total = max(int(runtime["total_frames"]), 0)
    if total > 0:
        target_idx = max(0, min(int(target_idx), total - 1))
    else:
        target_idx = max(0, int(target_idx))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        runtime["last_error"] = "Unable to seek to requested frame."
        return False

    runtime["frame_idx"] = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    process_frame_to_jpg(frame)
    return True


def next_frame_jpg():
    if not state["playing"]:
        return runtime["last_jpg"]
    if not ensure_capture_open():
        return runtime["last_jpg"]

    cap = runtime["cap"]
    frame_step = max(int(state["frame_step"]), 1)
    current = None
    ok = False
    for _ in range(frame_step):
        ok, frame = cap.read()
        if not ok:
            break
        current = frame
        runtime["frame_idx"] = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    if not ok or current is None:
        state["playing"] = False
        return runtime["last_jpg"]

    return process_frame_to_jpg(current)


def mjpeg_generator():
    while True:
        with state_lock:
            jpg = next_frame_jpg()
            realtime = bool(state["simulate_realtime"])
            fps = float(runtime["fps"]) if runtime["fps"] > 0 else 25.0
            frame_step = max(int(state["frame_step"]), 1)
            playing = bool(state["playing"])

        if jpg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )

        if playing and realtime:
            delay = max(0.0, (frame_step / fps) * 0.6)
            time.sleep(delay)
        else:
            time.sleep(0.05)


@app.get("/")
def index():
    with state_lock:
        videos = list_videos(state["video_folder"])
        if state["selected_video"] not in videos:
            state["selected_video"] = videos[0] if videos else ""
            save_settings()
        all_pt = discover_pt_models([str(MODELS_DIR)])
        packages = discover_model_packages(MODELS_DIR)
        package_paths = [p for _, p in packages]
        if packages:
            allowed_pkg = set(package_paths)
            allowed_pkg.add(NONE_MODEL_VALUE)
            if state["ppe_model_path"] not in allowed_pkg:
                state["ppe_model_path"] = package_paths[0]
            if state["sm_model_path"] not in allowed_pkg:
                state["sm_model_path"] = package_paths[0]
        else:
            state["ppe_model_path"] = NONE_MODEL_VALUE
            state["sm_model_path"] = NONE_MODEL_VALUE
        if all_pt:
            allowed_pt = set(all_pt)
            allowed_pt.add(NONE_MODEL_VALUE)
            if state["person_model_path"] not in allowed_pt:
                state["person_model_path"] = all_pt[0]
        else:
            state["person_model_path"] = NONE_MODEL_VALUE
        person_model_options_html = build_model_options_html(all_pt, state["person_model_path"])
        ppe_model_options_html = build_model_package_options_html(packages, state["ppe_model_path"])
        sm_model_options_html = build_model_package_options_html(packages, state["sm_model_path"])
        discovered_lines = [
            f"{name} → {Path(pt).name}" for name, pt in packages
        ]
        discovered_models_text = (
            os.linesep.join(discovered_lines) if discovered_lines else "No model packages in models/"
        )

    page_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{APP_TITLE}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; margin: 0; background: #0f1115; color: #eee; }}
    .wrap {{ padding: 16px; max-width: 1800px; margin: 0 auto; }}
    .title {{ margin: 0 0 12px 0; font-size: 22px; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 3.2fr) minmax(300px, 1fr); gap: 12px; align-items: start; }}
    .layout.right-hidden {{ grid-template-columns: 1fr 0; }}
    .layout.right-hidden .right-col {{ display: none; }}
    .left-col {{ min-width: 0; }}
    .right-col {{ min-width: 0; }}
    .right-toggle {{
      position: fixed;
      top: 14px;
      right: 14px;
      z-index: 1000;
      width: auto;
      min-height: 34px;
      padding: 6px 10px;
      border-radius: 8px;
      background: #111827;
      border: 1px solid #334155;
    }}
    .panel {{ border: 1px solid #2f3440; border-radius: 10px; background: #161a22; padding: 12px; margin-bottom: 12px; }}
    .panel-title {{ margin: 0 0 10px 0; color: #cdd5e1; font-size: 14px; letter-spacing: .2px; }}
    .grid-4 {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .grid-5 {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .nav-grid {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; }}
    .item {{ display: flex; flex-direction: column; gap: 6px; }}
    .item label {{ color: #aeb6c4; font-size: 12px; }}
    input, select, button {{
      width: 100%;
      min-height: 36px;
      padding: 8px 10px;
      background: #1a1d24;
      color: #eee;
      border: 1px solid #3a4150;
      border-radius: 6px;
      outline: none;
    }}
    input:focus, select:focus {{ border-color: #5f89ff; }}
    button {{ cursor: pointer; font-weight: 600; }}
    .btn-start {{ background: #1b5e20; border-color: #2e7d32; }}
    .btn-pause {{ background: #7b4f00; border-color: #9a6b08; }}
    .btn-stop {{ background: #7f1d1d; border-color: #991b1b; }}
    .btn-crop {{ background: #1f2937; border-color: #374151; }}
    .video-wrap {{ position: relative; }}
    .video {{ background: #000; border: 1px solid #333; border-radius: 8px; width: 100%; max-height: 80vh; height: auto; display: block; cursor: default; }}
    .video.pick-mode {{ cursor: crosshair; }}
    .video-actions {{
      position: absolute;
      top: 12px;
      right: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      width: 132px;
      z-index: 10;
    }}
    .video-actions button {{ min-height: 34px; }}
    .draw-overlay {{
      position: absolute;
      inset: 0;
      z-index: 8;
      display: none;
      cursor: crosshair;
    }}
    .draw-box {{
      position: absolute;
      border: 2px solid #38bdf8;
      background: rgba(56, 189, 248, 0.15);
      pointer-events: none;
      display: none;
    }}
    .video-note {{
      margin-top: 4px;
      font-size: 12px;
      color: #e5e7eb;
      background: rgba(17, 24, 39, 0.85);
      border: 1px solid #374151;
      border-radius: 6px;
      padding: 6px 8px;
      display: none;
    }}
    .toast {{
      position: fixed;
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%);
      background: rgba(22, 163, 74, 0.95);
      color: #fff;
      border: 1px solid #15803d;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 13px;
      z-index: 2000;
      display: none;
    }}
    .toast.error {{
      background: rgba(153, 27, 27, 0.95);
      border-color: #b91c1c;
    }}
    .meta {{ font-size: 13px; color: #bbb; }}
    .list {{ font-size: 12px; color: #aaa; white-space: pre-wrap; margin-top: 8px; }}
    @media (max-width: 1200px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .grid-4 {{ grid-template-columns: 1fr; }}
      .grid-5 {{ grid-template-columns: 1fr; }}
      .nav-grid {{ grid-template-columns: repeat(2, minmax(220px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h3 class="title">{APP_TITLE}</h3>
    <button id="right_toggle_btn" class="right-toggle" onclick="toggleRightPanel()">Hide Panel</button>
    <div class="layout" id="main_layout">
      <div class="left-col">
        <div class="panel">
          <div class="panel-title">Video</div>
          <div class="video-wrap">
            <img id="video_stream" class="video" src="/stream.mjpg" />
            <div id="draw_overlay" class="draw-overlay">
              <div id="draw_box" class="draw-box"></div>
            </div>
            <div class="video-actions">
              <button id="crop_person_btn" class="btn-crop" onclick="togglePersonCrop()">Person (P)</button>
              <button class="btn-crop" onclick="cropFrame()">Frame (F)</button>
              <button class="btn-crop" onclick="cropBackground()">Background (B)</button>
              <button id="crop_manual_btn" class="btn-crop" onclick="toggleManualMode()">Manual (M)</button>
              <button id="crop_cancel_btn" class="btn-crop" style="display:none;" onclick="closePersonCrop()">X</button>
              <div id="video_action_note" class="video-note"></div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-title">Video Navigation</div>
          <div class="nav-grid">
            <div class="item"><label>&nbsp;</label><button onclick="seekBy(-5, 'sec')">-5 sec</button></div>
            <div class="item"><label>&nbsp;</label><button onclick="seekBy(-1, 'frame')">-1 frame</button></div>
            <div class="item"><label>&nbsp;</label><button id="play_pause_btn" onclick="togglePlayPause()">&#9658;</button></div>
            <div class="item"><label>&nbsp;</label><button onclick="seekBy(1, 'frame')">+1 frame</button></div>
            <div class="item"><label>&nbsp;</label><button onclick="seekBy(5, 'sec')">+5 sec</button></div>
          </div>
          <div class="item" style="margin-top: 10px;">
            <label for="seek_slider">Dragger (frame seek)</label>
            <input id="seek_slider" type="range" min="0" max="0" step="1" value="0" />
          </div>
          <div class="item" style="margin-top: 10px;">
            <label>Status</label>
            <div class="meta" id="meta">Loading...</div>
          </div>
        </div>

      </div>

      <div class="right-col">
        <div class="panel">
          <div class="panel-title">Model Configuration</div>
          <div class="grid-4">
            <div class="item"><label for="person_model_path">Person model path</label><select id="person_model_path">{person_model_options_html}</select></div>
            <div class="item"><label for="ppe_model_path">PPE model path</label><select id="ppe_model_path">{ppe_model_options_html}</select></div>
            <div class="item"><label for="sm_model_path">SM model path</label><select id="sm_model_path">{sm_model_options_html}</select></div>
            <div class="item"><label>Models directory</label><input value="{html.escape(str(MODELS_DIR), quote=True)}" disabled /></div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-title">Playback and Thresholds</div>
          <div class="grid-4">
            <div class="item"><label for="video_folder">Video folder</label><input id="video_folder" value="{state["video_folder"]}" /></div>
            <div class="item"><label for="selected_video">Video file</label><select id="selected_video"></select></div>
            <div class="item"><label for="person_conf">Person conf</label><input id="person_conf" type="number" min="0" max="1" step="0.05" value="{state["person_conf"]}" /></div>
            <div class="item"><label for="ppe_conf">PPE conf</label><input id="ppe_conf" type="number" min="0" max="1" step="0.05" value="{state["ppe_conf"]}" /></div>
          </div>
          <div class="grid-5" style="margin-top: 10px;">
            <div class="item"><label for="sm_conf">SM conf</label><input id="sm_conf" type="number" min="0" max="1" step="0.05" value="{state["sm_conf"]}" /></div>
            <div class="item"><label for="frame_step">Frame step</label><input id="frame_step" type="number" min="1" max="60" step="1" value="{state["frame_step"]}" /></div>
            <div class="item"><label for="simulate_realtime">Realtime</label><select id="simulate_realtime"><option value="true" {"selected" if state["simulate_realtime"] else ""}>Enabled</option><option value="false" {"selected" if not state["simulate_realtime"] else ""}>Disabled</option></select></div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-title">Discovered Models</div>
          <div class="list">{html.escape(discovered_models_text)}</div>
        </div>
      </div>
    </div>
  </div>

<script>
  let allVideos = {json.dumps(videos)};
  let selectedVideo = {json.dumps(state["selected_video"])};
  let suppressSliderUpdate = false;
  let personPickMode = false;
  let manualDrawMode = false;
  let drawStart = null;
  let toastTimer = null;

  function setVideos(list, preferredVideo = null) {{
    allVideos = list || [];
    const el = document.getElementById("selected_video");
    el.innerHTML = "";
    const target = preferredVideo || selectedVideo;
    allVideos.forEach(v => {{
      const opt = document.createElement("option");
      opt.value = v;
      opt.text = v.split("/").pop();
      if (v === target) opt.selected = true;
      el.appendChild(opt);
    }});
    if (el.options.length > 0 && !el.value) {{
      el.selectedIndex = 0;
    }}
    selectedVideo = el.value || "";
  }}

  function payloadFromInputs() {{
    return {{
      person_model_path: document.getElementById("person_model_path").value,
      ppe_model_path: document.getElementById("ppe_model_path").value,
      sm_model_path: document.getElementById("sm_model_path").value,
      video_folder: document.getElementById("video_folder").value,
      selected_video: document.getElementById("selected_video").value,
      person_conf: Number(document.getElementById("person_conf").value),
      ppe_conf: Number(document.getElementById("ppe_conf").value),
      sm_conf: Number(document.getElementById("sm_conf").value),
      frame_step: Number(document.getElementById("frame_step").value),
      simulate_realtime: document.getElementById("simulate_realtime").value === "true",
    }};
  }}

  async function saveConfig(refreshVideos = false) {{
    const res = await fetch("/api/config", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(payloadFromInputs()),
    }});
    const data = await res.json();
    if (refreshVideos) setVideos(data.videos || [], data.selected_video || "");
    selectedVideo = data.selected_video || selectedVideo;
  }}

  async function control(action) {{
    await fetch("/api/control", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ action }}),
    }});
  }}

  async function togglePlayPause() {{
    await control("toggle");
  }}

  async function seekBy(delta, unit) {{
    await fetch("/api/seek", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ mode: "relative", delta, unit }}),
    }});
  }}

  async function seekToSlider() {{
    const value = Number(document.getElementById("seek_slider").value);
    await fetch("/api/seek", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ mode: "absolute", frame: value }}),
    }});
  }}

  async function refreshStatus() {{
    const res = await fetch("/api/status");
    const s = await res.json();
    const slider = document.getElementById("seek_slider");
    slider.max = String(Math.max((s.total_frames || 1) - 1, 0));
    if (!suppressSliderUpdate) {{
      slider.value = String(Math.max(s.frame_idx || 0, 0));
    }}
    const playPauseBtn = document.getElementById("play_pause_btn");
    playPauseBtn.innerHTML = s.playing ? "&#10074;&#10074;" : "&#9658;";
    const note = document.getElementById("video_action_note");
    if (personPickMode) {{
      note.style.display = "block";
      note.textContent = (s.available_person_count || 0) === 0
        ? "No person in the frame."
        : "Click a person in the video to crop.";
    }} else if (manualDrawMode) {{
      note.style.display = "block";
      note.textContent = "Drag on video and release to add manual person box.";
    }} else {{
      note.style.display = "none";
      note.textContent = "";
    }}
    document.getElementById("meta").textContent =
      `Status: ${{s.playing ? "Playing" : "Paused"}} | Frame: ${{s.frame_idx}}/${{Math.max(s.total_frames-1, 0)}} | Displayed: ${{s.displayed_count}} | Error: ${{s.last_error || "none"}} | Last Action: ${{s.last_action || "none"}}`;
  }}

  function togglePersonCrop() {{
    if (manualDrawMode) closeActionModes();
    const video = document.getElementById("video_stream");
    const cancelBtn = document.getElementById("crop_cancel_btn");
    const personBtn = document.getElementById("crop_person_btn");
    const manualBtn = document.getElementById("crop_manual_btn");
    personPickMode = true;
    video.classList.add("pick-mode");
    cancelBtn.style.display = "block";
    personBtn.style.display = "none";
    manualBtn.style.display = "none";
  }}

  function closeActionModes() {{
    const video = document.getElementById("video_stream");
    const drawOverlay = document.getElementById("draw_overlay");
    const drawBox = document.getElementById("draw_box");
    const cancelBtn = document.getElementById("crop_cancel_btn");
    const personBtn = document.getElementById("crop_person_btn");
    const manualBtn = document.getElementById("crop_manual_btn");
    const note = document.getElementById("video_action_note");
    personPickMode = false;
    manualDrawMode = false;
    drawStart = null;
    video.classList.remove("pick-mode");
    drawOverlay.style.display = "none";
    drawBox.style.display = "none";
    cancelBtn.style.display = "none";
    personBtn.style.display = "block";
    manualBtn.style.display = "block";
    note.style.display = "none";
    note.textContent = "";
  }}

  function closePersonCrop() {{
    closeActionModes();
  }}

  function toggleManualMode() {{
    if (personPickMode) closeActionModes();
    const cancelBtn = document.getElementById("crop_cancel_btn");
    const personBtn = document.getElementById("crop_person_btn");
    const manualBtn = document.getElementById("crop_manual_btn");
    const drawOverlay = document.getElementById("draw_overlay");
    manualDrawMode = true;
    drawStart = null;
    cancelBtn.style.display = "block";
    personBtn.style.display = "none";
    manualBtn.style.display = "none";
    drawOverlay.style.display = "block";
  }}

  function showToast(message, isError = false) {{
    const toast = document.getElementById("save_toast");
    toast.textContent = message;
    toast.classList.toggle("error", !!isError);
    toast.style.display = "block";
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {{
      toast.style.display = "none";
    }}, 500);
  }}

  async function requestCrop(payload) {{
    const res = await fetch("/api/crop", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(payload),
    }});
    const data = await res.json();
    if (res.ok && data.ok) {{
      showToast("Saved");
      return true;
    }}
    showToast(data.error || "Save failed", true);
    return false;
  }}

  async function cropRequest(type, personIndex = null) {{
    const payload = {{ type }};
    if (personIndex !== null) payload.person_index = personIndex;
    await requestCrop(payload);
    await refreshStatus();
  }}

  async function cropPersonByPoint(xRatio, yRatio) {{
    const ok = await requestCrop({{ type: "person_point", x_ratio: xRatio, y_ratio: yRatio }});
    await refreshStatus();
    return ok;
  }}

  async function cropFrame() {{
    await cropRequest("frame");
  }}

  async function cropBackground() {{
    await cropRequest("background");
  }}

  const videoStream = document.getElementById("video_stream");
  videoStream.addEventListener("click", async (evt) => {{
    if (!personPickMode) return;
    const rect = videoStream.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const xRatio = (evt.clientX - rect.left) / rect.width;
    const yRatio = (evt.clientY - rect.top) / rect.height;
    const ok = await cropPersonByPoint(xRatio, yRatio);
    if (ok) closeActionModes();
  }});

  const drawOverlay = document.getElementById("draw_overlay");
  const drawBox = document.getElementById("draw_box");

  function clamp01(v) {{
    return Math.max(0, Math.min(1, v));
  }}

  function updateDrawBoxDisplay(box) {{
    if (!drawStart || !box) return;
    const x1 = Math.min(box.x1, box.x2) * 100;
    const y1 = Math.min(box.y1, box.y2) * 100;
    const x2 = Math.max(box.x1, box.x2) * 100;
    const y2 = Math.max(box.y1, box.y2) * 100;
    drawBox.style.display = "block";
    drawBox.style.left = `${{x1}}%`;
    drawBox.style.top = `${{y1}}%`;
    drawBox.style.width = `${{Math.max(0, x2 - x1)}}%`;
    drawBox.style.height = `${{Math.max(0, y2 - y1)}}%`;
  }}

  drawOverlay.addEventListener("mousedown", (evt) => {{
    if (!manualDrawMode) return;
    const rect = drawOverlay.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = clamp01((evt.clientX - rect.left) / rect.width);
    const y = clamp01((evt.clientY - rect.top) / rect.height);
    drawStart = {{ x, y }};
    updateDrawBoxDisplay({{ x1: x, y1: y, x2: x, y2: y }});
  }});

  drawOverlay.addEventListener("mousemove", (evt) => {{
    if (!manualDrawMode || !drawStart) return;
    const rect = drawOverlay.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = clamp01((evt.clientX - rect.left) / rect.width);
    const y = clamp01((evt.clientY - rect.top) / rect.height);
    updateDrawBoxDisplay({{ x1: drawStart.x, y1: drawStart.y, x2: x, y2: y }});
  }});

  drawOverlay.addEventListener("mouseup", async (evt) => {{
    if (!manualDrawMode || !drawStart) return;
    const rect = drawOverlay.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = clamp01((evt.clientX - rect.left) / rect.width);
    const y = clamp01((evt.clientY - rect.top) / rect.height);
    const box = {{ x1: drawStart.x, y1: drawStart.y, x2: x, y2: y }};
    const ok = await requestCrop({{
      type: "manual_box",
      x1_ratio: box.x1,
      y1_ratio: box.y1,
      x2_ratio: box.x2,
      y2_ratio: box.y2,
    }});
    drawStart = null;
    drawBox.style.display = "none";
    await refreshStatus();
    if (ok) closeActionModes();
  }});

  drawOverlay.addEventListener("mouseleave", () => {{
    if (!manualDrawMode || !drawStart) return;
    drawStart = null;
    drawBox.style.display = "none";
  }});

  document.addEventListener("keydown", async (evt) => {{
    if (evt.ctrlKey || evt.metaKey || evt.altKey) return;
    const tag = (evt.target && evt.target.tagName ? evt.target.tagName.toLowerCase() : "");
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    const key = evt.key.toLowerCase();
    if (evt.key === "Escape") {{
      if (personPickMode || manualDrawMode) {{
        evt.preventDefault();
        closeActionModes();
      }}
      return;
    }}
    if (key === "p") {{
      evt.preventDefault();
      if (manualDrawMode) {{
        closeActionModes();
        togglePersonCrop();
      }} else if (personPickMode) {{
        closeActionModes();
      }} else {{
        togglePersonCrop();
      }}
    }} else if (key === "b") {{
      evt.preventDefault();
      await cropBackground();
    }} else if (key === "f") {{
      evt.preventDefault();
      await cropFrame();
    }} else if (key === "m") {{
      evt.preventDefault();
      if (manualDrawMode) closeActionModes();
      else toggleManualMode();
    }} else if (evt.key === "ArrowLeft") {{
      evt.preventDefault();
      await seekBy(-1, "frame");
    }} else if (evt.key === "ArrowRight") {{
      evt.preventDefault();
      await seekBy(1, "frame");
    }}
  }});

  function toggleRightPanel() {{
    const layout = document.getElementById("main_layout");
    const btn = document.getElementById("right_toggle_btn");
    const hidden = layout.classList.toggle("right-hidden");
    btn.textContent = hidden ? "Show Right Panel" : "Hide Panel";
  }}

  setVideos(allVideos, selectedVideo);

  const ids = [
    "person_model_path", "ppe_model_path", "sm_model_path",
    "video_folder", "selected_video", "person_conf", "ppe_conf", "sm_conf",
    "frame_step", "simulate_realtime"
  ];
  ids.forEach((id) => {{
    const el = document.getElementById(id);
    const eventName = (id.includes("conf") || id === "frame_step") ? "input" : "change";
    el.addEventListener(eventName, async () => {{
      if (id === "video_folder") {{
        await saveConfig(true);
      }} else {{
        await saveConfig(false);
      }}
    }});
  }});

  const seekSlider = document.getElementById("seek_slider");
  seekSlider.addEventListener("input", () => {{
    suppressSliderUpdate = true;
  }});
  seekSlider.addEventListener("change", async () => {{
    await seekToSlider();
    suppressSliderUpdate = false;
  }});

  setInterval(refreshStatus, 500);
  refreshStatus();
</script>
<div id="save_toast" class="toast"></div>
</body>
</html>
"""
    return Response(page_html, mimetype="text/html")


@app.post("/api/config")
def api_config():
    payload = request.get_json(silent=True) or {}
    with state_lock:
        old_video = state["selected_video"]
        for key in (
            "person_model_path",
            "ppe_model_path",
            "sm_model_path",
            "video_folder",
            "selected_video",
            "person_conf",
            "ppe_conf",
            "sm_conf",
            "frame_step",
            "simulate_realtime",
        ):
            if key in payload:
                state[key] = payload[key]

        state["person_conf"] = max(0.0, min(1.0, float(state["person_conf"])))
        state["ppe_conf"] = max(0.0, min(1.0, float(state["ppe_conf"])))
        state["sm_conf"] = max(0.0, min(1.0, float(state["sm_conf"])))
        state["frame_step"] = max(1, int(state["frame_step"]))

        all_pt = discover_pt_models([str(MODELS_DIR)])
        packages = discover_model_packages(MODELS_DIR)
        package_paths = [p for _, p in packages]
        if packages:
            allowed_pkg = set(package_paths)
            allowed_pkg.add(NONE_MODEL_VALUE)
            if state["ppe_model_path"] not in allowed_pkg:
                state["ppe_model_path"] = package_paths[0]
            if state["sm_model_path"] not in allowed_pkg:
                state["sm_model_path"] = package_paths[0]
        else:
            state["ppe_model_path"] = NONE_MODEL_VALUE
            state["sm_model_path"] = NONE_MODEL_VALUE
        if all_pt:
            allowed_pt = set(all_pt)
            allowed_pt.add(NONE_MODEL_VALUE)
            if state["person_model_path"] not in allowed_pt:
                state["person_model_path"] = all_pt[0]
        else:
            state["person_model_path"] = NONE_MODEL_VALUE

        videos = list_videos(str(state["video_folder"]))
        if state["selected_video"] not in videos:
            state["selected_video"] = videos[0] if videos else ""
        if old_video != state["selected_video"]:
            release_capture()
        save_settings()
        return jsonify({"ok": True, "videos": videos, "selected_video": state["selected_video"]})


@app.post("/api/control")
def api_control():
    payload = request.get_json(silent=True) or {}
    action = payload.get("action", "").lower()
    with state_lock:
        if action == "start":
            state["playing"] = True
            ensure_capture_open()
        elif action == "pause":
            state["playing"] = False
        elif action == "toggle":
            if not state["playing"]:
                state["playing"] = True
                ensure_capture_open()
            else:
                state["playing"] = False
        elif action == "stop":
            release_capture()
            runtime["last_jpg"] = None
            runtime["displayed_count"] = 0
            runtime["last_error"] = ""
        save_settings()
        return jsonify({"ok": True})


@app.post("/api/seek")
def api_seek():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "relative")).lower()
    with state_lock:
        if not ensure_capture_open():
            return jsonify({"ok": False, "error": runtime["last_error"]}), 400

        current_idx = runtime["frame_idx"]
        if current_idx < 0:
            current_idx = 0

        if mode == "absolute":
            target_idx = int(payload.get("frame", current_idx))
        else:
            delta = int(payload.get("delta", 0))
            unit = str(payload.get("unit", "frame")).lower()
            if unit == "sec":
                fps = float(runtime["fps"]) if runtime["fps"] > 0 else 25.0
                delta = int(round(delta * fps))
            target_idx = current_idx + delta

        ok = seek_to_frame(target_idx)
        if not ok:
            return jsonify({"ok": False, "error": runtime["last_error"]}), 400
        return jsonify({"ok": True, "frame_idx": runtime["frame_idx"]})


@app.post("/api/crop")
def api_crop():
    payload = request.get_json(silent=True) or {}
    crop_type = str(payload.get("type", "")).lower()
    with state_lock:
        if runtime["last_raw_frame"] is None:
            return jsonify({"ok": False, "error": "No frame loaded yet."}), 400

        try:
            if crop_type == "person":
                person_index = int(payload.get("person_index", 1))
                save_path = save_person_crop(person_index)
            elif crop_type == "person_point":
                x_ratio = float(payload.get("x_ratio", 0.0))
                y_ratio = float(payload.get("y_ratio", 0.0))
                save_path = save_person_crop_from_point(x_ratio, y_ratio)
            elif crop_type == "manual_box":
                x1_ratio = float(payload.get("x1_ratio", 0.0))
                y1_ratio = float(payload.get("y1_ratio", 0.0))
                x2_ratio = float(payload.get("x2_ratio", 0.0))
                y2_ratio = float(payload.get("y2_ratio", 0.0))
                next_person_index = len(runtime["last_person_boxes"]) + 1
                box, frame_idx = add_manual_box_for_current_frame(
                    x1_ratio, y1_ratio, x2_ratio, y2_ratio
                )
                save_path = save_person_crop_from_box(box, suffix=f"person_{next_person_index}")
                seek_to_frame(frame_idx)
            elif crop_type == "frame":
                save_path = save_frame_image(runtime["last_raw_frame"], "SM")
            elif crop_type == "background":
                save_path = save_frame_image(runtime["last_raw_frame"], "background")
            else:
                return jsonify({"ok": False, "error": "Invalid crop type."}), 400
        except Exception as exc:
            runtime["last_action"] = f"Crop failed: {exc}"
            return jsonify({"ok": False, "error": str(exc)}), 400

        runtime["last_action"] = f"Saved {crop_type} crop: {save_path}"
        return jsonify({"ok": True, "path": save_path})


@app.get("/api/status")
def api_status():
    with state_lock:
        return jsonify(
            {
                "playing": state["playing"],
                "frame_idx": runtime["frame_idx"],
                "total_frames": runtime["total_frames"],
                "displayed_count": runtime["displayed_count"],
                "last_error": runtime["last_error"],
                "last_action": runtime["last_action"],
                "available_person_count": len(runtime["last_person_boxes"]),
            }
        )


@app.get("/stream.mjpg")
def stream_mjpg():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _open_browser_when_ready(url: str, delay_sec: float = 0.8) -> None:
    def _open() -> None:
        time.sleep(delay_sec)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    load_settings()
    configure_quiet_logging()
    _open_browser_when_ready("http://127.0.0.1:8500")
    app.run(host="0.0.0.0", port=8500, debug=False, threaded=True)
