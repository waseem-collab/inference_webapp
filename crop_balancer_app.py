#!/usr/bin/env python3
"""
Person Crop Balancer
====================

A standalone tool (separate from web_app.py) that:
  1. Takes a video path, a target total number of person crops, a model, and a
     grid size (rows x cols).
  2. Runs a person-detection model across the whole video.
  3. Assigns every detected person to a grid cell (by box centroid) and selects
     a spatially balanced subset of crops: each cell gets a fair share, and any
     shortfall in sparse cells is back-filled from the cells that have a surplus
     so the requested total is still reached.
  4. Saves the chosen crops (flat) under  person_crops/<video_stem>/  with a
     manifest.json, plus a heatmap.png drawn on the first frame showing which
     grid cells most crops came from.

Runs on port 8600 so it never collides with web_app.py (port 8500).
"""
import html
import json
import os
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from ultralytics import YOLO


APP_TITLE = "Person Crop Balancer"
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CROPS_ROOT = APP_DIR / "person_crops"
# Heatmaps + manifests for every processed video land here, named by video.
REPORTS_ROOT = APP_DIR / "crop_reports"
# Teacher/student disagreement feature outputs: full frames + overlays per video.
DISAGREE_ROOT = APP_DIR / "disagreement_frames"
# Remembered form inputs so you don't re-enter them every run.
SETTINGS_FILE = APP_DIR / "crop_balancer_settings.json"
DISAGREE_SETTINGS_FILE = APP_DIR / "disagreement_settings.json"
REVIEW_SETTINGS_FILE = APP_DIR / "review_settings.json"
# Image types the review tool will list/serve.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
NONE_MODEL_VALUE = "__none__"
PERSON_CROP_PADDING = 10
PORT = 8600

DEFAULT_SETTINGS = {
    "video_path": "",
    "model_path": "",
    "total": 200,
    "rows": 4,
    "cols": 4,
    "conf": 0.4,
    "stride": 15,
}

DEFAULT_DISAGREE_SETTINGS = {
    "video_path": "",
    "teacher_model_path": "",
    "student_model_path": "",
    "rows": 4,
    "cols": 4,
    "teacher_conf": 0.4,
    "student_conf": 0.4,
    "stride": 15,
    # Max frames to keep per grid block (0 = unlimited).
    "per_cell": 0,
    # Named output folder under disagreement_frames/ that holds frames/ + overlays/.
    "dest": "",
    # Resume a crashed batch: skip videos that already have a manifest in the dest.
    "resume": False,
}
# Sentinel value the destination dropdown uses for "create a new folder".
NEW_DEST_VALUE = "__new__"


def load_settings() -> dict:
    """Return saved form inputs merged over defaults."""
    merged = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            for k in DEFAULT_SETTINGS:
                if k in saved:
                    merged[k] = saved[k]
    except (OSError, json.JSONDecodeError):
        pass
    return merged


def save_settings(values: dict):
    out = {k: values.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS}
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except OSError:
        pass


def load_disagree_settings() -> dict:
    """Return saved disagreement-feature form inputs merged over defaults."""
    merged = dict(DEFAULT_DISAGREE_SETTINGS)
    try:
        with open(DISAGREE_SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            for k in DEFAULT_DISAGREE_SETTINGS:
                if k in saved:
                    merged[k] = saved[k]
    except (OSError, json.JSONDecodeError):
        pass
    return merged


def save_disagree_settings(values: dict):
    out = {k: values.get(k, DEFAULT_DISAGREE_SETTINGS[k]) for k in DEFAULT_DISAGREE_SETTINGS}
    try:
        with open(DISAGREE_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except OSError:
        pass


def list_disagree_dests():
    """Named output folders under disagreement_frames/ (each holds frames/overlays)."""
    if not DISAGREE_ROOT.is_dir():
        return []
    names = [
        name for name in os.listdir(DISAGREE_ROOT)
        if (DISAGREE_ROOT / name).is_dir() and not name.startswith(".")
    ]
    names.sort()
    return names


def build_review_dest_options_html(dests, selected_path):
    """Options for the review folder dropdown: value is the absolute dest path."""
    options = ['<option value="">— pick a folder —</option>']
    for name in dests:
        path = str(DISAGREE_ROOT / name)
        sel = ' selected="selected"' if path == selected_path else ""
        options.append(
            f'<option value="{html.escape(path, quote=True)}"{sel}>{html.escape(name)}</option>'
        )
    return "".join(options)


def build_dest_options_html(dests, selected):
    options = []
    for name in dests:
        sel = ' selected="selected"' if name == selected else ""
        label = html.escape(name)
        value = html.escape(name, quote=True)
        options.append(f'<option value="{value}"{sel}>{label}</option>')
    # Always offer a "create new folder" choice at the end.
    new_sel = ' selected="selected"' if not dests else ""
    options.append(f'<option value="{NEW_DEST_VALUE}"{new_sel}>➕ New folder…</option>')
    return "".join(options)


def load_review_folder() -> str:
    """Return the last folder used in the review tool (or empty)."""
    try:
        with open(REVIEW_SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict) and isinstance(saved.get("folder"), str):
            return saved["folder"]
    except (OSError, json.JSONDecodeError):
        pass
    return ""


def save_review_folder(folder: str):
    try:
        with open(REVIEW_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({"folder": folder}, f, indent=2)
    except OSError:
        pass

# COCO "person" class id, used to restrict detections to people.
PERSON_CLASS_ID = 0
# Video extensions we scan when given a folder.
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
# Preferred default weights if present among discovered models (else first found).
DEFAULT_PERSON_MODEL_NAME = "yolo26n.pt"


def clamp_box(x1, y1, x2, y2, width, height):
    """Clamp a box to the frame bounds (inclusive of edges)."""
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


app = Flask(__name__)
job_lock = threading.Lock()
model_cache = {}

# Single in-process job (may cover one video or a whole folder).
# The UI polls /api/status while it runs.
job = {
    "running": False,
    "phase": "idle",          # idle | scanning | allocating | extracting | done | error
    "message": "",
    "scanned_frames": 0,
    "total_frames": 0,
    "candidates": 0,
    "saved": 0,
    "requested": 0,
    "started_at": 0.0,
    # Batch tracking across multiple videos in a folder.
    "videos_total": 0,
    "videos_done": 0,
    "current_video": "",
    "result": None,           # {"reports_dir":..., "videos":[per-video result, ...]}
    "error": "",
}


# Independent job for the teacher/student disagreement feature, so it never
# touches the crop-balancer job state above. The UI polls /api/disagree/status.
disagree_job_lock = threading.Lock()
disagree_job = {
    "running": False,
    "phase": "idle",          # idle | scanning | done | stopped | error
    "message": "",
    "scanned_frames": 0,
    "total_frames": 0,
    "kept": 0,                # frames saved so far (current video)
    "started_at": 0.0,
    "videos_total": 0,
    "videos_done": 0,
    "current_video": "",
    "result": None,           # {"out_root":..., "videos":[per-video result, ...]}
    "error": "",
    "stop_requested": False,  # abort the whole batch
    "skip_requested": False,  # abort just the current video, continue batch
}


def list_videos_in_folder(folder):
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        return []
    vids = [
        str(folder / name)
        for name in os.listdir(folder)
        if (folder / name).is_file() and name.lower().endswith(VIDEO_EXTENSIONS)
    ]
    vids.sort()
    return vids


# --------------------------------------------------------------------------- #
# Model discovery (kept local so this file does not depend on web_app.py)
# --------------------------------------------------------------------------- #
def discover_pt_models(root_dirs):
    found = set()
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(".pt"):
                    found.add(str(Path(dirpath) / filename))
    return sorted(found)


def build_model_options_html(model_paths, selected_path):
    options = []
    for path in model_paths:
        sel = ' selected="selected"' if path == selected_path else ""
        # Show the parent folder + filename so similarly-named weights are distinct.
        p = Path(path)
        label = html.escape(f"{p.parent.name}/{p.name}")
        value = html.escape(path, quote=True)
        options.append(f'<option value="{value}"{sel}>{label}</option>')
    if not options:
        options.append('<option value="">No .pt models found in models/</option>')
    return "".join(options)


def get_or_load_model(model_path):
    if model_path not in model_cache:
        model_cache[model_path] = YOLO(model_path)
    return model_cache[model_path]


def default_person_model(all_pt):
    # Prefer a model whose filename matches the preferred default, else first found.
    for path in all_pt:
        if Path(path).name == DEFAULT_PERSON_MODEL_NAME:
            return path
    return all_pt[0] if all_pt else ""


def person_class_ids(model):
    """
    Resolve which class id(s) mean "person" for a given model, since teacher and
    student can label the same concept at different indices. Match by class name
    (case-insensitive substring 'person'); fall back to COCO id 0 if no name maps.
    """
    names = getattr(model, "names", None)
    ids = []
    if isinstance(names, dict):
        for cid, cname in names.items():
            if isinstance(cname, str) and "person" in cname.lower():
                try:
                    ids.append(int(cid))
                except (TypeError, ValueError):
                    continue
    elif isinstance(names, (list, tuple)):
        for cid, cname in enumerate(names):
            if isinstance(cname, str) and "person" in cname.lower():
                ids.append(cid)
    return sorted(set(ids)) if ids else [PERSON_CLASS_ID]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def cell_for_centroid(cx, cy, w, h, rows, cols):
    col = min(cols - 1, max(0, int(cx / w * cols)))
    row = min(rows - 1, max(0, int(cy / h * rows)))
    return row, col


def draw_grid(frame, rows, cols, color=(80, 200, 255)):
    out = frame.copy()
    h, w = out.shape[:2]
    for r in range(1, rows):
        y = int(r * h / rows)
        cv2.line(out, (0, y), (w, y), color, 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(c * w / cols)
        cv2.line(out, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    return out


def render_heatmap(frame, counts, rows, cols, alpha=0.5):
    """counts: rows x cols int array. Blend a colormapped grid over the frame."""
    h, w = frame.shape[:2]
    cnt = np.array(counts, dtype=np.float32).reshape(rows, cols)
    mx = float(cnt.max()) if cnt.size else 0.0
    norm = (cnt / mx) if mx > 0 else cnt
    small = (norm * 255.0).astype(np.uint8)
    color_small = cv2.applyColorMap(small, cv2.COLORMAP_JET)
    heat = cv2.resize(color_small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = cv2.addWeighted(frame, 1.0 - alpha, heat, alpha, 0)

    for r in range(rows + 1):
        y = min(h - 1, int(r * h / rows))
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(cols + 1):
        x = min(w - 1, int(c * w / cols))
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)

    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * w / cols)
            cy = int((r + 0.5) * h / rows)
            txt = str(int(cnt[r][c]))
            cv2.putText(out, txt, (cx - 12, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, txt, (cx - 12, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def pick_even(items_sorted, k):
    """Pick k items evenly spaced across the list (preserves time spread)."""
    n = len(items_sorted)
    if k >= n:
        return list(items_sorted)
    return [items_sorted[int(i * n / k)] for i in range(k)]


def allocate_balanced(cand_by_cell, total, rows, cols):
    """
    cand_by_cell: dict (row,col) -> list of candidate dicts.
    Returns dict cell -> number of crops to take, balanced so the sum is
    min(total, available), pulling deficits from the richest cells.
    """
    import heapq

    g = rows * cols
    base = total // g if g else 0
    take = {}
    for cell, lst in cand_by_cell.items():
        take[cell] = min(base, len(lst))
    allocated = sum(take.values())
    deficit = total - allocated

    # Max-heap on remaining surplus per cell; back-fill the deficit fairly.
    heap = []
    for cell, lst in cand_by_cell.items():
        surplus = len(lst) - take[cell]
        if surplus > 0:
            heap.append((-surplus, cell))
    heapq.heapify(heap)
    while deficit > 0 and heap:
        neg_surplus, cell = heapq.heappop(heap)
        surplus = -neg_surplus
        take[cell] += 1
        surplus -= 1
        deficit -= 1
        if surplus > 0:
            heapq.heappush(heap, (-surplus, cell))
    return take


# --------------------------------------------------------------------------- #
# The processing job (runs in a background thread)
# --------------------------------------------------------------------------- #
def _set(**kw):
    with job_lock:
        job.update(kw)


def clear_output_dir(out_dir: Path):
    """Remove our own prior outputs (jpgs/manifest/heatmap) only, never other files."""
    if not out_dir.is_dir():
        return
    for p in out_dir.iterdir():
        if p.is_file() and (p.suffix.lower() == ".jpg" or p.name in ("manifest.json", "heatmap.png")):
            try:
                p.unlink()
            except OSError:
                pass


def process_one_video(video_path, model, total, rows, cols, conf, stride, reports_dir):
    """
    Run the full scan -> balance -> extract -> heatmap pipeline for ONE video.
    Crops go to person_crops/<stem>/ ; heatmap + manifest go to reports_dir,
    named with the video stem. Returns a per-video result dict (or raises).
    Progress is reported live via _set().
    """
    video_path = str(Path(video_path).expanduser())
    video_stem = Path(video_path).stem

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Could not read first frame: {video_path}")
    h, w = first_frame.shape[:2]

    _set(phase="scanning", message=f"Detecting persons in {video_stem}...",
         total_frames=total_frames, scanned_frames=0, candidates=0, saved=0,
         current_video=video_stem)

    # --- Pass 1: scan, collect candidate metadata per cell (no pixels kept) ---
    cand_by_cell = {}
    cand_total = 0
    idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        if not cap.grab():
            break
        if idx % stride == 0:
            ok2, frame = cap.retrieve()
            if ok2 and frame is not None:
                res = model.predict(frame, conf=conf, classes=[PERSON_CLASS_ID], verbose=False)[0]
                if res.boxes is not None:
                    for b in res.boxes:
                        if int(b.cls[0]) != PERSON_CLASS_ID:
                            continue
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        cxc = (x1 + x2) / 2.0
                        cyc = (y1 + y2) / 2.0
                        cell = cell_for_centroid(cxc, cyc, w, h, rows, cols)
                        cand_by_cell.setdefault(cell, []).append(
                            {"frame_idx": idx, "box": (x1, y1, x2, y2), "conf": float(b.conf[0])}
                        )
                        cand_total += 1
            if idx % (stride * 10) == 0:
                _set(scanned_frames=idx, candidates=cand_total)
        idx += 1
    _set(scanned_frames=idx, candidates=cand_total, phase="allocating",
         message=f"Balancing crops for {video_stem}...")

    # --- Pass 2: balanced allocation ---
    take = allocate_balanced(cand_by_cell, total, rows, cols)
    selected_by_frame = {}
    per_cell_counts = [[0] * cols for _ in range(rows)]
    for cell, lst in cand_by_cell.items():
        k = take.get(cell, 0)
        if k <= 0:
            continue
        lst_sorted = sorted(lst, key=lambda d: d["frame_idx"])
        for d in pick_even(lst_sorted, k):
            selected_by_frame.setdefault(d["frame_idx"], []).append((d, cell))
    total_selected = sum(take.values())

    # --- Pass 3: seek to each selected frame and save crops (flushed now) ---
    _set(phase="extracting", message=f"Saving crops for {video_stem}...")
    out_dir = CROPS_ROOT / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(out_dir)

    manifest = {
        "video": video_path,
        "video_stem": video_stem,
        "grid": {"rows": rows, "cols": cols},
        "requested_total": total,
        "frame_stride": stride,
        "person_conf": conf,
        "crops": [],
    }
    saved = 0
    sample_crops = []
    for fidx in sorted(selected_by_frame.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        fh, fw = frame.shape[:2]
        for k, (d, cell) in enumerate(selected_by_frame[fidx]):
            x1, y1, x2, y2 = d["box"]
            x1 = max(0, x1 - PERSON_CROP_PADDING)
            y1 = max(0, y1 - PERSON_CROP_PADDING)
            x2 = min(fw - 1, x2 + PERSON_CROP_PADDING)
            y2 = min(fh - 1, y2 + PERSON_CROP_PADDING)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            r, c = cell
            fname = f"r{r}_c{c}_f{fidx:06d}_{k}.jpg"
            cv2.imwrite(str(out_dir / fname), crop)
            per_cell_counts[r][c] += 1
            manifest["crops"].append(
                {"file": fname, "cell": [r, c], "frame": fidx,
                 "box": [x1, y1, x2, y2], "conf": round(d["conf"], 4)}
            )
            saved += 1
            if len(sample_crops) < 60:
                sample_crops.append(fname)
        _set(saved=saved)

    manifest["saved_total"] = saved
    manifest["per_cell_counts"] = per_cell_counts

    # --- Heatmap + manifest into the shared reports folder, named by video ---
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reports_dir / f"{video_stem}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    heat = render_heatmap(first_frame, per_cell_counts, rows, cols)
    heatmap_path = reports_dir / f"{video_stem}_heatmap.png"
    cv2.imwrite(str(heatmap_path), heat)
    cap.release()

    return {
        "video": video_path,
        "video_stem": video_stem,
        "output_dir": str(out_dir),
        "requested": total,
        "saved": saved,
        "shortfall": max(0, total - saved),
        "candidates": cand_total,
        "grid": {"rows": rows, "cols": cols},
        "per_cell_counts": per_cell_counts,
        "heatmap_path": str(heatmap_path),
        "manifest_path": str(manifest_path),
        "sample_crops": sample_crops,
        "complete": total_selected >= total,
    }


def run_batch(input_path, model_path, total, rows, cols, conf, stride):
    try:
        input_path = str(Path(input_path).expanduser())
        if not model_path or not os.path.isfile(model_path):
            _set(running=False, phase="error", error="Selected model file not found.")
            return

        # Resolve the work list: a single file, or every video in a folder.
        if os.path.isdir(input_path):
            videos = list_videos_in_folder(input_path)
            if not videos:
                _set(running=False, phase="error", error="No videos found in that folder.")
                return
        elif os.path.isfile(input_path):
            videos = [input_path]
        else:
            _set(running=False, phase="error", error="Path is not a file or folder.")
            return

        reports_dir = REPORTS_ROOT
        model = get_or_load_model(model_path)

        _set(videos_total=len(videos), videos_done=0,
             result={"reports_dir": str(reports_dir), "videos": []})

        for i, vpath in enumerate(videos):
            stem = Path(vpath).stem
            _set(message=f"Video {i + 1}/{len(videos)}: {stem}")
            try:
                vres = process_one_video(vpath, model, total, rows, cols, conf, stride, reports_dir)
            except Exception as exc:  # noqa: BLE001 - skip a bad video, keep the batch going
                vres = {"video": vpath, "video_stem": stem, "error": f"{type(exc).__name__}: {exc}",
                        "saved": 0, "requested": total}
            # Push this video's result as soon as it is done, then continue.
            with job_lock:
                job["result"]["videos"].append(vres)
                job["videos_done"] = i + 1

        with job_lock:
            done_results = job["result"]["videos"]
        total_saved = sum(int(v.get("saved", 0)) for v in done_results)
        ok_count = sum(1 for v in done_results if not v.get("error"))
        msg = f"Done. {ok_count}/{len(videos)} videos processed, {total_saved} crops saved total."
        _set(running=False, phase="done", message=msg)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        _set(running=False, phase="error", error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Teacher / student disagreement feature (separate pipeline)
# --------------------------------------------------------------------------- #
def _dset(**kw):
    with disagree_job_lock:
        disagree_job.update(kw)


def _disagree_flags():
    """Return (stop_requested, skip_requested) for the running job."""
    with disagree_job_lock:
        return disagree_job["stop_requested"], disagree_job["skip_requested"]


def _person_cells(res, person_ids, w, h, rows, cols):
    """
    Return (occupied_cells set, boxes list) for one model's result.
    A cell is 'occupied' if a person centroid falls in it.
    """
    occupied = set()
    boxes = []
    if res.boxes is None:
        return occupied, boxes
    id_set = set(person_ids)
    for b in res.boxes:
        if int(b.cls[0]) not in id_set:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        cxc = (x1 + x2) / 2.0
        cyc = (y1 + y2) / 2.0
        cell = cell_for_centroid(cxc, cyc, w, h, rows, cols)
        occupied.add(cell)
        boxes.append((x1, y1, x2, y2))
    return occupied, boxes


def draw_disagreement_overlay(frame, rows, cols, teacher_boxes, student_boxes, miss_cells):
    """
    Grid + teacher boxes (green) + student boxes (red) + highlighted cells
    where teacher saw a person but student did not (yellow fill).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Shade the disagreement cells.
    if miss_cells:
        shade = out.copy()
        for (r, c) in miss_cells:
            x0 = int(c * w / cols)
            y0 = int(r * h / rows)
            x1 = int((c + 1) * w / cols)
            y1 = int((r + 1) * h / rows)
            cv2.rectangle(shade, (x0, y0), (x1, y1), (0, 220, 255), -1)
        out = cv2.addWeighted(out, 0.7, shade, 0.3, 0)

    # Grid lines.
    for r in range(1, rows):
        y = int(r * h / rows)
        cv2.line(out, (0, y), (w, y), (80, 200, 255), 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(c * w / cols)
        cv2.line(out, (x, 0), (x, h), (80, 200, 255), 1, cv2.LINE_AA)

    # Teacher boxes (green), student boxes (red).
    for (x1, y1, x2, y2) in teacher_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)
    for (x1, y1, x2, y2) in student_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 230), 2, cv2.LINE_AA)

    cv2.putText(out, "teacher=green  student=red  miss=yellow", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, "teacher=green  student=red  miss=yellow", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def process_one_video_disagree(video_path, teacher, student, teacher_ids, student_ids,
                               rows, cols, teacher_conf, student_conf, stride, per_cell,
                               out_root):
    """
    Scan one video. Keep every sampled frame that has at least one grid block where
    the TEACHER detected a person but the STUDENT did not. Save the full frame under
    <stem>/frames and an annotated overlay under <stem>/overlays, plus a manifest.

    per_cell > 0 caps how many kept frames may count toward each grid block; once a
    block is full it no longer qualifies a frame. per_cell <= 0 means unlimited.
    Returns (result_dict, stopped_whole_batch_bool).
    """
    video_path = str(Path(video_path).expanduser())
    video_stem = Path(video_path).stem
    cap_target = per_cell if per_cell and per_cell > 0 else 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Videos in one run share the chosen destination's flat frames/ + overlays/;
    # filenames are prefixed with the video stem so they never collide.
    out_dir = out_root
    frames_dir = out_dir / "frames"
    overlays_dir = out_dir / "overlays"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "video": video_path,
        "video_stem": video_stem,
        "grid": {"rows": rows, "cols": cols},
        "frame_stride": stride,
        "teacher_conf": teacher_conf,
        "student_conf": student_conf,
        "per_cell_limit": cap_target,
        "teacher_person_class_ids": teacher_ids,
        "student_person_class_ids": student_ids,
        "frames": [],
    }

    _dset(phase="scanning", message=f"Comparing models on {video_stem}...",
          total_frames=total_frames, scanned_frames=0, kept=0, current_video=video_stem)

    cell_counts = {}              # (r,c) -> kept frames counted toward that block
    total_cells = rows * cols
    kept = 0
    idx = 0
    stopped_batch = False
    sample_overlays = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        if not cap.grab():
            break
        if idx % stride == 0:
            stop_req, skip_req = _disagree_flags()
            if stop_req:
                stopped_batch = True
                break
            if skip_req:
                break
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                t_res = teacher.predict(frame, conf=teacher_conf, classes=teacher_ids, verbose=False)[0]
                s_res = student.predict(frame, conf=student_conf, classes=student_ids, verbose=False)[0]
                t_cells, t_boxes = _person_cells(t_res, teacher_ids, w, h, rows, cols)
                s_cells, s_boxes = _person_cells(s_res, student_ids, w, h, rows, cols)
                # Blocks the teacher filled but the student missed.
                miss_cells = sorted(t_cells - s_cells)
                # Respect the per-block quota: keep only the still-hungry blocks.
                if cap_target:
                    eligible = [c for c in miss_cells if cell_counts.get(c, 0) < cap_target]
                else:
                    eligible = miss_cells
                if eligible:
                    fname = f"{video_stem}_{idx:06d}.jpg"
                    cv2.imwrite(str(frames_dir / fname), frame)
                    overlay = draw_disagreement_overlay(frame, rows, cols, t_boxes, s_boxes, eligible)
                    cv2.imwrite(str(overlays_dir / fname), overlay)
                    for c in eligible:
                        cell_counts[c] = cell_counts.get(c, 0) + 1
                    manifest["frames"].append({
                        "file": fname,
                        "frame": idx,
                        "miss_cells": [list(c) for c in eligible],
                        "teacher_persons": len(t_boxes),
                        "student_persons": len(s_boxes),
                    })
                    kept += 1
                    if len(sample_overlays) < 60:
                        sample_overlays.append(fname)
                    # Early exit once every block has met its quota.
                    if cap_target and len(cell_counts) >= total_cells and \
                            all(v >= cap_target for v in cell_counts.values()):
                        break
            if idx % (stride * 10) == 0:
                _dset(scanned_frames=idx, kept=kept)
        idx += 1

    manifest["kept_total"] = kept
    manifest["scanned_frames"] = idx
    manifest["per_cell_kept"] = {f"{r}_{c}": n for (r, c), n in sorted(cell_counts.items())}
    # Per-video manifest, stem-prefixed so multiple videos don't overwrite each other.
    manifest_path = out_dir / f"{video_stem}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    cap.release()
    _dset(scanned_frames=idx, kept=kept)

    result = {
        "video": video_path,
        "video_stem": video_stem,
        "output_dir": str(out_dir),
        "frames_dir": str(frames_dir),
        "overlays_dir": str(overlays_dir),
        "scanned_frames": idx,
        "kept": kept,
        "grid": {"rows": rows, "cols": cols},
        "manifest_path": str(manifest_path),
        "sample_overlays": sample_overlays,
    }
    return result, stopped_batch


def run_batch_disagree(input_path, teacher_path, student_path, rows, cols,
                       teacher_conf, student_conf, stride, per_cell, dest, resume):
    try:
        input_path = str(Path(input_path).expanduser())
        out_root = DISAGREE_ROOT / dest
        out_root.mkdir(parents=True, exist_ok=True)
        if not teacher_path or not os.path.isfile(teacher_path):
            _dset(running=False, phase="error", error="Teacher model file not found.")
            return
        if not student_path or not os.path.isfile(student_path):
            _dset(running=False, phase="error", error="Student model file not found.")
            return

        if os.path.isdir(input_path):
            videos = list_videos_in_folder(input_path)
            if not videos:
                _dset(running=False, phase="error", error="No videos found in that folder.")
                return
        elif os.path.isfile(input_path):
            videos = [input_path]
        else:
            _dset(running=False, phase="error", error="Path is not a file or folder.")
            return

        # Resume: drop videos already finished (manifest present) in this destination.
        skipped = 0
        if resume:
            remaining = []
            for v in videos:
                if (out_root / f"{Path(v).stem}_manifest.json").is_file():
                    skipped += 1
                else:
                    remaining.append(v)
            videos = remaining
            if not videos:
                _dset(running=False, phase="done",
                      message=f"Nothing to do — all {skipped} video(s) already processed in '{dest}'.")
                return

        teacher = get_or_load_model(teacher_path)
        student = get_or_load_model(student_path)
        teacher_ids = person_class_ids(teacher)
        student_ids = person_class_ids(student)

        _dset(videos_total=len(videos), videos_done=0,
              result={"out_root": str(out_root), "videos": []})

        stopped = False
        for i, vpath in enumerate(videos):
            stem = Path(vpath).stem
            _dset(message=f"Video {i + 1}/{len(videos)}: {stem}")
            stop_this_batch = False
            try:
                vres, stop_this_batch = process_one_video_disagree(
                    vpath, teacher, student, teacher_ids, student_ids,
                    rows, cols, teacher_conf, student_conf, stride, per_cell, out_root)
            except Exception as exc:  # noqa: BLE001 - skip a bad video, keep going
                vres = {"video": vpath, "video_stem": stem,
                        "error": f"{type(exc).__name__}: {exc}", "kept": 0}
            # Consume a one-shot skip request; a stop ends the whole batch.
            with disagree_job_lock:
                disagree_job["result"]["videos"].append(vres)
                disagree_job["videos_done"] = i + 1
                disagree_job["skip_requested"] = False
            if stop_this_batch:
                stopped = True
                break

        with disagree_job_lock:
            done_results = disagree_job["result"]["videos"]
        total_kept = sum(int(v.get("kept", 0)) for v in done_results)
        ok_count = sum(1 for v in done_results if not v.get("error"))
        skip_note = f" (skipped {skipped} already done)" if skipped else ""
        if stopped:
            msg = f"Stopped. {ok_count}/{len(videos)} videos touched, {total_kept} frames kept total{skip_note}."
            _dset(running=False, phase="stopped", message=msg)
        else:
            msg = f"Done. {ok_count}/{len(videos)} videos processed, {total_kept} frames kept total{skip_note}."
            _dset(running=False, phase="done", message=msg)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        _dset(running=False, phase="error", error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _safe_serve(path_str):
    """Only allow serving files inside our output roots."""
    try:
        p = Path(path_str).resolve()
    except OSError:
        return None
    for root in (CROPS_ROOT.resolve(), REPORTS_ROOT.resolve(), DISAGREE_ROOT.resolve()):
        try:
            p.relative_to(root)
            return p if p.is_file() else None
        except ValueError:
            continue
    return None


def _safe_basename(name):
    """Reject anything that isn't a plain filename (no separators, no traversal)."""
    name = str(name or "")
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        return None
    if name != os.path.basename(name):
        return None
    return name


TRASH_DIRNAME = ".trash"


def _review_child(folder, kind, name, trash=False):
    """
    Resolve <folder>[/.trash]/<kind>/<name> and confirm it stays inside that base.
    Returns the resolved Path or None. kind must be 'frames' or 'overlays'.
    """
    if kind not in ("frames", "overlays"):
        return None
    name = _safe_basename(name)
    if not name:
        return None
    parts = [TRASH_DIRNAME, kind] if trash else [kind]
    try:
        base = Path(folder).expanduser().joinpath(*parts).resolve()
        target = (base / name).resolve()
        target.relative_to(base)
    except (OSError, ValueError):
        return None
    return target


PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>__TITLE__</title>
  <style>
    :root { --bg:#0f1115; --panel:#161a22; --panel2:#1a1d24; --border:#2b3140; --border2:#3a4150;
            --text:#eef1f6; --muted:#9aa4b4; --accent:#5f89ff; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:'Segoe UI',system-ui,Arial,sans-serif; background:var(--bg); color:var(--text); }
    .wrap { max-width: 1500px; margin:0 auto; padding:18px; }
    h1 { font-size:20px; margin:0 0 14px; }
    .grid { display:grid; grid-template-columns: 380px minmax(0,1fr); gap:16px; align-items:start; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:14px; }
    .panel-title { font-size:12px; text-transform:uppercase; letter-spacing:.4px; color:#cdd5e1; font-weight:600; margin:0 0 10px; }
    .item { display:flex; flex-direction:column; gap:5px; margin-bottom:10px; }
    label { font-size:12px; color:var(--muted); }
    input, select, button { width:100%; min-height:36px; padding:8px 10px; background:var(--panel2);
            color:var(--text); border:1px solid var(--border2); border-radius:6px; outline:none; font-size:13px; }
    input:focus, select:focus { border-color:var(--accent); }
    .row2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    button { cursor:pointer; font-weight:600; transition:filter .12s; }
    button:hover { filter:brightness(1.15); }
    .btn-run { background:#1b5e20; border-color:#2e7d32; min-height:42px; font-size:15px; }
    .btn-prev { background:#23314b; border-color:#33507f; }
    .preview-img, .heatmap-img { width:100%; border:1px solid var(--border); border-radius:8px; background:#000; display:block; }
    .meta { font-size:13px; color:#bbb; }
    .progress { height:10px; background:#0c0e12; border:1px solid var(--border); border-radius:6px; overflow:hidden; margin:8px 0; }
    .progress > div { height:100%; width:0%; background:linear-gradient(90deg,#3b62d6,#5f89ff); transition:width .3s; }
    .gallery { display:grid; grid-template-columns: repeat(auto-fill,minmax(96px,1fr)); gap:6px; margin-top:10px; max-height:260px; overflow:auto; }
    .gallery img { width:100%; height:96px; object-fit:cover; border-radius:6px; border:1px solid var(--border); }
    .vid-block { border:1px solid var(--border); border-radius:8px; padding:10px; margin-top:10px; }
    .vid-block h3 { font-size:14px; margin:0 0 6px; }
    .heatmap-thumb { width:100%; max-height:300px; object-fit:contain; border-radius:6px; border:1px solid var(--border); background:#000; }
    .pill { display:inline-block; background:#23314b; border:1px solid #33507f; border-radius:999px; padding:2px 10px; font-size:12px; margin:2px 4px 2px 0; }
    .hidden { display:none; }
    .err { color:#fda4af; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>__TITLE__</h1>
    <div class="grid">
      <div class="left">
        <div class="panel">
          <div class="panel-title">Inputs</div>
          <div class="item"><label for="video_path">Video file OR folder of videos</label><input id="video_path" placeholder="/path/to/video.mp4  or  /path/to/folder" /></div>
          <div class="item"><label for="model_path">Model</label><select id="model_path">__MODEL_OPTIONS__</select></div>
          <div class="item"><label for="total">Total crops required (per video)</label><input id="total" type="number" min="1" value="200" /></div>
          <div class="row2">
            <div class="item"><label for="rows">Grid rows</label><input id="rows" type="number" min="1" max="20" value="4" /></div>
            <div class="item"><label for="cols">Grid cols</label><input id="cols" type="number" min="1" max="20" value="4" /></div>
          </div>
          <div class="row2">
            <div class="item"><label for="conf">Person conf</label><input id="conf" type="number" min="0" max="1" step="0.05" value="0.4" /></div>
            <div class="item"><label for="stride">Frame stride</label><input id="stride" type="number" min="1" max="120" value="15" /></div>
          </div>
          <div class="row2">
            <button class="btn-prev" onclick="loadPreview()">Preview grid</button>
            <button class="btn-run" id="run_btn" onclick="runJob()">Run</button>
          </div>
        </div>
        <div class="panel" id="status_panel">
          <div class="panel-title">Status</div>
          <div class="meta" id="status_msg">Idle.</div>
          <div class="progress"><div id="bar"></div></div>
          <div class="meta" id="status_detail"></div>
        </div>
      </div>

      <div class="right">
        <div class="panel">
          <div class="panel-title">Preview / Heatmap</div>
          <img id="preview_img" class="preview-img" />
          <div class="meta" id="preview_note" style="margin-top:8px;">Enter a video path and click "Preview grid".</div>
        </div>
        <div class="panel hidden" id="result_panel">
          <div class="panel-title">Results</div>
          <div id="result_overview" class="meta"></div>
          <div id="result_videos"></div>
        </div>
      </div>
    </div>
  </div>

<script>
  let pollTimer = null;
  const SAVED = __SETTINGS_JSON__;

  // Restore remembered inputs (model_path comes pre-selected server-side).
  function applySaved(){
    if (SAVED.video_path) document.getElementById("video_path").value = SAVED.video_path;
    ["total","rows","cols","conf","stride"].forEach(k => {
      if (SAVED[k] !== undefined && SAVED[k] !== null) document.getElementById(k).value = SAVED[k];
    });
  }

  function val(id){ return document.getElementById(id).value; }

  applySaved();

  function previewQuery(){
    const p = new URLSearchParams({
      path: val("video_path"), rows: val("rows"), cols: val("cols"),
      t: String(Date.now()),
    });
    return p.toString();
  }

  function loadPreview(){
    if(!val("video_path")){ document.getElementById("preview_note").textContent = "Please enter a video path."; return; }
    const img = document.getElementById("preview_img");
    img.onerror = () => { document.getElementById("preview_note").textContent = "Could not load a frame from that path."; };
    img.onload = () => { document.getElementById("preview_note").textContent = "Grid preview (" + val("rows") + " x " + val("cols") + ")."; };
    img.src = "/api/preview?" + previewQuery();
  }

  async function runJob(){
    const payload = {
      video_path: val("video_path"),
      model_path: val("model_path"),
      total: Number(val("total")),
      rows: Number(val("rows")),
      cols: Number(val("cols")),
      conf: Number(val("conf")),
      stride: Number(val("stride")),
    };
    const res = await fetch("/api/run", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload) });
    const data = await res.json();
    if(!data.ok){ document.getElementById("status_msg").innerHTML = '<span class="err">'+ (data.error||"Failed to start") +'</span>'; return; }
    document.getElementById("run_btn").disabled = true;
    document.getElementById("result_panel").classList.add("hidden");
    if(pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(poll, 700);
    poll();
  }

  async function poll(){
    const res = await fetch("/api/status");
    const s = await res.json();
    const phaseText = { scanning:"Scanning", allocating:"Allocating", extracting:"Saving crops", done:"Done", error:"Error", idle:"Idle" }[s.phase] || s.phase;
    document.getElementById("status_msg").textContent = phaseText + " - " + (s.message || "");
    let pct = 0;
    if(s.phase === "scanning" && s.total_frames > 0) pct = 100 * s.scanned_frames / s.total_frames;
    else if(s.phase === "allocating") pct = 100;
    else if(s.phase === "extracting" && s.requested > 0) pct = 100 * s.saved / s.requested;
    else if(s.phase === "done") pct = 100;
    document.getElementById("bar").style.width = Math.min(100, pct) + "%";
    let detail = "";
    if(s.videos_total > 1) detail += "Videos: " + s.videos_done + "/" + s.videos_total + " | ";
    detail += "Frames: " + s.scanned_frames + "/" + s.total_frames + " | Candidates: " + s.candidates + " | Saved: " + s.saved;
    document.getElementById("status_detail").textContent = detail;

    // Render finished videos incrementally (push one video at a time).
    if(s.result && s.result.videos) showResults(s.result.videos);

    if(s.phase === "error"){
      clearInterval(pollTimer);
      document.getElementById("run_btn").disabled = false;
      document.getElementById("status_msg").innerHTML = '<span class="err">' + (s.error||"Error") + '</span>';
    }
    if(s.phase === "done"){
      clearInterval(pollTimer);
      document.getElementById("run_btn").disabled = false;
    }
  }

  function fileUrl(path){ return "/api/file?path=" + encodeURIComponent(path) + "&t=" + Date.now(); }

  function showResults(videos){
    if(!videos || !videos.length) return;
    const panel = document.getElementById("result_panel");
    panel.classList.remove("hidden");
    document.getElementById("result_overview").textContent = videos.length + " video(s) processed so far.";
    const container = document.getElementById("result_videos");
    container.innerHTML = "";
    videos.forEach(v => {
      const block = document.createElement("div");
      block.className = "vid-block";
      if(v.error){
        block.innerHTML = "<h3>" + v.video_stem + "</h3><div class='err'>" + v.error + "</div>";
        container.appendChild(block);
        return;
      }
      let html = "<h3>" + v.video_stem + "</h3>";
      html += "<div><b>Saved " + v.saved + "</b> of " + v.requested + " requested";
      if(v.shortfall > 0) html += " (" + v.shortfall + " short - only " + v.candidates + " persons detected)";
      html += "</div>";
      html += "<div style='margin:4px 0;color:#9aa4b4;'>Crops: " + v.output_dir + "</div>";
      html += "<img class='heatmap-thumb' src='" + fileUrl(v.heatmap_path) + "' />";
      html += "<div style='margin-top:6px;'>";
      for(let i=0;i<v.per_cell_counts.length;i++)
        for(let j=0;j<v.per_cell_counts[i].length;j++)
          html += "<span class='pill'>r"+i+" c"+j+": "+v.per_cell_counts[i][j]+"</span>";
      html += "</div>";
      html += "<div class='gallery'>";
      (v.sample_crops||[]).forEach(fn => {
        html += "<img src='" + fileUrl(v.output_dir + "/" + fn) + "' />";
      });
      html += "</div>";
      block.innerHTML = html;
      container.appendChild(block);
    });
  }
</script>
</body>
</html>
"""


DISAGREE_TITLE = "Teacher / Student Disagreement Frames"

LANDING_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Crop Tools</title>
  <style>
    :root { --bg:#0f1115; --panel:#161a22; --border:#2b3140; --text:#eef1f6; --muted:#9aa4b4; --accent:#5f89ff; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:'Segoe UI',system-ui,Arial,sans-serif; background:var(--bg); color:var(--text); }
    .wrap { max-width: 980px; margin:0 auto; padding:40px 18px; }
    h1 { font-size:24px; margin:0 0 6px; }
    .sub { color:var(--muted); margin:0 0 28px; font-size:14px; }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:20px; }
    .card { display:block; text-decoration:none; color:inherit; background:var(--panel);
            border:1px solid var(--border); border-radius:14px; padding:22px; transition:border-color .15s, transform .15s; }
    .card:hover { border-color:var(--accent); transform:translateY(-2px); }
    .card h2 { font-size:18px; margin:0 0 8px; }
    .card p { color:var(--muted); font-size:13px; line-height:1.5; margin:0; }
    .tag { display:inline-block; font-size:11px; text-transform:uppercase; letter-spacing:.5px;
           color:#cdd5e1; background:#23314b; border:1px solid #33507f; border-radius:999px; padding:2px 10px; margin-bottom:12px; }
    @media (max-width:720px){ .cards { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Crop Tools</h1>
    <p class="sub">Choose a tool.</p>
    <div class="cards">
      <a class="card" href="/balancer">
        <div class="tag">Existing</div>
        <h2>Person Crop Balancer</h2>
        <p>Detect persons across a video, balance crops across a grid so every region is fairly
           represented, and export a chosen total with a heatmap.</p>
      </a>
      <a class="card" href="/disagreement">
        <div class="tag">New</div>
        <h2>Teacher / Student Disagreement</h2>
        <p>Pick a teacher and a student model. Keep every full frame where, in some grid block,
           the teacher detects a person but the student misses it &mdash; ideal for mining the
           student's failure cases.</p>
      </a>
      <a class="card" href="/review">
        <div class="tag">New</div>
        <h2>Review &amp; Clean Up</h2>
        <p>Point at a folder with <code>frames/</code> and <code>overlays/</code>. Flip through the
           overlays one at a time and delete a pair &mdash; removing both the overlay and its frame
           from their respective folders in one click.</p>
      </a>
    </div>
  </div>
</body>
</html>
"""

DISAGREE_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>__TITLE__</title>
  <style>
    :root { --bg:#0f1115; --panel:#161a22; --panel2:#1a1d24; --border:#2b3140; --border2:#3a4150;
            --text:#eef1f6; --muted:#9aa4b4; --accent:#5f89ff; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:'Segoe UI',system-ui,Arial,sans-serif; background:var(--bg); color:var(--text); }
    .wrap { max-width: 1500px; margin:0 auto; padding:18px; }
    h1 { font-size:20px; margin:0 0 14px; }
    a.back { color:var(--muted); text-decoration:none; font-size:13px; }
    a.back:hover { color:var(--accent); }
    .grid { display:grid; grid-template-columns: 380px minmax(0,1fr); gap:16px; align-items:start; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:14px; }
    .panel-title { font-size:12px; text-transform:uppercase; letter-spacing:.4px; color:#cdd5e1; font-weight:600; margin:0 0 10px; }
    .item { display:flex; flex-direction:column; gap:5px; margin-bottom:10px; }
    label { font-size:12px; color:var(--muted); }
    input, select, button { width:100%; min-height:36px; padding:8px 10px; background:var(--panel2);
            color:var(--text); border:1px solid var(--border2); border-radius:6px; outline:none; font-size:13px; }
    input:focus, select:focus { border-color:var(--accent); }
    .row2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    button { cursor:pointer; font-weight:600; transition:filter .12s; }
    button:hover { filter:brightness(1.15); }
    .btn-run { background:#1b5e20; border-color:#2e7d32; min-height:42px; font-size:15px; }
    .btn-prev { background:#23314b; border-color:#33507f; }
    .btn-skip { background:#5a4a12; border-color:#8a6d1c; }
    .btn-stop { background:#5e1b1b; border-color:#7d2e2e; }
    button:disabled { opacity:.45; cursor:not-allowed; filter:none; }
    .preview-img { width:100%; border:1px solid var(--border); border-radius:8px; background:#000; display:block; }
    .meta { font-size:13px; color:#bbb; }
    .progress { height:10px; background:#0c0e12; border:1px solid var(--border); border-radius:6px; overflow:hidden; margin:8px 0; }
    .progress > div { height:100%; width:0%; background:linear-gradient(90deg,#3b62d6,#5f89ff); transition:width .3s; }
    .gallery { display:grid; grid-template-columns: repeat(auto-fill,minmax(150px,1fr)); gap:6px; margin-top:10px; max-height:420px; overflow:auto; }
    .gallery img { width:100%; height:120px; object-fit:cover; border-radius:6px; border:1px solid var(--border); }
    .vid-block { border:1px solid var(--border); border-radius:8px; padding:10px; margin-top:10px; }
    .vid-block h3 { font-size:14px; margin:0 0 6px; }
    .hidden { display:none; }
    .err { color:#fda4af; }
  </style>
</head>
<body>
  <div class="wrap">
    <a class="back" href="/">&larr; Back to tools</a>
    <h1>__TITLE__</h1>
    <div class="grid">
      <div class="left">
        <div class="panel">
          <div class="panel-title">Inputs</div>
          <div class="item"><label for="video_path">Video file OR folder of videos</label><input id="video_path" placeholder="/path/to/video.mp4  or  /path/to/folder" onblur="updateResumeNote()" /></div>
          <div class="item"><label for="teacher_model_path">Teacher model</label><select id="teacher_model_path">__TEACHER_OPTIONS__</select></div>
          <div class="item"><label for="student_model_path">Student model</label><select id="student_model_path">__STUDENT_OPTIONS__</select></div>
          <div class="item"><label for="dest">Output destination (in disagreement_frames/)</label><select id="dest" onchange="onDestChange()">__DEST_OPTIONS__</select></div>
          <div class="item hidden" id="new_dest_item" style="display:none;"><label for="new_dest">New folder name</label><input id="new_dest" placeholder="e.g. egypt" /></div>
          <div class="row2">
            <div class="item"><label for="rows">Grid rows</label><input id="rows" type="number" min="1" max="20" value="4" /></div>
            <div class="item"><label for="cols">Grid cols</label><input id="cols" type="number" min="1" max="20" value="4" /></div>
          </div>
          <div class="row2">
            <div class="item"><label for="teacher_conf">Teacher conf</label><input id="teacher_conf" type="number" min="0" max="1" step="0.05" value="0.4" /></div>
            <div class="item"><label for="student_conf">Student conf</label><input id="student_conf" type="number" min="0" max="1" step="0.05" value="0.4" /></div>
          </div>
          <div class="row2">
            <div class="item"><label for="stride">Frame stride</label><input id="stride" type="number" min="1" max="120" value="15" /></div>
            <div class="item"><label for="per_cell">Max images / grid block (0 = all)</label><input id="per_cell" type="number" min="0" value="0" /></div>
          </div>
          <div class="item" style="flex-direction:row; align-items:center; gap:8px;">
            <input id="resume" type="checkbox" style="width:auto; min-height:0;" onchange="updateResumeNote()" />
            <label for="resume" style="cursor:pointer;">Resume — skip videos already done in this destination</label>
          </div>
          <div class="meta" id="resume_note" style="margin:-4px 0 8px; color:#8fb0ff;"></div>
          <div class="row2">
            <button class="btn-prev" onclick="loadPreview()">Preview grid</button>
            <button class="btn-run" id="run_btn" onclick="runJob()">Run</button>
          </div>
          <div class="row2" style="margin-top:10px;">
            <button class="btn-skip" id="skip_btn" onclick="skipJob()" disabled>Skip video</button>
            <button class="btn-stop" id="stop_btn" onclick="stopJob()" disabled>Stop</button>
          </div>
        </div>
        <div class="panel" id="status_panel">
          <div class="panel-title">Status</div>
          <div class="meta" id="status_msg">Idle.</div>
          <div class="progress"><div id="bar"></div></div>
          <div class="meta" id="status_detail"></div>
        </div>
      </div>

      <div class="right">
        <div class="panel">
          <div class="panel-title">Grid preview</div>
          <img id="preview_img" class="preview-img" />
          <div class="meta" id="preview_note" style="margin-top:8px;">Enter a video path and click "Preview grid".</div>
        </div>
        <div class="panel hidden" id="result_panel">
          <div class="panel-title">Results (overlays shown)</div>
          <div id="result_overview" class="meta"></div>
          <div id="result_videos"></div>
        </div>
      </div>
    </div>
  </div>

<script>
  let pollTimer = null;
  const SAVED = __SETTINGS_JSON__;

  function applySaved(){
    if (SAVED.video_path) document.getElementById("video_path").value = SAVED.video_path;
    ["rows","cols","teacher_conf","student_conf","stride","per_cell"].forEach(k => {
      if (SAVED[k] !== undefined && SAVED[k] !== null) document.getElementById(k).value = SAVED[k];
    });
    document.getElementById("resume").checked = !!SAVED.resume;
    onDestChange();
  }
  function val(id){ return document.getElementById(id).value; }

  // Show the "new folder name" input only when the dropdown is on "New folder…".
  function onDestChange(){
    const isNew = document.getElementById("dest").value === "__new__";
    document.getElementById("new_dest_item").style.display = isNew ? "flex" : "none";
    updateResumeNote();
  }

  // When Resume is checked, tell the user how many videos will be skipped.
  async function updateResumeNote(){
    const note = document.getElementById("resume_note");
    if(!document.getElementById("resume").checked){ note.textContent = ""; return; }
    const path = val("video_path").trim();
    if(!path){ note.textContent = "Enter a video folder to see how many are already done."; return; }
    const q = new URLSearchParams({ path: path, dest: val("dest"), new_dest: val("new_dest"), t: String(Date.now()) });
    try {
      const res = await fetch("/api/disagree/resume_info?" + q.toString());
      const d = await res.json();
      if(!d.ok){ note.textContent = d.error === "Path not found." ? "That path was not found." : ""; return; }
      const where = d.dest ? ("'" + d.dest + "'") : "a new folder";
      note.textContent = "Resume: " + d.done + " of " + d.total + " video(s) already done in " + where
                       + " will be skipped — " + d.remaining + " will be processed.";
    } catch(e){ note.textContent = ""; }
  }

  applySaved();

  function setRunning(on){
    document.getElementById("run_btn").disabled = on;
    document.getElementById("stop_btn").disabled = !on;
    document.getElementById("skip_btn").disabled = !on;
  }

  // If a run is already in progress (e.g. you came back to this card), re-attach to it.
  (async function reattach(){
    try {
      const res = await fetch("/api/disagree/status");
      const s = await res.json();
      if(s.running){
        setRunning(true);
        if(pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(poll, 700);
        poll();
      }
    } catch(e){}
  })();

  function previewQuery(){
    const p = new URLSearchParams({
      path: val("video_path"), rows: val("rows"), cols: val("cols"), t: String(Date.now()),
    });
    return p.toString();
  }
  function loadPreview(){
    if(!val("video_path")){ document.getElementById("preview_note").textContent = "Please enter a video path."; return; }
    const img = document.getElementById("preview_img");
    img.onerror = () => { document.getElementById("preview_note").textContent = "Could not load a frame from that path."; };
    img.onload = () => { document.getElementById("preview_note").textContent = "Grid preview (" + val("rows") + " x " + val("cols") + ")."; };
    img.src = "/api/preview?" + previewQuery();
  }

  async function runJob(){
    const payload = {
      video_path: val("video_path"),
      teacher_model_path: val("teacher_model_path"),
      student_model_path: val("student_model_path"),
      rows: Number(val("rows")), cols: Number(val("cols")),
      teacher_conf: Number(val("teacher_conf")), student_conf: Number(val("student_conf")),
      stride: Number(val("stride")), per_cell: Number(val("per_cell")),
      dest: val("dest"), new_dest: val("new_dest"),
      resume: document.getElementById("resume").checked,
    };
    const res = await fetch("/api/disagree/run", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload) });
    const data = await res.json();
    if(!data.ok){ document.getElementById("status_msg").innerHTML = '<span class="err">'+ (data.error||"Failed to start") +'</span>'; return; }
    setRunning(true);
    document.getElementById("result_panel").classList.add("hidden");
    if(pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(poll, 700);
    poll();
  }

  async function stopJob(){
    document.getElementById("stop_btn").disabled = true;
    await fetch("/api/disagree/stop", { method:"POST" });
  }
  async function skipJob(){
    document.getElementById("skip_btn").disabled = true;
    await fetch("/api/disagree/skip", { method:"POST" });
  }

  async function poll(){
    const res = await fetch("/api/disagree/status");
    const s = await res.json();
    const phaseText = { scanning:"Scanning", done:"Done", stopped:"Stopped", error:"Error", idle:"Idle" }[s.phase] || s.phase;
    document.getElementById("status_msg").textContent = phaseText + " - " + (s.message || "");
    let pct = 0;
    if(s.phase === "scanning" && s.total_frames > 0) pct = 100 * s.scanned_frames / s.total_frames;
    else if(s.phase === "done" || s.phase === "stopped") pct = 100;
    document.getElementById("bar").style.width = Math.min(100, pct) + "%";
    let detail = "";
    if(s.videos_total > 1) detail += "Videos: " + s.videos_done + "/" + s.videos_total + " | ";
    detail += "Frames: " + s.scanned_frames + "/" + s.total_frames + " | Kept: " + s.kept;
    document.getElementById("status_detail").textContent = detail;
    // Re-enable Skip after a skip lands but the batch is still running.
    if(s.phase === "scanning" && !s.skip_requested && !s.stop_requested)
      document.getElementById("skip_btn").disabled = false;

    if(s.result && s.result.videos) showResults(s.result.videos);

    if(s.phase === "error"){
      clearInterval(pollTimer);
      setRunning(false);
      document.getElementById("status_msg").innerHTML = '<span class="err">' + (s.error||"Error") + '</span>';
    }
    if(s.phase === "done" || s.phase === "stopped"){
      clearInterval(pollTimer);
      setRunning(false);
    }
  }

  function fileUrl(path){ return "/api/file?path=" + encodeURIComponent(path) + "&t=" + Date.now(); }

  function showResults(videos){
    if(!videos || !videos.length) return;
    const panel = document.getElementById("result_panel");
    panel.classList.remove("hidden");
    document.getElementById("result_overview").textContent = videos.length + " video(s) processed so far.";
    const container = document.getElementById("result_videos");
    container.innerHTML = "";
    videos.forEach(v => {
      const block = document.createElement("div");
      block.className = "vid-block";
      if(v.error){
        block.innerHTML = "<h3>" + v.video_stem + "</h3><div class='err'>" + v.error + "</div>";
        container.appendChild(block);
        return;
      }
      let html = "<h3>" + v.video_stem + "</h3>";
      html += "<div><b>Kept " + v.kept + "</b> frame(s) with a teacher-saw / student-missed block";
      html += " (scanned " + v.scanned_frames + " sampled frames)</div>";
      html += "<div style='margin:4px 0;color:#9aa4b4;'>Full frames: " + v.frames_dir + "</div>";
      html += "<div style='margin:0 0 4px;color:#9aa4b4;'>Overlays: " + v.overlays_dir + "</div>";
      html += "<div class='gallery'>";
      (v.sample_overlays||[]).forEach(fn => {
        html += "<img src='" + fileUrl(v.overlays_dir + "/" + fn) + "' />";
      });
      html += "</div>";
      block.innerHTML = html;
      container.appendChild(block);
    });
  }
</script>
</body>
</html>
"""


REVIEW_TITLE = "Review & Clean Up"

REVIEW_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>__TITLE__</title>
  <style>
    :root { --bg:#0f1115; --panel:#161a22; --panel2:#1a1d24; --border:#2b3140; --border2:#3a4150;
            --text:#eef1f6; --muted:#9aa4b4; --accent:#5f89ff; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:'Segoe UI',system-ui,Arial,sans-serif; background:var(--bg); color:var(--text); }
    .wrap { max-width: 1100px; margin:0 auto; padding:18px; }
    h1 { font-size:20px; margin:0 0 14px; }
    a.back { color:var(--muted); text-decoration:none; font-size:13px; }
    a.back:hover { color:var(--accent); }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:14px; }
    .bar { display:flex; gap:10px; align-items:flex-end; }
    .bar .item { flex:1; display:flex; flex-direction:column; gap:5px; }
    label { font-size:12px; color:var(--muted); }
    input, select, button { min-height:38px; padding:8px 12px; background:var(--panel2); color:var(--text);
            border:1px solid var(--border2); border-radius:6px; outline:none; font-size:13px; }
    input, select { width:100%; }
    input:focus, select:focus { border-color:var(--accent); }
    button { cursor:pointer; font-weight:600; transition:filter .12s; }
    button:hover { filter:brightness(1.15); }
    button:disabled { opacity:.4; cursor:not-allowed; filter:none; }
    .btn-load { background:#23314b; border-color:#33507f; min-width:120px; }
    .btn-nav { background:#23314b; border-color:#33507f; min-width:110px; }
    .btn-del { background:#5e1b1b; border-color:#7d2e2e; min-width:140px; }
    .btn-restore { background:#1b5e20; border-color:#2e7d32; min-width:120px; }
    .tabs { display:flex; gap:8px; margin-top:12px; }
    .tab { background:var(--panel2); border-color:var(--border2); }
    .tab.active { background:#23314b; border-color:var(--accent); color:#fff; }
    .filters { display:flex; gap:10px; align-items:flex-end; margin-top:12px; }
    .filters .item { display:flex; flex-direction:column; gap:5px; }
    .filters .item.small { width:130px; }
    .badge { display:inline-block; background:#23314b; border:1px solid #33507f; border-radius:999px;
             padding:1px 8px; font-size:12px; margin-left:6px; color:#cdd5e1; }
    .meta { font-size:13px; color:#bbb; margin-top:8px; }
    .err { color:#fda4af; }
    .viewer-head { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px; }
    .viewer-head .name { font-size:13px; color:#cdd5e1; word-break:break-all; }
    .viewer-head .count { font-size:13px; color:var(--muted); white-space:nowrap; }
    .stage { display:flex; align-items:center; justify-content:center; background:#000;
             border:1px solid var(--border); border-radius:8px; min-height:300px; }
    .stage img { max-width:100%; max-height:72vh; display:block; border-radius:8px; }
    .missing { color:var(--muted); padding:60px 0; font-size:14px; }
    .controls { display:flex; gap:10px; justify-content:center; margin-top:12px; }
    .hint { text-align:center; color:var(--muted); font-size:12px; margin-top:8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <a class="back" href="/">&larr; Back to tools</a>
    <h1>__TITLE__</h1>
    <div class="panel">
      <div class="bar">
        <div class="item" style="max-width:260px;">
          <label for="dest_pick">Pick a disagreement folder</label>
          <select id="dest_pick" onchange="pickDest()">__REVIEW_DEST_OPTIONS__</select>
        </div>
        <div class="item">
          <label for="folder">…or type a folder with frames/ and overlays/</label>
          <input id="folder" placeholder="/path/to/disagreement_frames" />
        </div>
        <button class="btn-load" onclick="loadFolder()">Load</button>
      </div>
      <div class="meta" id="info">Pick a folder from the dropdown, or type a path and click Load.</div>
      <div class="tabs hidden" id="tabs" style="display:none;">
        <button class="tab active" id="tab_live" onclick="switchMode('live')">Review (<span id="live_n">0</span>)</button>
        <button class="tab" id="tab_trash" onclick="switchMode('trash')">Trash (<span id="trash_n">0</span>)</button>
      </div>
      <div class="filters hidden" id="filters" style="display:none;">
        <div class="item small">
          <label for="f_teacher">Teacher boxes</label>
          <input id="f_teacher" type="number" min="0" placeholder="any" oninput="onFilterChange()" />
        </div>
        <div class="item" style="width:210px;">
          <label for="f_student_op">Student boxes (vs teacher)</label>
          <div style="display:flex; gap:6px;">
            <select id="f_student_op" style="width:80px;" onchange="onFilterChange()">
              <option value="any">any</option>
              <option value="eq">=</option>
              <option value="gt">&gt;</option>
              <option value="lt">&lt;</option>
              <option value="ge">&ge;</option>
              <option value="le">&le;</option>
            </select>
            <input id="f_student" type="number" min="0" placeholder="= teacher" oninput="onFilterChange()" />
          </div>
        </div>
        <button class="btn-nav" onclick="clearFilter()">Clear filter</button>
        <div class="meta" id="filter_note" style="margin:0 0 6px;"></div>
      </div>
    </div>

    <div class="panel hidden" id="viewer" style="display:none;">
      <div class="viewer-head">
        <div class="name" id="v_name"></div>
        <div class="count" id="v_count"></div>
      </div>
      <div class="stage" id="v_stage"></div>
      <div class="controls">
        <button class="btn-nav" id="prev_btn" onclick="go(-1)">&larr; Prev</button>
        <span id="action_btns"></span>
        <button class="btn-nav" id="next_btn" onclick="go(1)">Next &rarr;</button>
      </div>
      <div class="hint" id="hint"></div>
    </div>
  </div>

<script>
  const SAVED_FOLDER = __SAVED_FOLDER__;
  let ROOT = "";
  let live = [];    // [{name, has_overlay, has_frame}]
  let trash = [];
  let mode = "live";
  let idx = 0;
  let busy = false;

  if (SAVED_FOLDER) document.getElementById("folder").value = SAVED_FOLDER;
  function val(id){ return document.getElementById(id).value; }
  function curList(){ return mode === "live" ? live : trash; }
  function imgUrl(kind, name){
    const base = mode === "trash" ? "/api/review/trash/img" : "/api/review/img";
    return base + "?root=" + encodeURIComponent(ROOT)
         + "&kind=" + kind + "&name=" + encodeURIComponent(name) + "&t=" + Date.now();
  }

  function pickDest(){
    const v = document.getElementById("dest_pick").value;
    if(!v) return;
    document.getElementById("folder").value = v;
    loadFolder();
  }

  function filterVal(id){
    const raw = document.getElementById(id).value.trim();
    if(raw === "") return null;
    const n = parseInt(raw, 10);
    return isNaN(n) ? null : n;
  }
  function studentOp(){ return document.getElementById("f_student_op").value; }
  function studentOpSymbol(){
    return {eq:"=", gt:">", lt:"<", ge:"\\u2265", le:"\\u2264"}[studentOp()] || "=";
  }
  // What the student count is compared against, as a display label:
  // the student box if filled, else the teacher box, else each image's own teacher count.
  function studentCmpLabel(){
    const sb = filterVal("f_student");
    if(sb !== null) return String(sb);
    const tb = filterVal("f_teacher");
    if(tb !== null) return String(tb);
    return "teacher";
  }
  // Does this image's student count satisfy the operator? With no number entered,
  // it compares the image's student count against its own teacher count.
  function studentMatch(it){
    const op = studentOp();
    if(op === "any") return true;                       // operator off
    let cmp = filterVal("f_student");
    if(cmp === null) cmp = filterVal("f_teacher");
    if(cmp === null) cmp = it.teacher_persons;          // per-image teacher count
    const v = it.student_persons;
    if(cmp === null || cmp === undefined || v === null || v === undefined) return false;
    switch(op){
      case "gt": return v >  cmp;
      case "lt": return v <  cmp;
      case "ge": return v >= cmp;
      case "le": return v <= cmp;
      default:   return v === cmp;
    }
  }
  // The list actually shown = source (live/trash) narrowed by the box-count filter.
  function currentView(){
    const src = mode === "live" ? live : trash;
    const tb = filterVal("f_teacher");
    return src.filter(it =>
      (tb === null || it.teacher_persons === tb) && studentMatch(it));
  }
  function onFilterChange(){ idx = 0; render(); }
  function clearFilter(){
    document.getElementById("f_teacher").value = "";
    document.getElementById("f_student").value = "";
    document.getElementById("f_student_op").value = "any";
    onFilterChange();
  }

  async function loadFolder(){
    const folder = val("folder").trim();
    const info = document.getElementById("info");
    document.getElementById("viewer").style.display = "none";
    document.getElementById("tabs").style.display = "none";
    document.getElementById("filters").style.display = "none";
    document.getElementById("f_teacher").value = "";
    document.getElementById("f_student").value = "";
    live = []; trash = []; idx = 0; mode = "live";
    if(!folder){ info.textContent = "Please enter a folder path."; return; }
    info.textContent = "Loading...";
    let data, tdata;
    try {
      const res = await fetch("/api/review/list?path=" + encodeURIComponent(folder) + "&t=" + Date.now());
      data = await res.json();
    } catch(e){ info.innerHTML = '<span class="err">Failed to load folder.</span>'; return; }
    if(!data.ok){ info.innerHTML = '<span class="err">' + (data.error||"Failed to load") + '</span>'; return; }
    ROOT = data.root;
    live = data.items || [];
    try {
      const res2 = await fetch("/api/review/trash/list?path=" + encodeURIComponent(folder) + "&t=" + Date.now());
      tdata = await res2.json();
      if(tdata.ok) trash = tdata.items || [];
    } catch(e){ /* trash is optional */ }
    info.textContent = live.length + " pair(s)." +
      (data.has_overlays_dir ? "" : " (no overlays/ subfolder found)") +
      (data.has_counts ? "" : " — no manifests found, so box-count filter is unavailable.");
    document.getElementById("tabs").style.display = "flex";
    document.getElementById("filters").style.display = "flex";
    updateTabs();
    switchMode("live");
  }

  function updateTabs(){
    document.getElementById("live_n").textContent = live.length;
    document.getElementById("trash_n").textContent = trash.length;
    document.getElementById("tab_live").classList.toggle("active", mode === "live");
    document.getElementById("tab_trash").classList.toggle("active", mode === "trash");
  }

  function switchMode(m){
    mode = m; idx = 0;
    updateTabs();
    render();
  }

  function fmt(n){ return (n === null || n === undefined) ? "?" : n; }
  function updateFilterNote(viewLen){
    const note = document.getElementById("filter_note");
    const tb = filterVal("f_teacher");
    const studentActive = (studentOp() !== "any");
    if(tb === null && !studentActive){ note.textContent = ""; return; }
    const src = mode === "live" ? live : trash;
    const parts = [];
    if(tb !== null) parts.push("teacher = " + tb);
    if(studentActive) parts.push("student " + studentOpSymbol() + " " + studentCmpLabel());
    note.textContent = "Filter (" + parts.join(", ") + "): showing " + viewLen + " of " + src.length + ".";
  }

  function render(){
    const view = currentView();
    const viewer = document.getElementById("viewer");
    updateFilterNote(view.length);
    if(!view.length){
      viewer.style.display = "none";
      return;
    }
    viewer.style.display = "block";
    if(idx < 0) idx = 0;
    if(idx > view.length - 1) idx = view.length - 1;
    const it = view[idx];
    document.getElementById("v_name").textContent = it.name;
    document.getElementById("v_count").innerHTML = (idx + 1) + " / " + view.length +
      "<span class='badge'>T:" + fmt(it.teacher_persons) + "</span>" +
      "<span class='badge'>S:" + fmt(it.student_persons) + "</span>";
    const stage = document.getElementById("v_stage");
    stage.innerHTML = it.has_overlay
      ? "<img src='" + imgUrl("overlays", it.name) + "' />"
      : "<div class='missing'>No overlay image &mdash; showing frame.</div>" +
        (it.has_frame ? "<img src='" + imgUrl("frames", it.name) + "' />" : "");
    document.getElementById("prev_btn").disabled = (idx <= 0);
    document.getElementById("next_btn").disabled = (idx >= view.length - 1);

    const actions = document.getElementById("action_btns");
    if(mode === "live"){
      actions.innerHTML = "<button class='btn-del' onclick='deleteCurrent()'>Delete both</button>";
      document.getElementById("hint").innerHTML = "Shortcuts: &larr;/&rarr; to navigate, Delete to send the pair to Trash.";
    } else {
      actions.innerHTML =
        "<button class='btn-restore' onclick='restoreCurrent()'>Restore</button> " +
        "<button class='btn-del' onclick='purgeCurrent()'>Delete forever</button>";
      document.getElementById("hint").innerHTML = "Trash &mdash; restore puts the pair back; delete forever is permanent.";
    }
  }

  function go(step){ idx += step; render(); }

  async function postName(url, name){
    const res = await fetch(url, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ root: ROOT, name: name }),
    });
    return res.json();
  }

  // Move the current item out of one list (optionally into another), then re-render.
  async function actOnCurrent(url, fromList, toList){
    if(busy) return;
    const view = currentView();
    if(!view.length) return;
    busy = true;
    const it = view[idx];
    let data;
    try { data = await postName(url, it.name); }
    catch(e){ busy = false; return; }
    if(data.ok){
      const i = fromList.indexOf(it);
      if(i >= 0) fromList.splice(i, 1);
      if(toList) toList.push(it);
      updateTabs();
      render();
    }
    busy = false;
  }

  function deleteCurrent(){ actOnCurrent("/api/review/delete", live, trash); }
  function restoreCurrent(){ actOnCurrent("/api/review/trash/restore", trash, live); }
  function purgeCurrent(){ actOnCurrent("/api/review/trash/purge", trash, null); }

  document.addEventListener("keydown", (e) => {
    if(document.activeElement && document.activeElement.tagName === "INPUT") return;
    if(!currentView().length) return;
    if(e.key === "ArrowLeft") go(-1);
    else if(e.key === "ArrowRight") go(1);
    else if(e.key === "Delete"){ if(mode === "live") deleteCurrent(); else purgeCurrent(); }
  });

  if (SAVED_FOLDER) loadFolder();
</script>
</body>
</html>
"""


# A small floating popup, injected into every page except the disagreement page
# itself, so a running disagreement batch stays visible while you use other cards.
GLOBAL_PROGRESS = """
<style>
  #gp { position:fixed; right:16px; bottom:16px; width:300px; z-index:9999;
        background:#161a22; border:1px solid #2b3140; border-radius:10px; color:#eef1f6;
        font-family:'Segoe UI',system-ui,Arial,sans-serif; box-shadow:0 8px 30px rgba(0,0,0,.5); overflow:hidden; }
  #gp.gp-hidden { display:none; }
  #gp .gp-head { display:flex; align-items:center; justify-content:space-between; padding:8px 12px; background:#1b2f22; }
  #gp .gp-title { font-size:12px; font-weight:600; letter-spacing:.3px; }
  #gp .gp-actions button { background:transparent; border:none; color:#cdd5e1; font-size:16px; line-height:1; cursor:pointer; padding:0 5px; }
  #gp .gp-actions button:hover { color:#fff; }
  #gp .gp-body { padding:10px 12px; }
  #gp.gp-min .gp-body { display:none; }
  #gp .gp-msg { font-size:12px; color:#cbd3df; margin-bottom:6px; }
  #gp .gp-bar { height:8px; background:#0c0e12; border:1px solid #2b3140; border-radius:5px; overflow:hidden; }
  #gp .gp-bar > div { height:100%; width:0%; background:linear-gradient(90deg,#3b62d6,#5f89ff); transition:width .3s; }
  #gp .gp-detail { font-size:11px; color:#9aa4b4; margin-top:6px; }
  #gp .gp-open { display:inline-block; margin-top:8px; font-size:12px; color:#8fb0ff; text-decoration:none; }
  #gp .gp-open:hover { text-decoration:underline; }
</style>
<div id="gp" class="gp gp-hidden">
  <div class="gp-head">
    <span class="gp-title">Disagreement run</span>
    <span class="gp-actions">
      <button title="Minimize / expand" onclick="gpToggleMin()">–</button>
      <button title="Dismiss" onclick="gpDismiss()">&times;</button>
    </span>
  </div>
  <div class="gp-body">
    <div class="gp-msg" id="gp_msg"></div>
    <div class="gp-bar"><div id="gp_fill"></div></div>
    <div class="gp-detail" id="gp_detail"></div>
    <a class="gp-open" href="/disagreement">Open full view &rarr;</a>
  </div>
</div>
<script>
  (function(){
    let dismissed = false, wasRunning = false;
    const el = (id) => document.getElementById(id);
    window.gpToggleMin = function(){ el("gp").classList.toggle("gp-min"); };
    window.gpDismiss = function(){ dismissed = true; el("gp").classList.add("gp-hidden"); };
    async function tick(){
      let s;
      try { const r = await fetch("/api/disagree/status"); s = await r.json(); } catch(e){ return; }
      const running = !!s.running;
      if(running && !wasRunning) dismissed = false;   // a fresh run un-dismisses the popup
      wasRunning = running;
      const finalPhase = (s.phase==="done" || s.phase==="stopped" || s.phase==="error");
      const gp = el("gp");
      if(!((running || finalPhase) && !dismissed)){ gp.classList.add("gp-hidden"); return; }
      gp.classList.remove("gp-hidden");
      const phaseText = {scanning:"Scanning", done:"Done", stopped:"Stopped", error:"Error", idle:"Idle"}[s.phase] || s.phase;
      el("gp_msg").textContent = phaseText + (s.message ? (" — " + s.message) : "");
      let pct = 0;
      if(s.phase==="scanning" && s.total_frames>0) pct = 100*s.scanned_frames/s.total_frames;
      else if(finalPhase) pct = 100;
      el("gp_fill").style.width = Math.min(100, pct) + "%";
      let d = "";
      if(s.videos_total>1) d += "Videos: " + s.videos_done + "/" + s.videos_total + " \\u00b7 ";
      d += "Kept: " + (s.kept||0);
      el("gp_detail").textContent = d;
    }
    setInterval(tick, 1500);
    tick();
  })();
</script>
"""


def _inject_progress(page_html):
    """Drop the floating disagreement-progress popup in just before </body>."""
    return page_html.replace("</body>", GLOBAL_PROGRESS + "</body>", 1)


@app.get("/")
def index():
    return Response(_inject_progress(LANDING_PAGE), mimetype="text/html")


@app.get("/balancer")
def balancer():
    settings = load_settings()
    all_pt = discover_pt_models([str(MODELS_DIR)])
    selected_model = settings.get("model_path") or default_person_model(all_pt)
    options = build_model_options_html(all_pt, selected_model)
    page = (
        PAGE.replace("__TITLE__", html.escape(APP_TITLE))
        .replace("__MODEL_OPTIONS__", options)
        .replace("__SETTINGS_JSON__", json.dumps(settings))
    )
    return Response(_inject_progress(page), mimetype="text/html")


@app.get("/disagreement")
def disagreement():
    settings = load_disagree_settings()
    all_pt = discover_pt_models([str(MODELS_DIR)])
    teacher_sel = settings.get("teacher_model_path") or default_person_model(all_pt)
    student_sel = settings.get("student_model_path") or default_person_model(all_pt)
    dests = list_disagree_dests()
    dest_sel = settings.get("dest") if settings.get("dest") in dests else ""
    page = (
        DISAGREE_PAGE.replace("__TITLE__", html.escape(DISAGREE_TITLE))
        .replace("__TEACHER_OPTIONS__", build_model_options_html(all_pt, teacher_sel))
        .replace("__STUDENT_OPTIONS__", build_model_options_html(all_pt, student_sel))
        .replace("__DEST_OPTIONS__", build_dest_options_html(dests, dest_sel))
        .replace("__SETTINGS_JSON__", json.dumps(settings))
    )
    return Response(page, mimetype="text/html")


@app.get("/review")
def review():
    saved = load_review_folder()
    dests = list_disagree_dests()
    page = (
        REVIEW_PAGE.replace("__TITLE__", html.escape(REVIEW_TITLE))
        .replace("__REVIEW_DEST_OPTIONS__", build_review_dest_options_html(dests, saved))
        .replace("__SAVED_FOLDER__", json.dumps(saved))
    )
    return Response(_inject_progress(page), mimetype="text/html")


@app.get("/api/preview")
def api_preview():
    path = request.args.get("path", "")
    rows = max(1, int(request.args.get("rows", 4)))
    cols = max(1, int(request.args.get("cols", 4)))
    path = str(Path(path).expanduser())
    if os.path.isdir(path):
        # Folder given: grab a frame from the first video as a representative thumbnail.
        vids = list_videos_in_folder(path)
        if not vids:
            return Response("no videos in folder", status=404)
        path = vids[0]
    elif not os.path.isfile(path):
        return Response("not found", status=404)
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return Response("cannot read frame", status=400)
    out = draw_grid(frame, rows, cols)
    ok, buf = cv2.imencode(".jpg", out)
    if not ok:
        return Response("encode failed", status=500)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.post("/api/run")
def api_run():
    payload = request.get_json(silent=True) or {}
    with job_lock:
        if job["running"]:
            return jsonify({"ok": False, "error": "A job is already running."}), 409
        try:
            input_path = str(payload.get("video_path", "")).strip()
            model_path = str(payload.get("model_path", "")).strip()
            total = max(1, int(payload.get("total", 1)))
            rows = max(1, min(20, int(payload.get("rows", 4))))
            cols = max(1, min(20, int(payload.get("cols", 4))))
            conf = max(0.0, min(1.0, float(payload.get("conf", 0.4))))
            stride = max(1, min(120, int(payload.get("stride", 15))))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid input values."}), 400
        if not input_path:
            return jsonify({"ok": False, "error": "A video file or folder path is required."}), 400

        # Remember these inputs for next time.
        save_settings({
            "video_path": input_path, "model_path": model_path, "total": total,
            "rows": rows, "cols": cols, "conf": conf, "stride": stride,
        })

        job.update({
            "running": True, "phase": "scanning", "message": "Starting...",
            "scanned_frames": 0, "total_frames": 0, "candidates": 0, "saved": 0,
            "requested": total, "started_at": time.time(),
            "videos_total": 0, "videos_done": 0, "current_video": "",
            "result": None, "error": "",
        })
    t = threading.Thread(
        target=run_batch, args=(input_path, model_path, total, rows, cols, conf, stride), daemon=True
    )
    t.start()
    return jsonify({"ok": True})


@app.get("/api/status")
def api_status():
    with job_lock:
        return jsonify(dict(job))


@app.post("/api/disagree/run")
def api_disagree_run():
    payload = request.get_json(silent=True) or {}
    with disagree_job_lock:
        if disagree_job["running"]:
            return jsonify({"ok": False, "error": "A job is already running."}), 409
        try:
            input_path = str(payload.get("video_path", "")).strip()
            teacher_path = str(payload.get("teacher_model_path", "")).strip()
            student_path = str(payload.get("student_model_path", "")).strip()
            rows = max(1, min(20, int(payload.get("rows", 4))))
            cols = max(1, min(20, int(payload.get("cols", 4))))
            teacher_conf = max(0.0, min(1.0, float(payload.get("teacher_conf", 0.4))))
            student_conf = max(0.0, min(1.0, float(payload.get("student_conf", 0.4))))
            stride = max(1, min(120, int(payload.get("stride", 15))))
            per_cell = max(0, int(payload.get("per_cell", 0)))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid input values."}), 400
        if not input_path:
            return jsonify({"ok": False, "error": "A video file or folder path is required."}), 400
        if not teacher_path or not student_path:
            return jsonify({"ok": False, "error": "Both teacher and student models are required."}), 400

        # Resolve the output destination: a new folder name (if provided) or a picked one.
        dest_choice = str(payload.get("dest", "")).strip()
        new_dest = str(payload.get("new_dest", "")).strip()
        dest = new_dest if dest_choice == NEW_DEST_VALUE else dest_choice
        dest = _safe_basename(dest)
        if not dest or dest.startswith("."):
            return jsonify({"ok": False, "error": "A valid output folder name is required."}), 400
        resume = bool(payload.get("resume", False))

        save_disagree_settings({
            "video_path": input_path, "teacher_model_path": teacher_path,
            "student_model_path": student_path, "rows": rows, "cols": cols,
            "teacher_conf": teacher_conf, "student_conf": student_conf,
            "stride": stride, "per_cell": per_cell, "dest": dest, "resume": resume,
        })

        disagree_job.update({
            "running": True, "phase": "scanning", "message": "Starting...",
            "scanned_frames": 0, "total_frames": 0, "kept": 0,
            "started_at": time.time(), "videos_total": 0, "videos_done": 0,
            "current_video": "", "result": None, "error": "",
            "stop_requested": False, "skip_requested": False,
        })
    t = threading.Thread(
        target=run_batch_disagree,
        args=(input_path, teacher_path, student_path, rows, cols,
              teacher_conf, student_conf, stride, per_cell, dest, resume),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True})


@app.post("/api/disagree/stop")
def api_disagree_stop():
    with disagree_job_lock:
        if not disagree_job["running"]:
            return jsonify({"ok": False, "error": "No job is running."}), 409
        disagree_job["stop_requested"] = True
        disagree_job["message"] = "Stopping after the current frame..."
    return jsonify({"ok": True})


@app.post("/api/disagree/skip")
def api_disagree_skip():
    with disagree_job_lock:
        if not disagree_job["running"]:
            return jsonify({"ok": False, "error": "No job is running."}), 409
        disagree_job["skip_requested"] = True
        disagree_job["message"] = "Skipping the current video..."
    return jsonify({"ok": True})


@app.get("/api/disagree/status")
def api_disagree_status():
    with disagree_job_lock:
        return jsonify(dict(disagree_job))


@app.get("/api/disagree/resume_info")
def api_disagree_resume_info():
    """How many videos in the chosen input already have a manifest in the destination."""
    path = request.args.get("path", "").strip()
    dest_choice = request.args.get("dest", "").strip()
    new_dest = request.args.get("new_dest", "").strip()
    dest = new_dest if dest_choice == NEW_DEST_VALUE else dest_choice
    dest = _safe_basename(dest) if dest else None
    if not path:
        return jsonify({"ok": False, "error": "no path"})
    p = Path(path).expanduser()
    if p.is_dir():
        videos = list_videos_in_folder(str(p))
    elif p.is_file():
        videos = [str(p)]
    else:
        return jsonify({"ok": False, "error": "Path not found."})
    total = len(videos)
    done = 0
    if dest:
        out_root = DISAGREE_ROOT / dest
        for v in videos:
            if (out_root / f"{Path(v).stem}_manifest.json").is_file():
                done += 1
    return jsonify({"ok": True, "total": total, "done": done,
                    "remaining": total - done, "dest": dest or ""})


@app.get("/api/review/list")
def api_review_list():
    """List frame/overlay pairs inside a folder that has frames/ and overlays/ subdirs."""
    folder = request.args.get("path", "").strip()
    if not folder:
        return jsonify({"ok": False, "error": "A folder path is required."}), 400
    root = Path(folder).expanduser()
    frames_dir = root / "frames"
    overlays_dir = root / "overlays"
    if not frames_dir.is_dir():
        return jsonify({"ok": False, "error": "No 'frames' subfolder found in that path."}), 404
    save_review_folder(folder)
    counts = _manifest_counts(root)
    items = _list_review_pairs(frames_dir, overlays_dir, counts)
    return jsonify({
        "ok": True,
        "root": str(root),
        "count": len(items),
        "items": items,
        "has_overlays_dir": overlays_dir.is_dir(),
        "has_counts": bool(counts),
    })


@app.get("/api/review/img")
def api_review_img():
    """Serve one frame or overlay image from a review folder."""
    target = _review_child(request.args.get("root", ""),
                            request.args.get("kind", ""),
                            request.args.get("name", ""))
    if not target or not target.is_file():
        return Response("not found", status=404)
    mime = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
    return send_file(str(target), mimetype=mime)


def _manifest_counts(folder):
    """
    Map image filename -> {teacher, student} person counts, read from every
    *_manifest.json at the folder root. Lets the review UI filter by box counts.
    """
    counts = {}
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        return counts
    for mf in folder.glob("*_manifest.json"):
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for fr in data.get("frames", []):
            fn = fr.get("file")
            if fn:
                counts[fn] = {"teacher": fr.get("teacher_persons"),
                              "student": fr.get("student_persons")}
    return counts


def _list_review_pairs(frames_dir, overlays_dir, counts=None):
    """Union of image basenames present in either dir -> pair dicts (+ box counts)."""
    counts = counts or {}
    names = set()
    for d in (frames_dir, overlays_dir):
        if d.is_dir():
            for name in os.listdir(d):
                if name.lower().endswith(IMAGE_EXTENSIONS) and (d / name).is_file():
                    names.add(name)
    out = []
    for n in sorted(names):
        c = counts.get(n) or {}
        out.append({
            "name": n,
            "has_frame": (frames_dir / n).is_file(),
            "has_overlay": (overlays_dir / n).is_file(),
            "teacher_persons": c.get("teacher"),
            "student_persons": c.get("student"),
        })
    return out


def _move_review_pair(folder, name, to_trash):
    """Move a frame+overlay pair between the live dirs and .trash (and back)."""
    moved = []
    for kind in ("frames", "overlays"):
        # Source is the opposite of the destination: deleting moves live -> trash.
        src = _review_child(folder, kind, name, trash=not to_trash)
        dst = _review_child(folder, kind, name, trash=to_trash)
        if src and dst and src.is_file():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(src), str(dst))
                moved.append(kind)
            except OSError:
                pass
    return moved


@app.post("/api/review/delete")
def api_review_delete():
    """Move a frame and its matching overlay into the folder's .trash bin."""
    payload = request.get_json(silent=True) or {}
    folder = str(payload.get("root", "")).strip()
    name = _safe_basename(payload.get("name", ""))
    if not folder or not name:
        return jsonify({"ok": False, "error": "Invalid folder or filename."}), 400
    moved = _move_review_pair(folder, name, to_trash=True)
    return jsonify({"ok": True, "trashed": moved})


@app.get("/api/review/trash/list")
def api_review_trash_list():
    """List pairs currently in the folder's .trash bin."""
    folder = request.args.get("path", "").strip()
    if not folder:
        return jsonify({"ok": False, "error": "A folder path is required."}), 400
    root = Path(folder).expanduser()
    counts = _manifest_counts(root)
    items = _list_review_pairs(root / TRASH_DIRNAME / "frames", root / TRASH_DIRNAME / "overlays", counts)
    return jsonify({"ok": True, "root": str(root), "count": len(items), "items": items})


@app.get("/api/review/trash/img")
def api_review_trash_img():
    """Serve a frame or overlay image from the .trash bin."""
    target = _review_child(request.args.get("root", ""),
                           request.args.get("kind", ""),
                           request.args.get("name", ""),
                           trash=True)
    if not target or not target.is_file():
        return Response("not found", status=404)
    mime = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
    return send_file(str(target), mimetype=mime)


@app.post("/api/review/trash/restore")
def api_review_trash_restore():
    """Move a trashed pair back into the live frames/ and overlays/ folders."""
    payload = request.get_json(silent=True) or {}
    folder = str(payload.get("root", "")).strip()
    name = _safe_basename(payload.get("name", ""))
    if not folder or not name:
        return jsonify({"ok": False, "error": "Invalid folder or filename."}), 400
    moved = _move_review_pair(folder, name, to_trash=False)
    return jsonify({"ok": True, "restored": moved})


@app.post("/api/review/trash/purge")
def api_review_trash_purge():
    """Permanently delete a trashed pair from the .trash bin."""
    payload = request.get_json(silent=True) or {}
    folder = str(payload.get("root", "")).strip()
    name = _safe_basename(payload.get("name", ""))
    if not folder or not name:
        return jsonify({"ok": False, "error": "Invalid folder or filename."}), 400
    purged = []
    for kind in ("frames", "overlays"):
        target = _review_child(folder, kind, name, trash=True)
        if target and target.is_file():
            try:
                target.unlink()
                purged.append(kind)
            except OSError:
                pass
    return jsonify({"ok": True, "purged": purged})


@app.get("/api/file")
def api_file():
    """Serve a crop or heatmap by absolute path, restricted to our output roots."""
    p = _safe_serve(request.args.get("path", ""))
    if not p:
        return Response("not found", status=404)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return send_file(str(p), mimetype=mime)


def _open_browser_when_ready(url, delay_sec=0.8):
    def _open():
        time.sleep(delay_sec)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    CROPS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    DISAGREE_ROOT.mkdir(parents=True, exist_ok=True)
    _open_browser_when_ready(f"http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
