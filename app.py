import os
import logging
import warnings
import base64
import io
import json
import queue
import shutil
import threading
import time
import zipfile

# Suppress flask-limiter in-memory storage warning (expected for local use)
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter")

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from ultralytics import YOLO
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── GPU check ────────────────────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        log.info("GPU READY: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
        torch.cuda.set_device(0)
    else:
        log.warning("NO GPU DETECTED — running on CPU. Training will be slow.")
        log.warning("Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    log.warning("PyTorch not installed — GPU unavailable.")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["FLASK_SKIP_DOTENV"] = True

CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=['600 per minute'],
)

MODELS_DIR  = os.path.realpath(os.path.join(os.path.dirname(__file__), 'models'))
RUNS_DIR    = os.path.realpath(os.path.join(os.path.dirname(__file__), 'runs'))
DATASET_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), 'datasets'))

_models: dict = {}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB

# ── Training state ────────────────────────────────────────────────────────────
_train_state = {
    "running":   False,
    "cancelled": False,
    "progress":  [],          # list of epoch dicts
    "error":     None,
    "result":    None,        # path to best.pt when done
}
_train_queue: queue.Queue = queue.Queue()
_train_lock = threading.Lock()


def _resolve_model_path(raw_path: str) -> str | None:
    if not raw_path:
        return None
    real = os.path.realpath(os.path.abspath(raw_path))
    if not real.lower().startswith(MODELS_DIR.lower() + os.sep.lower()):
        log.warning("Path blocked. Path: %s | MODELS_DIR: %s", real, MODELS_DIR)
        return None
    return real


# ── Health ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Reef AI Python backend running"})


# ── Model class names ─────────────────────────────────────────────────────────
@app.route('/model-classes', methods=['POST'])
def model_classes():
    data = request.json or {}
    raw_path = data.get('model_path', '')
    real_path = _resolve_model_path(raw_path)
    if not real_path:
        return jsonify({"error": f"Model path not in models folder. Got: {raw_path}"}), 400
    if not os.path.exists(real_path):
        return jsonify({"error": "Model not found"}), 404
    try:
        if real_path not in _models:
            _models[real_path] = YOLO(real_path)
        names = _models[real_path].names
        return jsonify({"classes": [names[i] for i in sorted(names)]})
    except Exception as exc:
        return jsonify({"error": f"Failed to load model: {exc}"}), 500


# ── Folder scanning ──────────────────────────────────────────────────────────
@app.route('/scan-folder', methods=['POST'])
def scan_folder():
    """List all images in a folder (non-recursive option available)."""
    data = request.json or {}
    folder = data.get('folder', '').strip()
    recursive = data.get('recursive', False)

    if not folder or not os.path.isdir(folder):
        return jsonify({"error": f"Folder not found: {folder}"}), 400

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []

    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    full = os.path.join(root, f)
                    images.append({"name": os.path.relpath(full, folder).replace("\\", "/"), "path": full})
    else:
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                full = os.path.join(folder, f)
                images.append({"name": f, "path": full})

    log.info("Scanned %s: found %d images", folder, len(images))
    return jsonify({"images": images, "total": len(images)})


@app.route('/image', methods=['GET'])
def serve_image():
    """Serve an image file by absolute path for local display."""
    path = request.args.get('path', '')
    if not path or not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404
    # Security: only serve common image types
    ext = os.path.splitext(path)[1].lower()
    mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                '.webp': 'image/webp', '.bmp': 'image/bmp'}
    if ext not in mime_map:
        return jsonify({"error": "Not an image"}), 400
    from flask import send_file
    return send_file(path, mimetype=mime_map[ext])


# ── Detection ─────────────────────────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    raw_path = data.get('model_path', '')
    log.info("Detection request. Model: %s", raw_path)

    real_path = _resolve_model_path(raw_path)
    if not real_path:
        return jsonify({"error": f"Model path not in models folder. Got: {raw_path}"}), 400
    if not os.path.exists(real_path):
        return jsonify({"error": "Model not found. Re-upload it in the MODELS tab."}), 404

    if real_path not in _models:
        try:
            log.info("Loading model: %s", real_path)
            _models[real_path] = YOLO(real_path)
        except Exception as exc:
            return jsonify({"error": f"Failed to load model: {exc}"}), 500

    model = _models[real_path]

    try:
        raw_b64 = data['image']
        b64_data = raw_b64.split(',')[1] if ',' in raw_b64 else raw_b64
        img_bytes = base64.b64decode(b64_data)
        if len(img_bytes) > MAX_IMAGE_BYTES:
            return jsonify({"error": f"Image too large (max {MAX_IMAGE_BYTES // 1024 // 1024} MB)"}), 413
        img = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img)
    except Exception as exc:
        return jsonify({"error": f"Invalid image data: {exc}"}), 400

    try:
        confidence = float(data.get('confidence', 0.25))
        results = model(img_np, conf=confidence)
    except Exception as exc:
        return jsonify({"error": f"Inference error: {exc}"}), 500

    detections = []
    h, w = img_np.shape[:2]
    for r in results:
        for box in r.boxes:
            b    = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            detections.append({
                "label":      model.names[cls],
                "confidence": conf,
                "bbox": [
                    (b[0] / w) * 1000, (b[1] / h) * 1000,
                    (b[2] / w) * 1000, (b[3] / h) * 1000,
                ],
            })

    log.info("Detection complete. Found %d objects.", len(detections))
    return jsonify(detections)


# ── Training ──────────────────────────────────────────────────────────────────

def _run_training(dataset_path: str, base_model: str, epochs: int,
                  img_size: int, batch: int, output_name: str, augmentation: str = "heavy", dest_name: str = ""):
    """Runs in a background thread. Pushes progress dicts to _train_queue."""
    global _train_state

    try:
        log.info("Training started. dataset=%s base=%s epochs=%d", dataset_path, base_model, epochs)

        # Custom callback to capture per-epoch metrics
        epoch_results = []

        def on_train_epoch_end(trainer):
            if _train_state["cancelled"]:
                trainer.stop = True
                return
            metrics = trainer.metrics or {}
            ep = {
                "epoch":      int(trainer.epoch) + 1,
                "total":      epochs,
                "box_loss":   round(float(trainer.loss_items[0]), 4) if trainer.loss_items is not None else None,
                "cls_loss":   round(float(trainer.loss_items[1]), 4) if trainer.loss_items is not None and len(trainer.loss_items) > 1 else None,
                "mAP50":      round(float(metrics.get("metrics/mAP50(B)", 0)), 4),
                "mAP50_95":   round(float(metrics.get("metrics/mAP50-95(B)", 0)), 4),
                "precision":  round(float(metrics.get("metrics/precision(B)", 0)), 4),
                "recall":     round(float(metrics.get("metrics/recall(B)", 0)), 4),
            }
            epoch_results.append(ep)
            _train_queue.put({"type": "epoch", **ep})
            log.info("Epoch %d/%d — mAP50=%.4f box_loss=%.4f",
                     ep["epoch"], epochs, ep["mAP50"], ep["box_loss"] or 0)

        model = YOLO(base_model)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        project_dir = os.path.join(RUNS_DIR, "train")
        os.makedirs(project_dir, exist_ok=True)

        # Augmentation settings
        aug_params = {}
        if augmentation == "off":
            aug_params = dict(
                fliplr=0.0, flipud=0.0, degrees=0.0,
                translate=0.0, scale=0.0, mosaic=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            )
        elif augmentation == "standard":
            aug_params = dict(
                fliplr=0.5, flipud=0.1, degrees=5.0,
                translate=0.1, scale=0.3, mosaic=0.5,
                hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
            )
        else:  # heavy — best for small datasets
            aug_params = dict(
                fliplr=0.5, flipud=0.3, degrees=15.0,
                translate=0.2, scale=0.5, mosaic=1.0,
                hsv_h=0.03, hsv_s=0.7, hsv_v=0.5,
                mixup=0.1, copy_paste=0.1,
            )

        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            project=project_dir,
            name=output_name,
            exist_ok=True,
            verbose=False,
            **aug_params,
        )

        if _train_state["cancelled"]:
            _train_queue.put({"type": "cancelled"})
            return

        # Copy best.pt into models/ folder
        best_src = os.path.join(project_dir, output_name, "weights", "best.pt")
        if os.path.exists(best_src):
            if not dest_name:
                dest_name = output_name + ".pt"
            dest_path = os.path.join(MODELS_DIR, dest_name)
            shutil.copy2(best_src, dest_path)
            log.info("Saved best model to %s", dest_path)
            _train_state["result"] = dest_path
            _train_queue.put({"type": "done", "model_path": dest_path, "model_name": dest_name})
        else:
            _train_queue.put({"type": "error", "message": "Training completed but best.pt not found"})

    except Exception as exc:
        log.error("Training failed: %s", exc)
        _train_state["error"] = str(exc)
        _train_queue.put({"type": "error", "message": str(exc)})
    finally:
        _train_state["running"] = False


@app.route('/train/start', methods=['POST'])
def train_start():
    global _train_state

    with _train_lock:
        if _train_state["running"]:
            return jsonify({"error": "Training already running"}), 409

    data = request.json or {}

    # Accept either a dataset zip (uploaded separately) or a directory path
    dataset_path = data.get('dataset_path', '')
    base_model   = data.get('base_model',   'yolov8n.pt')
    epochs       = int(data.get('epochs',    50))
    img_size     = int(data.get('img_size',  640))
    batch        = int(data.get('batch',     -1))   # -1 = auto
    augmentation = data.get('augmentation', 'heavy')
    model_name   = data.get('model_name', '').strip()
    # Use custom name if provided, otherwise timestamp
    safe_name    = ''.join(c for c in model_name if c.isalnum() or c in '_-')[:40]
    ts           = int(time.time())
    output_name  = f'{safe_name}_{ts}' if safe_name else f'reef_train_{ts}'
    # No _best suffix — use the name the user chose directly
    dest_name    = f'{safe_name}.pt' if safe_name else f'reef_train_{ts}.pt' 

    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"error": f"Dataset not found at: {dataset_path}"}), 400

    yaml_path = dataset_path if dataset_path.endswith('.yaml') else os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        return jsonify({"error": f"data.yaml not found in dataset folder: {dataset_path}"}), 400

    # Reset state
    with _train_lock:
        _train_state = {
            "running":   True,
            "cancelled": False,
            "progress":  [],
            "error":     None,
            "result":    None,
        }
    # Drain queue
    while not _train_queue.empty():
        try: _train_queue.get_nowait()
        except: pass

    thread = threading.Thread(
        target=_run_training,
        args=(yaml_path, base_model, epochs, img_size, batch, output_name, augmentation, dest_name),
        daemon=True,
    )
    thread.start()

    log.info("Training thread started.")
    return jsonify({"status": "started"})


@app.route('/train/progress', methods=['GET'])
def train_progress():
    """Server-Sent Events stream of training progress."""
    def generate():
        yield "data: {\"type\": \"connected\"}\n\n"
        while True:
            try:
                msg = _train_queue.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error", "cancelled"):
                    break
            except queue.Empty:
                # Heartbeat to keep connection alive
                yield "data: {\"type\": \"heartbeat\"}\n\n"
                if not _train_state["running"]:
                    break

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/train/status', methods=['GET'])
def train_status():
    return jsonify({
        "running":  _train_state["running"],
        "error":    _train_state["error"],
        "result":   _train_state["result"],
        "progress": _train_state["progress"][-1] if _train_state["progress"] else None,
    })


@app.route('/train/cancel', methods=['POST'])
def train_cancel():
    _train_state["cancelled"] = True
    log.info("Training cancel requested.")
    return jsonify({"status": "cancelling"})


@app.route('/train/upload-dataset', methods=['POST'])
def upload_dataset():
    """Accept a zip of the Roboflow export and unpack it into datasets/."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    if not f.filename.endswith('.zip'):
        return jsonify({"error": "Must be a .zip file"}), 400

    os.makedirs(DATASET_DIR, exist_ok=True)
    name   = f"dataset_{int(time.time())}"
    dest   = os.path.join(DATASET_DIR, name)
    zip_path = dest + ".zip"
    f.save(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    os.remove(zip_path)

    return _process_extracted_dataset(dest, name)


@app.route('/train/upload-dataset-path', methods=['POST'])
def upload_dataset_path():
    """Accept a path to a zip already on disk — no upload needed."""
    data = request.json or {}
    zip_path = data.get('zip_path', '')
    filename = data.get('filename', 'dataset.zip')

    if not zip_path or not os.path.exists(zip_path):
        return jsonify({"error": f"Zip not found: {zip_path}"}), 400

    os.makedirs(DATASET_DIR, exist_ok=True)
    name = f"dataset_{int(time.time())}"
    dest = os.path.join(DATASET_DIR, name)

    log.info("Extracting dataset from %s → %s", zip_path, dest)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)

    return _process_extracted_dataset(dest, name)


def _process_extracted_dataset(dest: str, name: str):
    """Find data.yaml, fix paths, return result. Used by both upload endpoints."""

    # Find data.yaml — could be nested
    yaml_path = None
    for root, dirs, files in os.walk(dest):
        for fn in files:
            if fn == 'data.yaml':
                yaml_path = os.path.join(root, fn)
                break
        if yaml_path:
            break

    if not yaml_path:
        shutil.rmtree(dest, ignore_errors=True)
        return jsonify({"error": "No data.yaml found in zip"}), 400

    # Rewrite data.yaml with correct absolute paths to extracted location
    with open(yaml_path) as fp:
        content = fp.read()

    dataset_root = os.path.dirname(yaml_path).replace('\\', '/')

    import re, yaml as _yaml

    # Try to parse and rewrite properly
    try:
        data = _yaml.safe_load(content)
        # Find the images folders — look for train/images, valid/images etc inside extracted dir
        def find_images_dir(subdir):
            """Try common image folder locations inside the dataset."""
            candidates = [
                os.path.join(dataset_root, subdir, 'images'),
                os.path.join(dataset_root, subdir),
                os.path.join(dataset_root, 'images'),
                dataset_root,
            ]
            for c in candidates:
                if os.path.isdir(c) and any(
                    f.lower().endswith(('.jpg','.jpeg','.png','.webp'))
                    for f in os.listdir(c)[:5] if os.path.isfile(os.path.join(c, f))
                ):
                    return c.replace('\\', '/')
            return os.path.join(dataset_root, 'images').replace('\\', '/')

        # If path key exists, use it as base for relative train/val paths
        base = data.get('path', dataset_root)
        if not os.path.isabs(base):
            base = os.path.join(dataset_root, base)
        base = base.replace('\\', '/')

        def resolve_split(val):
            if not val:
                return find_images_dir('train')
            if os.path.isabs(val):
                # Absolute path — find actual images dir relative to dataset_root instead
                return find_images_dir('train')
            # Relative path — resolve against base
            candidate = os.path.join(base, val).replace('\\', '/')
            if os.path.isdir(candidate):
                return candidate
            return find_images_dir('train')

        data['train'] = resolve_split(data.get('train', ''))
        data['val']   = resolve_split(data.get('val', data.get('valid', '')))
        if 'path' in data:
            del data['path']  # remove path key — use absolute paths instead
        if 'test' in data:
            data['test'] = resolve_split(data.get('test', ''))

        with open(yaml_path, 'w') as fp:
            _yaml.dump(data, fp, default_flow_style=False)
        log.info("data.yaml paths fixed: train=%s val=%s", data['train'], data['val'])
    except Exception as yaml_err:
        log.warning("Could not parse yaml, using regex fallback: %s", yaml_err)
        def fix_path(match):
            key = match.group(1)
            folder = 'train' if key == 'train' else ('valid' if key == 'val' else key)
            images_path = os.path.join(dataset_root, folder, 'images')
            if not os.path.isdir(images_path):
                images_path = os.path.join(dataset_root, 'images')
            return f"{key}: {images_path.replace(chr(92), '/')}"
        content = re.sub(r'(train|val|test):\s*(.+)', fix_path, content)
        with open(yaml_path, 'w') as fp:
            fp.write(content)

    log.info("Dataset extracted to %s", dest)
    return jsonify({"status": "ok", "dataset_path": yaml_path, "dataset_name": name})


@app.route('/train/datasets', methods=['GET'])
def list_datasets():
    """List available datasets in the datasets/ folder."""
    if not os.path.exists(DATASET_DIR):
        return jsonify({"datasets": []})
    datasets = []
    for name in os.listdir(DATASET_DIR):
        full = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(full):
            continue
        yaml_path = None
        for root, dirs, files in os.walk(full):
            for fn in files:
                if fn == 'data.yaml':
                    yaml_path = os.path.join(root, fn)
                    break
            if yaml_path:
                break
        if yaml_path:
            # Count images
            img_count = 0
            for root, dirs, files in os.walk(full):
                img_count += sum(1 for f in files if f.lower().endswith(('.jpg','.jpeg','.png','.webp')))
            datasets.append({"name": name, "yaml_path": yaml_path, "image_count": img_count})
    return jsonify({"datasets": datasets})


# ── YouTube frame extraction ─────────────────────────────────────────────────
@app.route('/youtube-frames', methods=['POST'])
def youtube_frames():
    """Download a YouTube video and extract frames as a zip. Streams SSE progress."""
    data = request.json or {}
    url        = data.get('url', '').strip()
    interval   = max(1, int(data.get('interval', 5)))   # seconds between frames
    max_frames = max(1, int(data.get('max_frames', 200)))
    label      = data.get('label', 'youtube').strip().replace(' ', '_')[:40]

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    def sse(obj):
        return "data: " + json.dumps(obj) + "\n\n"

    def generate():
        try:
            try:
                import yt_dlp
            except ImportError:
                yield sse({"type": "error", "message": "yt-dlp not installed. Run: pip install yt-dlp"})
                return

            import tempfile, zipfile as zf, pathlib
            import cv2 as _cv2

            tmpdir = tempfile.mkdtemp(prefix='reef_yt_')
            video_path = os.path.join(tmpdir, 'video.mp4')

            yield sse({"type": "progress", "stage": "downloading", "message": "Downloading video from YouTube..."})

            ydl_opts = {
                'format': 'best[height<=720]/best',
                'outtmpl': video_path,
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'video')[:50]
                duration = info.get('duration', 0)

            yield sse({"type": "progress", "stage": "extracting", "message": "Extracting frames..."})

            cap = _cv2.VideoCapture(video_path)
            fps = cap.get(_cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, int(fps * interval))

            frames_saved = 0
            frame_idx = 0
            frame_paths = []

            while frames_saved < max_frames:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                out_path = os.path.join(tmpdir, f"frame_{frames_saved:04d}.jpg")
                _cv2.imwrite(out_path, frame, [_cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_paths.append(out_path)
                frames_saved += 1
                frame_idx += frame_interval

                if frames_saved % 10 == 0:
                    yield sse({"type": "progress", "stage": "extracting",
                               "message": "Extracted " + str(frames_saved) + " frames...",
                               "frames": frames_saved})

            cap.release()

            yield sse({"type": "progress", "stage": "zipping", "message": "Saving zip to Downloads..."})

            downloads = pathlib.Path.home() / "Downloads"
            downloads.mkdir(exist_ok=True)
            safe_label = label.replace(" ", "_")[:40]
            zip_name = "reef_yt_" + safe_label + "_" + str(int(time.time())) + ".zip"
            zip_path = downloads / zip_name

            with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as z:
                for fp in frame_paths:
                    z.write(fp, "images/" + os.path.basename(fp))

            shutil.rmtree(tmpdir, ignore_errors=True)
            log.info("YouTube extraction complete: %d frames -> %s", frames_saved, zip_path)
            yield sse({"type": "done", "frames": frames_saved, "zip_name": zip_name, "title": title, "duration": duration})

        except Exception as exc:
            log.error("YouTube extraction error: %s", exc)
            yield sse({"type": "error", "message": str(exc)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Enhanced image scraping (multi-source icrawler) ──────────────────────────
@app.route('/screenshot-scrape', methods=['POST'])
def screenshot_scrape():
    """Enhanced scraping using multiple icrawler sources for higher yield."""
    data = request.json or {}
    query      = data.get('query', '').strip()
    max_images = max(1, int(data.get('max_images', 50)))
    label      = data.get('label', query).strip().replace(' ', '_')[:40]

    if not query:
        return jsonify({"error": "No query provided"}), 400

    def sse(obj):
        return "data: " + json.dumps(obj) + "\n\n"

    def generate():
        try:
            try:
                from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, FlickrImageCrawler
            except ImportError:
                yield sse({"type": "error", "message": "icrawler not installed. Run: pip install icrawler"})
                return

            import tempfile, zipfile as zf, pathlib

            tmpdir = tempfile.mkdtemp(prefix='reef_enhanced_')
            per_source = max(10, max_images // 3)

            yield sse({"type": "progress", "found": 0, "saved": 0, "message": "Searching Bing Images..."})

            # Source 1: Bing
            try:
                bing_dir = os.path.join(tmpdir, "bing")
                os.makedirs(bing_dir)
                crawler = BingImageCrawler(storage={"root_dir": bing_dir}, log_level=logging.ERROR, downloader_threads=4)
                crawler.crawl(keyword=query, max_num=per_source, min_size=(100, 100))
            except Exception as e:
                log.warning("Bing crawl failed: %s", e)

            yield sse({"type": "progress", "found": 0, "saved": 0, "message": "Searching Google Images..."})

            # Source 2: Google
            try:
                google_dir = os.path.join(tmpdir, "google")
                os.makedirs(google_dir)
                crawler = GoogleImageCrawler(storage={"root_dir": google_dir}, log_level=logging.ERROR, downloader_threads=4)
                crawler.crawl(keyword=query, max_num=per_source, min_size=(100, 100))
            except Exception as e:
                log.warning("Google crawl failed: %s", e)

            yield sse({"type": "progress", "found": 0, "saved": 0, "message": "Searching Flickr..."})

            # Source 3: Flickr (great for nature/wildlife photos)
            try:
                flickr_dir = os.path.join(tmpdir, "flickr")
                os.makedirs(flickr_dir)
                crawler = FlickrImageCrawler("", storage={"root_dir": flickr_dir}, log_level=logging.ERROR)
                crawler.crawl(keyword=query, max_num=per_source, min_size=(100, 100))
            except Exception as e:
                log.warning("Flickr crawl failed: %s", e)

            # Collect all images
            imgs = [f for f in pathlib.Path(tmpdir).rglob("*")
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
            saved = len(imgs)

            yield sse({"type": "progress", "found": saved, "saved": saved, "message": "Zipping " + str(saved) + " images..."})

            if saved == 0:
                yield sse({"type": "error", "message": "No images found. Try Direct Download mode."})
                shutil.rmtree(tmpdir, ignore_errors=True)
                return

            downloads = pathlib.Path.home() / "Downloads"
            downloads.mkdir(exist_ok=True)
            zip_name = "reef_enhanced_" + label + "_" + str(int(time.time())) + ".zip"
            zip_path = downloads / zip_name

            with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as z:
                for i, img in enumerate(imgs[:max_images]):
                    z.write(img, "images/img_{:04d}{}".format(i, img.suffix))

            shutil.rmtree(tmpdir, ignore_errors=True)
            yield sse({"type": "done", "saved": min(saved, max_images), "zip_name": zip_name})

        except Exception as exc:
            log.error("Enhanced scrape error: %s", exc)
            yield sse({"type": "error", "message": str(exc)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Image scraping ────────────────────────────────────────────────────────────# ── Image scraping ────────────────────────────────────────────────────────────
@app.route('/scrape-images', methods=['POST'])
def scrape_images():
    """Search the web and download images for training. Streams progress via SSE."""
    data = request.json or {}
    query   = data.get('query', '').strip()
    count   = max(1, int(data.get('count', 50)))
    label   = data.get('label', query).strip().replace(' ', '_')

    if not query:
        return jsonify({"error": "No search query provided"}), 400

    def sse(obj):
        return "data: " + json.dumps(obj) + "\n\n"

    def generate():
        try:
            try:
                from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
            except ImportError:
                yield sse({"type": "error", "message": "icrawler not installed. Run: pip install icrawler"})
                return

            import tempfile, zipfile as zf, pathlib

            tmpdir = tempfile.mkdtemp(prefix='reef_scrape_')

            yield sse({"type": "progress", "found": 0, "downloaded": 0, "failed": 0, "status": "Searching for: " + query})

            for CrawlerClass in [BingImageCrawler, GoogleImageCrawler]:
                try:
                    crawler = CrawlerClass(
                        storage={"root_dir": tmpdir},
                        log_level=logging.ERROR,
                        downloader_threads=4,
                    )
                    crawler.crawl(keyword=query, max_num=count, min_size=(100, 100))
                    break
                except Exception as e:
                    log.warning("Crawler %s failed: %s", CrawlerClass.__name__, e)
                    continue

            # icrawler saves into subfolders — search recursively for actual images
            imgs = [f for f in pathlib.Path(tmpdir).rglob('*')
                    if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
            downloaded = len(imgs)
            failed = max(0, count - downloaded)

            yield sse({"type": "progress", "found": count, "downloaded": downloaded, "failed": failed, "status": "Zipping " + str(downloaded) + " images..."})

            if downloaded == 0:
                yield sse({"type": "error", "message": "No images downloaded. Try a different search term."})
                shutil.rmtree(tmpdir, ignore_errors=True)
                return

            downloads = pathlib.Path.home() / "Downloads"
            downloads.mkdir(exist_ok=True)
            safe_label = label.replace(" ", "_").replace("/", "_")[:50]
            zip_name = "reef_scrape_" + safe_label + "_" + str(int(time.time())) + ".zip"
            zip_path = downloads / zip_name

            with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as z:
                for img in imgs:
                    z.write(img, "images/" + img.name)

            shutil.rmtree(tmpdir, ignore_errors=True)

            log.info("Scrape complete: %d images -> %s", downloaded, zip_path)
            yield sse({"type": "done", "found": count, "downloaded": downloaded, "failed": failed, "zip_name": zip_name})

        except Exception as exc:
            log.error("Scrape error: %s", exc)
            yield sse({"type": "error", "message": str(exc)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Model management ─────────────────────────────────────────────────────────
@app.route('/model/delete', methods=['POST'])
def delete_model():
    data = request.json or {}
    filename = data.get('filename', '').strip()
    if not filename:
        return jsonify({"error": "No filename"}), 400
    path = os.path.join(MODELS_DIR, filename)
    real = os.path.realpath(path)
    if not real.lower().startswith(MODELS_DIR.lower()):
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.exists(real):
        return jsonify({"error": "File not found"}), 404
    os.remove(real)
    log.info("Deleted model: %s", filename)
    return jsonify({"status": "deleted"})


@app.route('/model/rename', methods=['POST'])
def rename_model():
    data = request.json or {}
    old_name = data.get('old_name', '').strip()
    new_name = data.get('new_name', '').strip()
    if not old_name or not new_name:
        return jsonify({"error": "Missing names"}), 400
    # Sanitise new name
    safe = ''.join(c for c in new_name if c.isalnum() or c in '_-')
    if not safe.endswith('.pt'):
        safe = safe + '.pt'
    old_path = os.path.realpath(os.path.join(MODELS_DIR, old_name))
    new_path = os.path.realpath(os.path.join(MODELS_DIR, safe))
    if not old_path.lower().startswith(MODELS_DIR.lower()):
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.exists(old_path):
        return jsonify({"error": "File not found"}), 404
    if os.path.exists(new_path):
        return jsonify({"error": "A model with that name already exists"}), 409
    os.rename(old_path, new_path)
    log.info("Renamed model: %s -> %s", old_name, safe)
    return jsonify({"status": "renamed", "new_name": safe})


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    log.info("Starting Reef AI GPU backend on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
