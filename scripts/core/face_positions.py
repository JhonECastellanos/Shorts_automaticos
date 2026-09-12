"""Detección de posiciones faciales — paso 'face_positions' del pipeline v3.5.

Diseñado para videos de **plano general fijo** donde N personas permanecen
aproximadamente en la misma zona durante todo el video.

Enfoque ultra-simple:
1. Muestrear 60 frames distribuidos uniformemente en el video.
2. Detectar caras con MediaPipe en cada muestra.
3. K-means 2D con auto-K (silhouette) sobre las detecciones agregadas.
4. Emitir N centroides fijos (cx, cy, w, h).

Salida: `faces/face_positions.json`:
    [{"position_id": 0, "cx": 0.23, "cy": 0.45, "w": 0.12, "h": 0.18,
      "confidence": 0.82, "n_samples": 47}, ...]

Reemplaza a `face_tracker.py` (que hacía tracking temporal innecesario).
Tiempo objetivo: 20-40s para un video de 1h.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
try:
    import absl.logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
    _absl_logging.set_stderrthreshold(_absl_logging.ERROR)
except ImportError:
    pass

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir,
    find_video, find_ffmpeg_exe, find_ffprobe_exe, save_json,
)

logger = logging.getLogger(__name__)

_N_SAMPLES = 60
_MAX_SPEAKERS_DEFAULT = 5
# Filtro anti-perro/anti-objeto: los humanos en un podcast con plano estático
# suelen estar en el tercio superior-medio del frame (cy < 0.45). Objetos
# como mascotas en el suelo o patrones de muebles se detectan a veces como
# caras por MediaPipe pero tienen cy alto → se filtran.
_MAX_CY_FOR_HUMAN = 0.55


def detect_face_positions(
    *,
    project: str,
    episode: str,
    root: Path,
    force: bool = False,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    """Genera `faces/face_positions.json` con N centroides estáticos."""
    settings = load_settings(root)
    cfg = settings.get("diarization", {})
    face_confidence = float(cfg.get("face_confidence", 0.25))
    face_model_pref = cfg.get("face_model", "mediapipe")
    max_speakers = int(cfg.get("max_speakers", _MAX_SPEAKERS_DEFAULT))

    ep_dir = get_episode_dir(root, project, episode)
    faces_dir = ep_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    out_path = faces_dir / "face_positions.json"

    if out_path.exists() and not force:
        on_progress("face_positions", "face_positions.json ya existe", 100)
        return _load_json(out_path)

    video_path = find_video(ep_dir / "input")
    info = _probe_video(video_path)
    fps = info["fps"]
    duration = info["duration"]
    on_progress(
        "face_positions",
        f"Video: {info['width']}x{info['height']} @ {fps:.0f}fps, {duration:.0f}s "
        f"— muestreando {_N_SAMPLES} frames",
        5,
    )

    detector = _create_detector(root, face_confidence, face_model_pref)

    # Timestamps distribuidos uniformemente (evitando primeros/últimos 5s)
    t_min = min(5.0, duration * 0.05)
    t_max = max(duration - 5.0, duration * 0.95)
    timestamps = [t_min + (t_max - t_min) * i / (_N_SAMPLES - 1) for i in range(_N_SAMPLES)]

    cap = cv2.VideoCapture(str(video_path))
    all_detections: list[tuple[float, float, float, float, float]] = []  # (cx, cy, w, h, conf)

    last_pct = 5
    for i, t in enumerate(timestamps):
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        for d in _detect_faces(detector, frame):
            all_detections.append(d)
        pct = 5 + int((i / _N_SAMPLES) * 75)
        if pct - last_pct >= 10:
            on_progress(
                "face_positions",
                f"Muestra {i+1}/{_N_SAMPLES} — {len(all_detections)} caras acumuladas",
                pct,
            )
            last_pct = pct

    cap.release()
    _close_detector(detector)

    if not all_detections:
        on_progress("face_positions", "⚠ No se detectaron caras", 95)
        save_json(out_path, [])
        return []

    on_progress("face_positions", f"{len(all_detections)} caras detectadas — clustering", 85)

    n_clusters = _auto_k(all_detections, max_speakers)
    clusters = _cluster(all_detections, n_clusters)
    on_progress("face_positions", f"{n_clusters} posiciones identificadas (auto-k)", 92)

    positions: list[dict] = []
    filtered_low = 0
    for k_idx, cluster_dets in enumerate(clusters):
        if not cluster_dets:
            continue
        cx = float(np.mean([d[0] for d in cluster_dets]))
        cy = float(np.mean([d[1] for d in cluster_dets]))
        # Filtro anti-perro: si cy está muy abajo en el frame, probablemente
        # es un objeto, pet o falso positivo. Descartamos el cluster entero.
        if cy > _MAX_CY_FOR_HUMAN:
            filtered_low += 1
            logger.info("face_positions: cluster filtrado (cy=%.2f > %.2f) — probable no-humano", cy, _MAX_CY_FOR_HUMAN)
            continue
        w = float(np.mean([d[2] for d in cluster_dets]))
        h = float(np.mean([d[3] for d in cluster_dets]))
        confidence = float(np.mean([d[4] for d in cluster_dets]))
        positions.append({
            "position_id": k_idx,
            "cx": cx, "cy": cy, "w": w, "h": h,
            "confidence": round(confidence, 3),
            "n_samples": len(cluster_dets),
        })

    if filtered_low:
        on_progress("face_positions", f"Filtrados {filtered_low} clusters con cy>{_MAX_CY_FOR_HUMAN} (probables no-humanos)", 93)

    # Ordenar de izquierda a derecha y reasignar IDs
    positions.sort(key=lambda p: p["cx"])
    for new_id, p in enumerate(positions):
        p["position_id"] = new_id

    save_json(out_path, positions)

    # Thumbnails: usar el frame central del video para cada posición
    thumbs_dir = faces_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)
    mid_t = duration / 2
    for p in positions:
        _extract_thumbnail(video_path, mid_t, thumbs_dir / f"position_{p['position_id']}.jpg", crop=p, src=info)

    on_progress(
        "face_positions",
        f"{len(positions)} posiciones fijas detectadas (izq→der). Listo.",
        100,
    )
    return positions


# ── Clustering 2D ──────────────────────────────────────────────────

def _auto_k(detections: list, max_k: int) -> int:
    """Auto-K por silhouette score 2D."""
    if len(detections) < 4:
        return max(1, min(len(detections), max_k))
    points = np.array([[d[0], d[1] * 0.3] for d in detections])  # escala y para priorizar x

    best_k = 2
    best_score = -1.0
    for k in range(2, min(max_k + 1, len(detections))):
        centroids = _kmeans(points, k)
        assignments = _assign(points, centroids)
        score = _silhouette(points, assignments, k)
        if score > best_score:
            best_score = score
            best_k = k
    logger.info("face_positions auto-k=%d silhouette=%.3f", best_k, best_score)
    return best_k


def _cluster(detections: list, k: int) -> list[list]:
    points = np.array([[d[0], d[1] * 0.3] for d in detections])
    centroids = _kmeans(points, k)
    assignments = _assign(points, centroids)
    clusters: list[list] = [[] for _ in range(k)]
    for det, a in zip(detections, assignments):
        clusters[a].append(det)
    return clusters


def _kmeans(points: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
    centroids = np.column_stack([
        np.linspace(points[:, 0].min(), points[:, 0].max(), k),
        np.full(k, points[:, 1].mean()),
    ])
    for _ in range(max_iter):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        new_centroids = np.array([
            points[assignments == c].mean(axis=0) if np.any(assignments == c) else centroids[c]
            for c in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    return centroids


def _assign(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def _silhouette(points: np.ndarray, assignments: np.ndarray, k: int) -> float:
    n = len(points)
    if n < 2 or k < 2:
        return 0.0
    scores = []
    for i in range(min(n, 200)):
        ci = assignments[i]
        same = assignments == ci
        if same.sum() > 1:
            a_i = np.linalg.norm(points[same] - points[i], axis=1).sum() / (same.sum() - 1)
        else:
            a_i = 0.0
        b_i = float("inf")
        for c in range(k):
            if c == ci:
                continue
            other = assignments == c
            if not np.any(other):
                continue
            b_i = min(b_i, float(np.linalg.norm(points[other] - points[i], axis=1).mean()))
        if b_i == float("inf"):
            b_i = 0.0
        denom = max(a_i, b_i)
        scores.append((b_i - a_i) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


# ── Detección MediaPipe / OpenCV ──────────────────────────────────

def _create_detector(root: Path, confidence: float, prefer: str):
    if _HAS_MEDIAPIPE and prefer != "opencv":
        for model in ["blaze_face_full_range.tflite", "blaze_face_short_range.tflite"]:
            mp_path = root / "models" / model
            if mp_path.exists():
                try:
                    opts = mp_vision.FaceDetectorOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(mp_path)),
                        min_detection_confidence=confidence,
                    )
                    return ("mediapipe", mp_vision.FaceDetector.create_from_options(opts))
                except Exception as exc:
                    logger.warning("Fallo cargando %s: %s", model, exc)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return ("opencv", cascade)


def _detect_faces(detector_tuple, frame: np.ndarray):
    kind, d = detector_tuple
    h, w = frame.shape[:2]
    out = []
    if kind == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = d.detect(img)
        if res.detections:
            for det in res.detections:
                bb = det.bounding_box
                cx = (bb.origin_x + bb.width / 2) / w
                cy = (bb.origin_y + bb.height / 2) / h
                conf = det.categories[0].score if det.categories else 0.5
                out.append((cx, cy, bb.width / w, bb.height / h, float(conf)))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = d.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, rw, rh) in rects:
            out.append(((x + rw / 2) / w, (y + rh / 2) / h, rw / w, rh / h, 0.5))
    return out


def _close_detector(detector_tuple) -> None:
    kind, d = detector_tuple
    if kind == "mediapipe":
        d.close()


# ── Helpers ────────────────────────────────────────────────────────

def _probe_video(video_path: Path) -> dict:
    cmd = [
        find_ffprobe_exe(), "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", "-select_streams", "v:0", str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    stream = info["streams"][0]
    fps_parts = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and float(fps_parts[1]) > 0 else 30.0
    duration = float(info.get("format", {}).get("duration", 0) or stream.get("duration", 0) or 0)
    if duration <= 0 and stream.get("nb_frames"):
        duration = int(stream["nb_frames"]) / fps
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
    }


def _extract_thumbnail(video_path: Path, t: float, out: Path, crop: dict | None = None, src: dict | None = None) -> None:
    vf = None
    if crop and src:
        sw, sh = src["width"], src["height"]
        cx_px = int(crop["cx"] * sw)
        cy_px = int(crop["cy"] * sh)
        cw = max(80, int(crop["w"] * sw * 1.2))
        ch = max(80, int(crop["h"] * sh * 1.3))
        vf = f"crop={cw}:{ch}:{cx_px - cw // 2}:{cy_px - ch // 2},scale=160:-1"
    cmd = [
        find_ffmpeg_exe(), "-y", "-ss", str(t), "-i", str(video_path),
        "-frames:v", "1", "-q:v", "3",
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += [str(out)]
    subprocess.run(cmd, capture_output=True, check=False)


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["detect_face_positions"]
