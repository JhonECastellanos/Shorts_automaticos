"""Tracking facial para plano general fijo — paso 'face_track' del pipeline.

Contexto: el video es UN SOLO PLANO ESTÁTICO donde N personas (típicamente 4-5)
permanecen aproximadamente en la misma zona espacial durante todo el video.
Muestrear a 5fps los 60 minutos completos sería desperdicio (y lento: 4h).

Enfoque v3.2:
1. Muestrear M_SAMPLES=120 frames distribuidos uniformemente en el video.
2. Detectar caras en cada frame muestreado (MediaPipe + fallback OpenCV).
3. Clustering 2D (k-means con auto-K por silhouette) de TODAS las detecciones.
4. Cada cluster = una persona. Se genera un track cubriendo todo el video,
   con bbox = promedio del cluster + los timestamps de las muestras donde
   apareció esa zona.
5. ASD (siguiente paso) usa el bbox promedio para muestrear la región bucal
   — en plano fijo el bbox casi no varía, así que esto funciona bien.

Output: ``faces/face_tracks.json``
    [{"track_id": 0,
      "centroid": [cx, cy],
      "bbox_mean": [cx, cy, w, h],
      "first_seen": t0, "last_seen": tN,
      "duration": tN-t0,
      "frames": [{"t": 1.23, "bbox": [...], "confidence": ...}, ...],
      "n_frames": int}, ...]

Coordenadas normalizadas a [0, 1] relativas al frame.
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


# ── Parámetros ─────────────────────────────────────────────────────
_DEFAULT_SAMPLES = 120         # frames a procesar en todo el video
_MIN_SAMPLES = 60              # mínimo absoluto (videos muy cortos)
_MAX_SAMPLES = 240             # cap para videos muy largos
_MAX_SPEAKERS_FALLBACK = 5


def track_faces(
    *,
    project: str,
    episode: str,
    root: Path,
    force: bool = False,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    """Genera ``faces/face_tracks.json`` con tracks pseudo-permanentes por zona facial."""
    settings = load_settings(root)
    cfg = settings.get("diarization", {})
    face_confidence = float(cfg.get("face_confidence", 0.25))
    face_model_pref = cfg.get("face_model", "mediapipe")
    max_speakers = int(cfg.get("max_speakers", _MAX_SPEAKERS_FALLBACK))

    ep_dir = get_episode_dir(root, project, episode)
    faces_dir = ep_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = faces_dir / "face_tracks.json"

    if tracks_path.exists() and not force:
        on_progress("face_track", "face_tracks.json ya existe", 100)
        return _load_json(tracks_path)

    video_path = find_video(ep_dir / "input")
    info = _get_video_info(video_path)
    fps = info["fps"]
    duration = info["duration"]

    # Número de samples adaptativo: base 120, +1 por cada minuto extra sobre 5 min
    minutes = duration / 60.0
    n_samples = max(_MIN_SAMPLES, min(_MAX_SAMPLES, int(_DEFAULT_SAMPLES + max(0, minutes - 5) * 1)))
    on_progress(
        "face_track",
        f"Video: {info['width']}x{info['height']} @ {fps:.0f}fps, {duration:.0f}s ({minutes:.1f}min) "
        f"— muestreando {n_samples} frames ({n_samples/duration:.2f}fps efectivos)",
        5,
    )

    detector = _create_face_detector(root, confidence=face_confidence, prefer=face_model_pref)

    timestamps = [duration * i / n_samples for i in range(1, n_samples + 1)]

    cap = cv2.VideoCapture(str(video_path))
    all_detections: list[tuple[float, tuple[float, float, float, float], float]] = []
    # (timestamp, bbox, confidence)

    last_pct = 5
    for i, t in enumerate(timestamps):
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        dets = _detect_faces(detector, frame)
        for d in dets:
            all_detections.append((t, d["bbox"], d["confidence"]))

        pct = 5 + int((i / max(1, n_samples)) * 70)  # 5% → 75%
        if pct - last_pct >= 5:
            on_progress(
                "face_track",
                f"Muestra {i+1}/{n_samples} — detecciones acumuladas: {len(all_detections)}",
                pct,
            )
            last_pct = pct

    cap.release()
    _close_detector(detector)

    if not all_detections:
        on_progress("face_track", "⚠ No se detectaron caras — emitiendo tracks vacíos", 95)
        save_json(tracks_path, [])
        return []

    on_progress("face_track", f"{len(all_detections)} detecciones totales — clustering por zona", 78)

    # ── Clustering 2D con auto-K ──
    n_clusters = _auto_detect_k(all_detections, max_speakers)
    clusters = _cluster_detections(all_detections, n_clusters)
    on_progress("face_track", f"{n_clusters} clusters detectados (auto-k por silhouette)", 85)

    # ── Construir tracks ──
    tracks: list[dict] = []
    for k_idx, (bboxes_conf, timestamps_list) in enumerate(clusters):
        if not bboxes_conf:
            continue
        cx = float(np.mean([b[0] for b, _ in bboxes_conf]))
        cy = float(np.mean([b[1] for b, _ in bboxes_conf]))
        bw = float(np.mean([b[2] for b, _ in bboxes_conf]))
        bh = float(np.mean([b[3] for b, _ in bboxes_conf]))

        frames = [
            {"t": round(t, 3), "bbox": list(b), "confidence": float(c)}
            for (b, c), t in zip(bboxes_conf, timestamps_list)
        ]
        frames.sort(key=lambda f: f["t"])

        tracks.append({
            "track_id": k_idx,
            "centroid": [cx, cy],
            "bbox_mean": [cx, cy, bw, bh],
            "first_seen": frames[0]["t"],
            "last_seen": frames[-1]["t"],
            "duration": round(frames[-1]["t"] - frames[0]["t"], 2),
            "frames": frames,
            "n_frames": len(frames),
        })

    # Ordenar tracks de izquierda a derecha (por cx) y reasignar IDs
    tracks.sort(key=lambda t: t["centroid"][0])
    for new_id, tr in enumerate(tracks):
        tr["track_id"] = new_id

    save_json(tracks_path, tracks)

    # Thumbnails por track: usar el frame de mayor confidence
    thumbs_dir = faces_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)
    for tr in tracks:
        best = max(tr["frames"], key=lambda f: f.get("confidence", 0))
        _extract_thumbnail(video_path, best["t"], thumbs_dir / f"track_{tr['track_id']}.jpg")

    on_progress(
        "face_track",
        f"{len(tracks)} tracks generados (1 por zona facial) — bbox promedio + timestamps reales",
        100,
    )
    return tracks


# ── Clustering 2D ──────────────────────────────────────────────────

def _auto_detect_k(detections: list, max_k: int) -> int:
    """Auto-detecta el número óptimo de clusters usando silhouette sobre (cx, cy)."""
    if len(detections) < 4:
        return max(1, min(len(detections), max_k))

    points = np.array([[d[1][0], d[1][1] * 0.3] for d in detections])

    best_k = 2
    best_score = -1.0
    for k in range(2, min(max_k + 1, len(detections))):
        centroids = _kmeans_2d(points, k)
        assignments = _assign_2d(points, centroids)
        score = _silhouette_2d(points, assignments, k)
        if score > best_score:
            best_score = score
            best_k = k

    logger.info("face_track auto-k: best_k=%d, silhouette=%.3f", best_k, best_score)
    return best_k


def _cluster_detections(detections: list, n_clusters: int):
    """Agrupa detecciones en clusters 2D por posición (cx, cy).

    Devuelve list de (bboxes_con_conf, timestamps) por cluster.
    Cada item: ([(bbox, conf), ...], [t, ...]).
    """
    points = np.array([[d[1][0], d[1][1] * 0.3] for d in detections])
    centroids = _kmeans_2d(points, n_clusters)
    assignments = _assign_2d(points, centroids)

    clusters = [([], []) for _ in range(n_clusters)]
    for det, k in zip(detections, assignments):
        t, bbox, conf = det
        clusters[k][0].append((bbox, conf))
        clusters[k][1].append(t)
    return clusters


def _kmeans_2d(points: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
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


def _assign_2d(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def _silhouette_2d(points: np.ndarray, assignments: np.ndarray, k: int) -> float:
    n = len(points)
    if n < 2 or k < 2:
        return 0.0
    scores = []
    for i in range(min(n, 200)):
        ci = assignments[i]
        same_mask = assignments == ci
        n_same = same_mask.sum()
        if n_same > 1:
            dists_same = np.linalg.norm(points[same_mask] - points[i], axis=1)
            a_i = dists_same.sum() / (n_same - 1)
        else:
            a_i = 0.0
        b_i = float("inf")
        for c in range(k):
            if c == ci:
                continue
            other_mask = assignments == c
            if not np.any(other_mask):
                continue
            dist = np.linalg.norm(points[other_mask] - points[i], axis=1).mean()
            b_i = min(b_i, dist)
        if b_i == float("inf"):
            b_i = 0.0
        denom = max(a_i, b_i)
        scores.append((b_i - a_i) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


# ── Detección de caras ─────────────────────────────────────────────

def _create_face_detector(root: Path, *, confidence: float, prefer: str):
    if _HAS_MEDIAPIPE and prefer != "opencv":
        for model_name in ["blaze_face_full_range.tflite", "blaze_face_short_range.tflite"]:
            model_path = root / "models" / model_name
            if model_path.exists():
                try:
                    options = mp_vision.FaceDetectorOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                        min_detection_confidence=confidence,
                    )
                    detector = mp_vision.FaceDetector.create_from_options(options)
                    logger.info("Detector MediaPipe: %s", model_name)
                    return ("mediapipe", detector)
                except Exception as exc:
                    logger.warning("Fallo cargando %s: %s", model_name, exc)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    logger.info("Fallback OpenCV Haar cascade")
    return ("opencv", cascade)


def _detect_faces(detector_tuple, frame: np.ndarray) -> list[dict]:
    kind, detector = detector_tuple
    h, w = frame.shape[:2]
    out: list[dict] = []
    if kind == "mediapipe":
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)
        if results.detections:
            for det in results.detections:
                bb = det.bounding_box
                cx = (bb.origin_x + bb.width / 2) / w
                cy = (bb.origin_y + bb.height / 2) / h
                conf = det.categories[0].score if det.categories else 0.5
                out.append({"bbox": (cx, cy, bb.width / w, bb.height / h), "confidence": float(conf)})
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, rw, rh) in rects:
            cx = (x + rw / 2) / w
            cy = (y + rh / 2) / h
            out.append({"bbox": (cx, cy, rw / w, rh / h), "confidence": 0.5})
    return out


def _close_detector(detector_tuple) -> None:
    kind, detector = detector_tuple
    if kind == "mediapipe":
        detector.close()


# ── Video helpers ──────────────────────────────────────────────────

def _get_video_info(video_path: Path) -> dict:
    ffprobe = find_ffprobe_exe()
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", "-select_streams", "v:0", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
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


def _extract_thumbnail(video_path: Path, timestamp: float, out: Path) -> None:
    cmd = [
        find_ffmpeg_exe(), "-y", "-ss", str(timestamp),
        "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=False)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
