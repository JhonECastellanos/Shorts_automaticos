"""Detección y clustering de caras en video — paso 'detect' del pipeline."""

import json
import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

# Suprimir warnings de TensorFlow Lite / MediaPipe antes de importar
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Suppress XNNPACK delegate info messages
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
    find_video, find_ffmpeg_exe, find_ffprobe_exe, save_json, load_json,
)

logger = logging.getLogger(__name__)


def detect_faces(
    *,
    project: str,
    episode: str,
    force: bool = False,
    root: Path,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    """Detecta y agrupa posiciones faciales en el video.

    Produce ``calibration/face_slots.json`` (lista de {cx, cy, w, h})
    y thumbnails por slot.  NO asigna nombres de speaker.
    """
    settings = load_settings(root)
    cfg_diarization = settings.get("diarization", {})
    face_confidence = cfg_diarization.get("face_confidence", 0.25)
    face_model_pref = cfg_diarization.get("face_model", "mediapipe")

    ep_dir = get_episode_dir(root, project, episode)
    calibration_dir = ep_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    slots_path = calibration_dir / "face_slots.json"
    if slots_path.exists() and not force:
        on_progress("detect", "face_slots.json ya existe", 100)
        return load_json(slots_path)

    video_path = find_video(ep_dir / "input")

    # Obtener info del video
    video_info = _get_video_info(video_path)
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    on_progress("detect", f"Video: {width}x{height} @ {fps:.1f} fps — modelo: {face_model_pref}, confianza: {face_confidence}", 10)

    # ── Paso 1: Muestreo adaptativo de frames ──
    total_frames = int(video_info.get("nb_frames", 0))
    if total_frames <= 0:
        cap_tmp = cv2.VideoCapture(str(video_path))
        total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_tmp.release()
    total_duration = total_frames / fps if fps > 0 else 0

    # Muestreo adaptativo: más frames para videos largos (50-150)
    base_samples = 50
    extra_per_minute = 2
    minutes = total_duration / 60
    n_sample_frames = min(150, base_samples + int(extra_per_minute * minutes))
    sample_timestamps = [
        total_duration * i / n_sample_frames
        for i in range(1, n_sample_frames + 1)
    ]
    on_progress("detect", f"Muestreando {n_sample_frames} frames (adaptativo, {minutes:.0f} min video)", 15)

    # Crear detector de caras
    mp_face = _create_face_detector(root, confidence=face_confidence, prefer=face_model_pref)

    cap = cv2.VideoCapture(str(video_path))
    all_face_detections: list[tuple[float, float, float, float]] = []

    for i, ts in enumerate(sample_timestamps):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        faces = _detect_faces(mp_face, frame)
        all_face_detections.extend(faces)
        if (i + 1) % 30 == 0:
            pct = 15 + int((i / n_sample_frames) * 30)
            on_progress("detect", f"Frame {i+1}/{n_sample_frames}: {len(all_face_detections)} caras", pct)

    cap.release()
    _close_detector(mp_face)

    logger.info("Muestreo adaptativo: %d caras en %d frames", len(all_face_detections), n_sample_frames)
    on_progress("detect", f"{len(all_face_detections)} caras detectadas en muestreo adaptativo", 50)

    # ── Paso 2: Clustering de caras por posición horizontal ──
    max_speakers = cfg_diarization.get("max_speakers", 5)

    # Intentar auto-detectar el número óptimo de clusters con silhouette score (2D)
    n_clusters = _auto_detect_n_clusters_2d(all_face_detections, max_speakers)

    if len(all_face_detections) >= n_clusters and n_clusters > 0:
        face_slots = _cluster_faces_2d(all_face_detections, n_clusters)
    else:
        # Fallback: distribuir slots equidistantemente
        face_slots = []
        for i in range(max(n_clusters, 1)):
            cx = (i + 0.5) / max(n_clusters, 1)
            face_slots.append({"cx": cx, "cy": 0.4, "w": 0.12, "h": 0.15})

    on_progress("detect", f"{len(face_slots)} slots de posición identificados", 70)

    # ── Paso 3: Generar thumbnails por slot ──
    thumbs_dir = calibration_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)

    # Usar el timestamp más cercano al centroide de cada slot
    for slot_idx, slot in enumerate(face_slots):
        best_ts = total_duration * slot["cx"]  # aproximación por posición relativa
        best_ts = max(1.0, min(best_ts, total_duration - 1.0))
        _extract_thumbnail(video_path, best_ts, thumbs_dir / f"slot_{slot_idx}.jpg")

    save_json(slots_path, face_slots)
    on_progress("detect", f"{len(face_slots)} face slots detectados y guardados", 100)
    return face_slots


# ── Auto-detección de N clusters ──────────────────────────────────────────

def _auto_detect_n_clusters_2d(
    detections: list[tuple[float, float, float, float]],
    max_k: int,
) -> int:
    """Estima el número óptimo de clusters usando silhouette score 2D (cx, cy)."""
    if len(detections) < 4:
        return max(1, min(len(detections), max_k))

    # Usar cx (peso 1.0) y cy (peso 0.3) para clustering 2D
    points = np.array([[d[0], d[1] * 0.3] for d in detections])

    best_k = 2
    best_score = -1.0

    for k in range(2, min(max_k + 1, len(detections))):
        centroids = _kmeans_2d(points, k)
        assignments = _assign_2d(points, centroids)
        score = _silhouette_2d(points, assignments, k)
        if score > best_score:
            best_score = score
            best_k = k

    logger.info("Auto-detect clusters 2D: best_k=%d, silhouette=%.3f", best_k, best_score)
    return best_k


def _kmeans_2d(points: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
    """K-means simple para puntos 2D."""
    n = len(points)
    # Inicializar centroids espaciados uniformemente en X
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
    """Calcula silhouette score para datos 2D."""
    n = len(points)
    if n < 2 or k < 2:
        return 0.0

    scores = []
    for i in range(min(n, 200)):  # Limitar para performance en muestras grandes
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


# ── Clustering 2D ─────────────────────────────────────────────────────────

def _cluster_faces_2d(
    detections: list[tuple[float, float, float, float]],
    n_clusters: int,
) -> list[dict]:
    """Agrupa detecciones de cara en N clusters por posición (cx, cy) usando k-means 2D."""
    points = np.array([[d[0], d[1] * 0.3] for d in detections])
    centroids = _kmeans_2d(points, n_clusters)
    assignments = _assign_2d(points, centroids)

    slots = []
    for k in range(n_clusters):
        mask = assignments == k
        if np.any(mask):
            cluster_dets = [detections[i] for i in range(len(detections)) if mask[i]]
            avg_cx = sum(d[0] for d in cluster_dets) / len(cluster_dets)
            avg_cy = sum(d[1] for d in cluster_dets) / len(cluster_dets)
            avg_w = sum(d[2] for d in cluster_dets) / len(cluster_dets)
            avg_h = sum(d[3] for d in cluster_dets) / len(cluster_dets)
            slots.append({"cx": avg_cx, "cy": avg_cy, "w": avg_w, "h": avg_h})
        else:
            slots.append({
                "cx": float(centroids[k][0]),
                "cy": float(centroids[k][1] / 0.3),
                "w": 0.12,
                "h": 0.15,
            })

    # Ordenar slots de izquierda a derecha
    slots.sort(key=lambda s: s["cx"])
    return slots


# ── Detección de caras con fallback chain ──────────────────────────────────

def _create_face_detector(root: Path, *, confidence: float = 0.25, prefer: str = "mediapipe"):
    """Crea un detector de caras: full_range → short_range → OpenCV Haar."""
    if _HAS_MEDIAPIPE and prefer != "opencv":
        for model_name in [
            "blaze_face_full_range.tflite",
            "blaze_face_short_range.tflite",
        ]:
            model_path = root / "models" / model_name
            if model_path.exists():
                try:
                    options = mp_vision.FaceDetectorOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                        min_detection_confidence=confidence,
                    )
                    detector = mp_vision.FaceDetector.create_from_options(options)
                    logger.info("Detector MediaPipe creado: %s (conf=%.2f)", model_name, confidence)
                    return ("mediapipe", detector)
                except Exception as e:
                    logger.warning("Fallo cargando %s: %s", model_name, e)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    logger.info("Usando fallback OpenCV Haar cascade para detección de caras")
    return ("opencv", cascade)


def _detect_faces(detector_tuple, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Detecta caras en un frame. Retorna [(cx_norm, cy_norm, w_norm, h_norm)]."""
    kind, detector = detector_tuple
    h, w = frame.shape[:2]
    faces = []

    if kind == "mediapipe":
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)
        if results.detections:
            for det in results.detections:
                bb = det.bounding_box
                cx = (bb.origin_x + bb.width / 2) / w
                cy = (bb.origin_y + bb.height / 2) / h
                faces.append((cx, cy, bb.width / w, bb.height / h))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, rw, rh) in rects:
            cx = (x + rw / 2) / w
            cy = (y + rh / 2) / h
            faces.append((cx, cy, rw / w, rh / h))

    return faces


def _close_detector(detector_tuple) -> None:
    kind, detector = detector_tuple
    if kind == "mediapipe":
        detector.close()


# ── Utilidades ─────────────────────────────────────────────────────────────

def _get_video_info(video_path: Path) -> dict:
    ffprobe = find_ffprobe_exe()
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    fps_parts = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": int(stream.get("nb_frames", 0)),
    }


def _extract_thumbnail(video_path: Path, timestamp: float, out: Path) -> None:
    cmd = [
        find_ffmpeg_exe(), "-y", "-ss", str(timestamp),
        "-i", str(video_path), "-frames:v", "1",
        "-q:v", "2", str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=False)
