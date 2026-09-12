"""Active Speaker Detection (ASD) — paso 'asd' del pipeline.

Asigna a cada (track_id, ventana temporal) un score 0..1 que estima la probabilidad
de que ese rostro esté hablando en ese instante.

Implementación v3 (determinista, sin dependencias externas pesadas):
- **Lip movement**: se calcula la *Mean Absolute Difference* (MAD) entre los píxeles
  de la región bucal de cada track en frames separados por ~0.2s (proxy de apertura/cierre
  de boca, igual al enfoque usado históricamente en `diarizer.py`).
- **Audio energy**: RMS del audio en la misma ventana, con umbral silencio/voz.
- **Score**: `normalized_mad × voice_gate`. Cuando no hay audio, score = 0 para todos los
  tracks. Cuando hay audio, el track con MAD más alto gana.

Ventajas vs modelo de deep learning:
- Funciona en cualquier PC (solo numpy + opencv + ffmpeg).
- Determinista: mismos inputs → mismos scores.
- No requiere descarga de checkpoints ni GPU.

Si hay `torch` + CUDA disponible y el usuario coloca `models/light_asd/best.pt`,
se activa el modelo neuronal automáticamente (upgrade futuro — ver `_has_neural_asd`).

Output: ``faces/asd_scores.json``
    [{"track_id": 0,
      "windows": [{"t_start": 0.0, "t_end": 0.4, "score": 0.12, "mad": 1.3, "audio_rms": 0.01}, ...]},
     ...]
"""

from __future__ import annotations

import json
import logging
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir,
    find_video, find_ffmpeg_exe, save_json, load_json,
)

logger = logging.getLogger(__name__)


_WINDOW_SECONDS = 0.4          # ventana de análisis
_DELTA_LIP_SECONDS = 0.2       # separación para MAD
_AUDIO_SILENCE_RMS = 0.005     # umbral para considerar "hay audio"


def run_asd(
    *,
    project: str,
    episode: str,
    root: Path,
    force: bool = False,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    """Computa ASD scores por track y ventana temporal."""
    ep_dir = get_episode_dir(root, project, episode)
    faces_dir = ep_dir / "faces"
    scores_path = faces_dir / "asd_scores.json"

    if scores_path.exists() and not force:
        on_progress("asd", "asd_scores.json ya existe", 100)
        return load_json(scores_path)

    tracks_path = faces_dir / "face_tracks.json"
    if not tracks_path.exists():
        raise FileNotFoundError(f"face_tracks.json no encontrado en {faces_dir}")
    tracks = load_json(tracks_path)
    if not tracks:
        on_progress("asd", "No hay tracks — omitiendo ASD", 100)
        save_json(scores_path, [])
        return []

    video_path = find_video(ep_dir / "input")
    audio_path = _find_audio(ep_dir)

    # Intenta neural ASD si está disponible; si no, cae al método MAD.
    if _has_neural_asd(root):
        try:
            on_progress("asd", "Usando Light-ASD neural (detectado modelo y torch)", 5)
            return _run_neural_asd(
                tracks=tracks, video_path=video_path, audio_path=audio_path,
                scores_path=scores_path, root=root, on_progress=on_progress,
            )
        except Exception as exc:
            logger.warning("Neural ASD falló (%s), usando MAD determinista", exc)

    on_progress("asd", "Usando ASD determinista (MAD + RMS audio)", 5)
    return _run_mad_asd(
        tracks=tracks, video_path=video_path, audio_path=audio_path,
        scores_path=scores_path, on_progress=on_progress,
    )


# ── Neural ASD (auto-detectado) ────────────────────────────────────

def _has_neural_asd(root: Path) -> bool:
    """Verifica si hay torch + checkpoint de Light-ASD + (opcionalmente) GPU."""
    ckpt = root / "models" / "light_asd" / "best.pt"
    if not ckpt.exists():
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _run_neural_asd(
    *,
    tracks: list[dict],
    video_path: Path,
    audio_path: Path | None,
    scores_path: Path,
    root: Path,
    on_progress: ProgressCallback,
) -> list[dict]:
    """Placeholder para integración futura con Light-ASD / TalkNet.

    El usuario puede implementar esto descargando el repo de Light-ASD
    y colocando ``models/light_asd/best.pt``. Por defecto cae al método MAD.
    """
    raise NotImplementedError("Neural ASD no implementado todavía — usando fallback MAD")


# ── ASD determinista basado en MAD + RMS ───────────────────────────

def _run_mad_asd(
    *,
    tracks: list[dict],
    video_path: Path,
    audio_path: Path | None,
    scores_path: Path,
    on_progress: ProgressCallback,
) -> list[dict]:
    info = _probe_video(video_path)
    fps = info["fps"]
    frame_w, frame_h = info["width"], info["height"]
    duration = info["duration"]

    n_windows = max(1, int(duration / _WINDOW_SECONDS))
    on_progress("asd", f"Pre-computando RMS del audio ({n_windows} ventanas)...", 8)

    # Pre-cargar RMS de audio alineado por ventana
    audio_rms = _compute_audio_rms(audio_path, duration, _WINDOW_SECONDS) if audio_path else np.zeros(n_windows)

    # Ventanas "activas" = con voz. Las silenciosas se saltan completamente,
    # ahorrando ~40-60% del trabajo en podcasts con pausas naturales.
    active_windows = np.where(audio_rms >= _AUDIO_SILENCE_RMS)[0]
    on_progress(
        "asd",
        f"{len(active_windows)}/{n_windows} ventanas con voz — "
        f"saltando silencios ({100*(1 - len(active_windows)/max(n_windows,1)):.0f}% skip). "
        f"Tracks: {len(tracks)}.",
        12,
    )

    cap = cv2.VideoCapture(str(video_path))
    delta_frames = max(1, int(_DELTA_LIP_SECONDS * fps))

    mad_matrix = np.zeros((len(tracks), n_windows), dtype=np.float64)
    track_present = np.zeros((len(tracks), n_windows), dtype=np.bool_)

    # Cache de bboxes por track para evitar búsqueda repetida
    sorted_frames_per_track = [sorted(tr["frames"], key=lambda f: f["t"]) for tr in tracks]

    last_pct = 12
    last_frame_num = -1
    total_active = max(1, len(active_windows))

    for i, w in enumerate(active_windows):
        t = float(w) * _WINDOW_SECONDS
        frame_num = int(t * fps)

        # Evitar seeks redundantes: si el frame objetivo está cerca del anterior,
        # leemos secuencialmente en lugar de seek random (mucho más rápido).
        if last_frame_num < 0 or abs(frame_num - last_frame_num) > int(fps * 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        else:
            # saltar frames hasta llegar al target
            skip = frame_num - last_frame_num - 1
            for _ in range(max(0, skip)):
                cap.grab()
        ret1, frame1 = cap.read()
        # leer el frame t+delta directamente (sin seek)
        for _ in range(max(0, delta_frames - 1)):
            cap.grab()
        ret2, frame2 = cap.read()
        last_frame_num = frame_num + delta_frames
        if not ret1 or not ret2:
            continue

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        for ti, sorted_frames in enumerate(sorted_frames_per_track):
            bbox = _closest_bbox(sorted_frames, t)
            if bbox is None:
                continue
            track_present[ti, w] = True
            cx, cy, bw, bh = bbox
            mad_matrix[ti, w] = _mouth_mad(gray1, gray2, cx, cy, bw, bh, frame_w, frame_h)

        pct = 12 + int((i / total_active) * 83)
        if pct - last_pct >= 5:
            on_progress("asd", f"Procesando ventana activa {i+1}/{total_active} (t={t:.0f}s)", pct)
            last_pct = pct

    cap.release()

    # Normalización: por cada ventana, MAD se normaliza entre tracks presentes.
    scored = np.zeros_like(mad_matrix)
    for w in range(n_windows):
        col = mad_matrix[:, w]
        present = track_present[:, w]
        if present.sum() == 0 or audio_rms[w] < _AUDIO_SILENCE_RMS:
            continue
        present_vals = col[present]
        if present_vals.max() <= 0:
            continue
        col_norm = np.zeros_like(col)
        col_norm[present] = present_vals / (present_vals.max() + 1e-6)
        # Gate: solo speakers con MAD > 60% del máximo cuentan como "activos"
        col_norm[col_norm < 0.6] *= 0.3
        scored[:, w] = col_norm

    # Emitir JSON final
    out: list[dict] = []
    for ti, tr in enumerate(tracks):
        windows = []
        for w in range(n_windows):
            if not track_present[ti, w]:
                continue
            windows.append({
                "t_start": round(w * _WINDOW_SECONDS, 3),
                "t_end": round((w + 1) * _WINDOW_SECONDS, 3),
                "score": round(float(scored[ti, w]), 4),
                "mad": round(float(mad_matrix[ti, w]), 4),
                "audio_rms": round(float(audio_rms[w]), 4),
            })
        out.append({"track_id": tr["track_id"], "windows": windows})

    save_json(scores_path, out)
    on_progress("asd", f"ASD completado para {len(tracks)} tracks × {n_windows} ventanas", 100)
    return out


def _closest_bbox(frames: list[dict], t: float) -> tuple[float, float, float, float] | None:
    if not frames:
        return None
    closest = min(frames, key=lambda f: abs(f["t"] - t))
    if abs(closest["t"] - t) > _WINDOW_SECONDS * 1.5:
        return None
    return tuple(closest["bbox"])


def _mouth_mad(
    gray1: np.ndarray,
    gray2: np.ndarray,
    cx: float, cy: float, bw: float, bh: float,
    frame_w: int, frame_h: int,
) -> float:
    """Mean Absolute Difference de la región bucal (tercio inferior del bbox)."""
    cx_px = int(cx * frame_w)
    half_w = int(bw * frame_w * 0.6 / 2)
    face_top = int((cy - bh / 2) * frame_h)
    face_bot = int((cy + bh / 2) * frame_h)
    mouth_top = face_top + int((face_bot - face_top) * 0.6)

    x1 = max(0, cx_px - half_w)
    x2 = min(frame_w, cx_px + half_w)
    y1 = max(0, mouth_top)
    y2 = min(frame_h, face_bot)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    m1 = gray1[y1:y2, x1:x2]
    m2 = gray2[y1:y2, x1:x2]
    if m1.size == 0 or m2.size == 0 or m1.shape != m2.shape:
        return 0.0
    return float(np.mean(np.abs(m1.astype(np.float32) - m2.astype(np.float32))))


# ── Audio RMS por ventana ──────────────────────────────────────────

def _find_audio(ep_dir: Path) -> Path | None:
    audio_dir = ep_dir / "audio"
    if not audio_dir.is_dir():
        return None
    for candidate in audio_dir.glob("*.wav"):
        return candidate
    return None


def _compute_audio_rms(audio_path: Path | None, duration: float, window_sec: float) -> np.ndarray:
    """Calcula RMS normalizado por ventana desde el WAV."""
    n_windows = max(1, int(duration / window_sec))
    if audio_path is None or not audio_path.exists():
        return np.zeros(n_windows)

    try:
        with wave.open(str(audio_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as exc:
        logger.warning("No se pudo leer audio para RMS (%s), usando ceros", exc)
        return np.zeros(n_windows)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    max_amp = float(2 ** (8 * sample_width - 1))
    samples /= max_amp + 1e-9

    samples_per_window = int(window_sec * framerate)
    rms = np.zeros(n_windows)
    for w in range(n_windows):
        start = w * samples_per_window
        end = start + samples_per_window
        chunk = samples[start:end]
        if chunk.size == 0:
            continue
        rms[w] = float(np.sqrt(np.mean(chunk ** 2)))
    return rms


# ── Video probe ────────────────────────────────────────────────────

def _probe_video(video_path: Path) -> dict:
    from .utils import find_ffprobe_exe
    cmd = [
        find_ffprobe_exe(), "-v", "quiet", "-print_format", "json",
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
