"""Speaker ↔ Position binding — paso 'speaker_bind' del pipeline v3.5.

Reemplaza al par ASD + CMIA con un binding determinista y mucho más simple:

1. Correr una **correlación ligera** de MAD labial **solo sobre los primeros
   N minutos** del video (default 5 min, configurable). No toda la hora.
2. Para cada `SPEAKER_xx` que aparece en ese intervalo, calcular cuál de las
   `face_positions` tiene mayor movimiento bucal durante los turnos del speaker.
3. Matching Hungarian 1:1 para evitar colisiones.
4. Si la confianza es baja (< 0.4), se marca `needs_manual: true` — la UI
   puede mostrar un modal de calibración manual.

Ventajas vs ASD+CMIA v3:
- 10-20× más rápido (5 min vs 60 min de video procesado).
- Mucho menos ruidoso (los primeros minutos de un podcast suelen tener
  intervenciones claras de cada participante).
- La asignación es ESTÁTICA: se usa el mismo mapa durante todo el video.

Inputs:
    - diarization/speaker_segments.json
    - faces/face_positions.json
    - audio/*.wav (para RMS gating)

Outputs:
    - diarization/speaker_face_map.json (mismo schema que CMIA)
    - calibration/speaker_zones.json    (compat con camera_switcher/exporter)
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
    find_video, find_ffprobe_exe, save_json, load_json,
)

logger = logging.getLogger(__name__)

_CALIBRATION_WINDOW_SEC = 300.0   # primeros 5 min
_MIN_SEGMENT_DUR = 1.0            # turnos < 1s se ignoran
_MAX_SEGMENTS_PER_SPEAKER = 20    # cap de cómputo
_DELTA_LIP_SEC = 0.2
_AUDIO_SILENCE_RMS = 0.005
_MIN_CONFIDENCE = 0.4


def bind_speakers_to_positions(
    *,
    project: str,
    episode: str,
    root: Path,
    force: bool = False,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    settings = load_settings(root)
    cfg = settings.get("diarization", {})
    calibration_sec = float(cfg.get("calibration_window_sec", _CALIBRATION_WINDOW_SEC))

    ep_dir = get_episode_dir(root, project, episode)
    diar_dir = ep_dir / "diarization"
    faces_dir = ep_dir / "faces"
    cal_dir = ep_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    map_path = diar_dir / "speaker_face_map.json"
    zones_path = cal_dir / "speaker_zones.json"

    if map_path.exists() and zones_path.exists() and not force:
        on_progress("speaker_bind", "speaker_face_map.json ya existe", 100)
        return load_json(map_path)

    segments_path = diar_dir / "speaker_segments.json"
    positions_path = faces_dir / "face_positions.json"
    for required in (segments_path, positions_path):
        if not required.exists():
            raise FileNotFoundError(f"speaker_bind requiere {required}")

    segments = load_json(segments_path)
    positions = load_json(positions_path)

    if not segments or not positions:
        on_progress("speaker_bind", "Sin segments o positions — binding vacío", 100)
        save_json(map_path, {})
        save_json(zones_path, {})
        return {}

    speakers = sorted({s["speaker"] for s in segments})
    on_progress(
        "speaker_bind",
        f"{len(speakers)} speakers × {len(positions)} posiciones — calibrando con primeros {int(calibration_sec)}s",
        5,
    )

    # Seleccionar segmentos en la ventana de calibración
    cal_segments = [s for s in segments if s["start"] < calibration_sec and s["duration"] >= _MIN_SEGMENT_DUR]
    if not cal_segments:
        # fallback: usar los primeros 10 segmentos de cada speaker
        cal_segments = segments[: len(speakers) * 5]

    # Para cada speaker, picar hasta N segmentos bien distribuidos
    speaker_segs: dict[str, list[dict]] = {s: [] for s in speakers}
    for seg in cal_segments:
        sp = seg["speaker"]
        if sp in speaker_segs and len(speaker_segs[sp]) < _MAX_SEGMENTS_PER_SPEAKER:
            speaker_segs[sp].append(seg)

    video_path = find_video(ep_dir / "input")
    video_info = _probe_video(video_path)
    audio_path = _find_audio(ep_dir)

    on_progress("speaker_bind", "Computando MAD labial por posición en segmentos de calibración", 20)

    # Pre-cargar RMS del audio para filtrar segmentos con voz real
    audio_rms = _compute_audio_rms(audio_path, video_info["duration"], _DELTA_LIP_SEC) if audio_path else None

    score_matrix = np.zeros((len(speakers), len(positions)), dtype=np.float64)
    sample_counts = np.zeros((len(speakers), len(positions)), dtype=np.int64)

    cap = cv2.VideoCapture(str(video_path))
    fps = video_info["fps"]
    frame_w, frame_h = video_info["width"], video_info["height"]
    delta_frames = max(1, int(_DELTA_LIP_SEC * fps))

    last_pct = 20
    for sp_idx, speaker in enumerate(speakers):
        segs = speaker_segs.get(speaker, [])
        if not segs:
            continue
        for seg_i, seg in enumerate(segs):
            # Muestreamos 3 timestamps dentro del segmento
            ts_list = _sample_timestamps(seg["start"], seg["end"], k=3)
            for t in ts_list:
                # Skip si el RMS del audio en ese instante es bajo (evita silencios / ruido)
                if audio_rms is not None:
                    idx = int(t / _DELTA_LIP_SEC)
                    if 0 <= idx < len(audio_rms) and audio_rms[idx] < _AUDIO_SILENCE_RMS:
                        continue

                frame_num = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret1, frame1 = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + delta_frames)
                ret2, frame2 = cap.read()
                if not ret1 or not ret2:
                    continue
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                for pos_idx, pos in enumerate(positions):
                    mad = _mouth_mad(gray1, gray2, pos["cx"], pos["cy"], pos["w"], pos["h"], frame_w, frame_h)
                    score_matrix[sp_idx, pos_idx] += mad
                    sample_counts[sp_idx, pos_idx] += 1

        pct = 20 + int((sp_idx + 1) / max(len(speakers), 1) * 60)
        if pct - last_pct >= 10:
            on_progress("speaker_bind", f"Speaker {speaker} procesado ({sp_idx+1}/{len(speakers)})", pct)
            last_pct = pct

    cap.release()

    # Normalizar: score promedio
    avg = np.zeros_like(score_matrix)
    mask = sample_counts > 0
    avg[mask] = score_matrix[mask] / sample_counts[mask]

    on_progress("speaker_bind", "Aplicando matching Hungarian 1:1", 85)

    assignments = _hungarian(avg, [p["position_id"] for p in positions])

    speaker_face_map: dict = {}
    position_by_id = {p["position_id"]: p for p in positions}

    for sp_idx, speaker in enumerate(speakers):
        pos_id = assignments.get(sp_idx)
        if pos_id is None:
            continue
        pos = position_by_id[pos_id]
        row_score = float(score_matrix[sp_idx].sum())
        selected_score = float(score_matrix[sp_idx, [p["position_id"] for p in positions].index(pos_id)])
        confidence = (selected_score / row_score) if row_score > 0 else 0.0
        speaker_face_map[speaker] = {
            "position_id": pos_id,
            "track_id": pos_id,  # alias para compat
            "cx": pos["cx"], "cy": pos["cy"], "w": pos["w"], "h": pos["h"],
            "centroid": [pos["cx"], pos["cy"]],
            "confidence": round(confidence, 3),
            "source": "auto",
            "needs_manual": confidence < _MIN_CONFIDENCE,
        }

    save_json(map_path, speaker_face_map)

    # Legacy speaker_zones.json — consumido por camera_switcher / exporter
    zones = {
        sp: {"cx": info["cx"], "cy": info["cy"], "w": info["w"], "h": info["h"]}
        for sp, info in speaker_face_map.items()
    }
    save_json(zones_path, zones)

    low_conf = [sp for sp, info in speaker_face_map.items() if info["needs_manual"]]
    if low_conf:
        on_progress(
            "speaker_bind",
            f"⚠ Confianza < {_MIN_CONFIDENCE} para: {', '.join(low_conf)} — recomendable calibración manual",
            95,
        )

    on_progress(
        "speaker_bind",
        f"Binding listo: {len(speaker_face_map)} speakers → posiciones fijas",
        100,
    )
    return speaker_face_map


def override_speaker_binding(
    *,
    project: str,
    episode: str,
    root: Path,
    mapping: dict[str, int],
) -> dict:
    """Overwrite manual del binding desde la UI.

    `mapping`: {"SPEAKER_00": position_id, ...}. Se persiste con source="manual"
    y confidence=1.0 para que el resto del pipeline lo respete.
    """
    ep_dir = get_episode_dir(root, project, episode)
    positions = load_json(ep_dir / "faces" / "face_positions.json")
    position_by_id = {p["position_id"]: p for p in positions}

    out: dict = {}
    for speaker, pos_id in mapping.items():
        pos = position_by_id.get(int(pos_id))
        if not pos:
            continue
        out[speaker] = {
            "position_id": pos["position_id"],
            "track_id": pos["position_id"],
            "cx": pos["cx"], "cy": pos["cy"], "w": pos["w"], "h": pos["h"],
            "centroid": [pos["cx"], pos["cy"]],
            "confidence": 1.0,
            "source": "manual",
            "needs_manual": False,
        }

    diar_dir = ep_dir / "diarization"
    cal_dir = ep_dir / "calibration"
    diar_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    save_json(diar_dir / "speaker_face_map.json", out)
    save_json(cal_dir / "speaker_zones.json", {
        sp: {"cx": i["cx"], "cy": i["cy"], "w": i["w"], "h": i["h"]}
        for sp, i in out.items()
    })
    return out


# ── Helpers ────────────────────────────────────────────────────────

def _sample_timestamps(start: float, end: float, k: int = 3) -> list[float]:
    dur = end - start
    if dur <= 0:
        return []
    if dur < _DELTA_LIP_SEC * 2:
        return [start + dur / 2]
    return [start + dur * (i + 0.5) / k for i in range(k)]


def _mouth_mad(
    gray1: np.ndarray, gray2: np.ndarray,
    cx: float, cy: float, bw: float, bh: float,
    frame_w: int, frame_h: int,
) -> float:
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


def _hungarian(matrix: np.ndarray, position_ids: list[int]) -> dict[int, int]:
    n_sp, n_pos = matrix.shape
    if n_sp == 0 or n_pos == 0:
        return {}
    try:
        from scipy.optimize import linear_sum_assignment
        cost = -matrix.copy()
        if n_sp > n_pos:
            cost = np.hstack([cost, np.zeros((n_sp, n_sp - n_pos))])
        elif n_pos > n_sp:
            cost = np.vstack([cost, np.zeros((n_pos - n_sp, n_pos))])
        row_ind, col_ind = linear_sum_assignment(cost)
        out: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_sp and c < n_pos:
                out[r] = position_ids[c]
        return out
    except ImportError:
        # Greedy fallback
        pairs = []
        for sp in range(n_sp):
            for pos in range(n_pos):
                pairs.append((float(matrix[sp, pos]), sp, pos))
        pairs.sort(key=lambda x: -x[0])
        used_sp: set[int] = set()
        used_pos: set[int] = set()
        out: dict[int, int] = {}
        for score, sp, pos in pairs:
            if score <= 0:
                break
            if sp in used_sp or pos in used_pos:
                continue
            out[sp] = position_ids[pos]
            used_sp.add(sp)
            used_pos.add(pos)
        return out


def _find_audio(ep_dir: Path) -> Path | None:
    for p in (ep_dir / "audio").glob("*.wav"):
        return p
    return None


def _compute_audio_rms(audio_path: Path | None, duration: float, window_sec: float) -> np.ndarray:
    n_windows = max(1, int(duration / window_sec))
    if audio_path is None or not audio_path.exists():
        return np.zeros(n_windows)
    try:
        with wave.open(str(audio_path), "rb") as wf:
            sw, ch, rate = wf.getsampwidth(), wf.getnchannels(), wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except Exception:
        return np.zeros(n_windows)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)
    max_amp = float(2 ** (8 * sw - 1))
    samples /= max_amp + 1e-9
    spw = int(window_sec * rate)
    rms = np.zeros(n_windows)
    for w in range(n_windows):
        chunk = samples[w * spw:(w + 1) * spw]
        if chunk.size:
            rms[w] = float(np.sqrt(np.mean(chunk ** 2)))
    return rms


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


__all__ = ["bind_speakers_to_positions", "override_speaker_binding"]
