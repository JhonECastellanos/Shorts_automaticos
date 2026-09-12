"""Cross-Modal Identity Association (CMIA) — paso 'cmia_bind' del pipeline.

Asocia cada ``SPEAKER_xx`` de la diarización con el ``track_id`` del rostro que
articula ese audio. Usa los ASD scores para votar: para cada segmento del
speaker, el track con mayor score promedio en esa ventana "gana" ese voto;
el track_id con más votos gana el binding global. Luego se aplica Hungarian
para optimizar el matching 1:1 global.

Inputs:
    - diarization/speaker_segments.json (de pyannote)
    - faces/face_tracks.json           (de face_tracker)
    - faces/asd_scores.json            (de asd)

Outputs:
    - diarization/speaker_face_map.json  (estructura completa de binding)
    - calibration/speaker_zones.json     (compat con exporter/camera_switcher)

Formato speaker_face_map.json:
    {
      "SPEAKER_00": {
        "track_id": 4,
        "centroid": [0.73, 0.45],
        "confidence": 0.87,
        "votes": 42,
        "total_windows": 48
      },
      ...
    }
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .utils import (
    ProgressCallback, noop_progress, get_episode_dir, save_json, load_json,
)

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = 0.5   # por debajo de esto, emitir warning
_WINDOW_SECONDS = 0.4    # debe coincidir con asd._WINDOW_SECONDS


def bind_speakers_to_tracks(
    *,
    project: str,
    episode: str,
    root: Path,
    force: bool = False,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    ep_dir = get_episode_dir(root, project, episode)
    diar_dir = ep_dir / "diarization"
    faces_dir = ep_dir / "faces"
    cal_dir = ep_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    map_path = diar_dir / "speaker_face_map.json"
    zones_path = cal_dir / "speaker_zones.json"

    if map_path.exists() and zones_path.exists() and not force:
        on_progress("cmia_bind", "speaker_face_map.json ya existe", 100)
        return load_json(map_path)

    segments_path = diar_dir / "speaker_segments.json"
    tracks_path = faces_dir / "face_tracks.json"
    scores_path = faces_dir / "asd_scores.json"

    for required in (segments_path, tracks_path, scores_path):
        if not required.exists():
            raise FileNotFoundError(f"CMIA requiere {required}")

    segments = load_json(segments_path)
    tracks = load_json(tracks_path)
    scores = load_json(scores_path)

    if not segments:
        on_progress("cmia_bind", "No hay segmentos de speaker — omitiendo", 100)
        save_json(map_path, {})
        save_json(zones_path, {})
        return {}
    if not tracks:
        on_progress("cmia_bind", "No hay tracks faciales — omitiendo", 100)
        save_json(map_path, {})
        save_json(zones_path, {})
        return {}

    on_progress("cmia_bind", f"Votando speakers vs {len(tracks)} tracks sobre {len(segments)} segmentos", 10)

    # Preprocesar scores a array [n_tracks × n_windows] indexado por track_id
    scores_by_track = {item["track_id"]: item["windows"] for item in scores}

    # Duración total por speaker (para identificar speakers dominantes)
    speakers = sorted(set(seg["speaker"] for seg in segments))
    track_ids = [tr["track_id"] for tr in tracks]

    # Matriz de votos [n_speakers × n_tracks]: suma de ASD scores en segmentos del speaker
    vote_matrix = np.zeros((len(speakers), len(track_ids)), dtype=np.float64)
    window_counts = np.zeros((len(speakers), len(track_ids)), dtype=np.int64)

    for sp_idx, speaker in enumerate(speakers):
        speaker_segs = [s for s in segments if s["speaker"] == speaker]
        for seg in speaker_segs:
            start, end = seg["start"], seg["end"]
            # Para cada track, acumular scores dentro de [start, end]
            for ti, tid in enumerate(track_ids):
                windows = scores_by_track.get(tid, [])
                for w in windows:
                    if w["t_end"] < start or w["t_start"] > end:
                        continue
                    vote_matrix[sp_idx, ti] += float(w.get("score", 0.0))
                    window_counts[sp_idx, ti] += 1

    on_progress("cmia_bind", "Aplicando Hungarian matching global", 55)

    # Normalizar por número de ventanas (score promedio)
    avg_matrix = np.zeros_like(vote_matrix)
    mask = window_counts > 0
    avg_matrix[mask] = vote_matrix[mask] / window_counts[mask]

    assignments = _hungarian_match(avg_matrix, speakers, track_ids)

    # Construir speaker_face_map
    speaker_face_map: dict = {}
    for sp_idx, speaker in enumerate(speakers):
        tid = assignments.get(sp_idx)
        if tid is None:
            logger.warning("No track assigned for %s", speaker)
            continue
        ti = track_ids.index(tid)
        track_obj = next(t for t in tracks if t["track_id"] == tid)
        total_windows = int(window_counts[sp_idx, ti])
        score_sum = float(vote_matrix[sp_idx, ti])
        # Confidence: qué fracción del score total recibió este track vs el mejor alternativo
        row_sum = float(vote_matrix[sp_idx].sum())
        confidence = (score_sum / row_sum) if row_sum > 0 else 0.0
        speaker_face_map[speaker] = {
            "track_id": tid,
            "centroid": track_obj.get("centroid", [0.5, 0.5]),
            "confidence": round(confidence, 3),
            "votes": round(score_sum, 3),
            "total_windows": total_windows,
        }

    # Warnings si mismatch o confianza baja
    if len(speakers) > len(tracks):
        on_progress(
            "cmia_bind",
            f"⚠️ {len(speakers)} speakers vs {len(tracks)} tracks — algunos speakers compartirán rostro",
            75,
        )
    low_conf = [sp for sp, info in speaker_face_map.items() if info["confidence"] < _MIN_CONFIDENCE]
    if low_conf:
        on_progress(
            "cmia_bind",
            f"⚠️ Binding con confianza < {_MIN_CONFIDENCE}: {', '.join(low_conf)}",
            80,
        )

    save_json(map_path, speaker_face_map)

    # ── Generar speaker_zones.json (backward compat con exporter/camera_switcher) ─
    speaker_zones = _emit_speaker_zones(speaker_face_map, tracks)
    save_json(zones_path, speaker_zones)

    on_progress(
        "cmia_bind",
        f"Binding completado: {len(speaker_face_map)} speakers asociados a rostros",
        100,
    )
    return speaker_face_map


def _hungarian_match(
    avg_matrix: np.ndarray,
    speakers: list[str],
    track_ids: list[int],
) -> dict[int, int]:
    """Devuelve {speaker_idx: track_id} maximizando score global."""
    n_sp, n_tr = avg_matrix.shape
    if n_sp == 0 or n_tr == 0:
        return {}

    try:
        from scipy.optimize import linear_sum_assignment
        # linear_sum_assignment minimiza — invertimos signo
        cost = -avg_matrix.copy()
        # Pad si hay más speakers que tracks (algunos speakers se quedan sin track)
        if n_sp > n_tr:
            pad = np.zeros((n_sp, n_sp - n_tr))
            cost = np.hstack([cost, pad])
        elif n_tr > n_sp:
            pad = np.zeros((n_tr - n_sp, n_tr))
            cost = np.vstack([cost, pad])

        row_ind, col_ind = linear_sum_assignment(cost)
        assignments: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_sp and c < n_tr:
                assignments[r] = track_ids[c]
        return assignments
    except ImportError:
        # Fallback greedy
        logger.info("scipy no disponible, CMIA greedy")
        return _greedy_match(avg_matrix, track_ids)


def _greedy_match(avg_matrix: np.ndarray, track_ids: list[int]) -> dict[int, int]:
    n_sp, n_tr = avg_matrix.shape
    pairs = []
    for sp in range(n_sp):
        for tr in range(n_tr):
            pairs.append((float(avg_matrix[sp, tr]), sp, tr))
    pairs.sort(key=lambda x: -x[0])
    assigned_sp, assigned_tr = set(), set()
    out: dict[int, int] = {}
    for score, sp, tr in pairs:
        if score <= 0:
            break
        if sp in assigned_sp or tr in assigned_tr:
            continue
        out[sp] = track_ids[tr]
        assigned_sp.add(sp)
        assigned_tr.add(tr)
    return out


def _emit_speaker_zones(speaker_face_map: dict, tracks: list[dict]) -> dict:
    """Emite speaker_zones.json compatible con el exporter/camera_switcher existente.

    Formato histórico: {"SPEAKER_00": {"cx": 0.73, "cy": 0.45, "w": 0.15, "h": 0.22}, ...}
    """
    track_by_id = {tr["track_id"]: tr for tr in tracks}
    zones: dict = {}
    for speaker, info in speaker_face_map.items():
        tr = track_by_id.get(info["track_id"])
        if not tr:
            continue
        # Usar el bbox medio del track para w, h
        bboxes = [f["bbox"] for f in tr.get("frames", [])]
        if bboxes:
            cx = float(np.mean([b[0] for b in bboxes]))
            cy = float(np.mean([b[1] for b in bboxes]))
            w = float(np.mean([b[2] for b in bboxes]))
            h = float(np.mean([b[3] for b in bboxes]))
        else:
            cx, cy = info.get("centroid", [0.5, 0.5])
            w, h = 0.15, 0.22
        zones[speaker] = {"cx": cx, "cy": cy, "w": w, "h": h}
    return zones
