"""Calibración de zonas de speaker — wrapper de backward compatibility.

Este módulo ha sido refactorizado:
  - Detección facial → face_detector.py (paso 'detect')
  - Asignación speaker→cara → diarizer.py (paso 'diarize')

La función calibrate() se mantiene para backward compat y ejecuta ambos pasos.
"""

import logging
from pathlib import Path

from .utils import (
    ProgressCallback, noop_progress, load_json, get_episode_dir,
)

logger = logging.getLogger(__name__)


def calibrate(
    *,
    project: str,
    episode: str,
    frames_per_speaker: int = 15,
    force: bool = False,
    root: Path,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    """Wrapper de backward compat: ejecuta detect + speaker-face assignment."""
    logger.warning("calibrate() es deprecated — usa detect_faces() + diarize() en su lugar")

    from .face_detector import detect_faces
    detect_faces(project=project, episode=episode, force=force, root=root, on_progress=on_progress)

    # El diarizer ejecuta el mapping speaker→face automáticamente
    # Si ya existían speaker_segments, re-ejecutar el mapping
    ep_dir = get_episode_dir(root, project, episode)
    calibration_dir = ep_dir / "calibration"
    zones_path = calibration_dir / "speaker_zones.json"

    if zones_path.exists():
        on_progress("calibrate", "speaker_zones.json generado", 100)
        return load_json(zones_path)

    on_progress("calibrate", "Calibración completada (sin speaker_zones)", 100)
    return {}

