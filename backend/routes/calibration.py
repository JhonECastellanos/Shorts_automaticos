"""Endpoints de calibración manual del binding speaker ↔ posición facial.

Cuando `speaker_bind` reporta confianza baja (<0.4) para algún SPEAKER_xx,
el frontend muestra un modal donde el user asocia cada voz con el rostro
correcto. Este endpoint persiste ese mapeo manual.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import validate_path_params

router = APIRouter()

ROOT = Path(__file__).parent.parent.parent


class BindOverride(BaseModel):
    # {"SPEAKER_00": 2, "SPEAKER_01": 0, ...}
    mapping: dict[str, int]


@router.get("/{project}/{episode}/bind-status")
async def bind_status(project: str, episode: str):
    """Devuelve el binding actual + las posiciones detectadas con thumbnails."""
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode

    positions_path = ep_dir / "faces" / "face_positions.json"
    map_path = ep_dir / "diarization" / "speaker_face_map.json"
    segments_path = ep_dir / "diarization" / "speaker_segments.json"

    positions = json.loads(positions_path.read_text(encoding="utf-8")) if positions_path.exists() else []
    face_map = json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}
    segments = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else []

    # Añadir URL pública del thumbnail a cada position
    thumbs_dir = ep_dir / "faces" / "thumbnails"
    for p in positions:
        thumb = thumbs_dir / f"position_{p['position_id']}.jpg"
        p["thumb_url"] = (
            f"/projects/{project}/{episode}/faces/thumbnails/position_{p['position_id']}.jpg"
            if thumb.exists() else None
        )

    # Para cada speaker, encontrar el primer segmento con voz audible (para playback muestra)
    speaker_samples: dict[str, dict] = {}
    for seg in segments:
        sp = seg["speaker"]
        if sp in speaker_samples:
            continue
        if seg.get("duration", 0) >= 1.0:
            speaker_samples[sp] = {
                "start": seg["start"],
                "end": min(seg["start"] + 4.0, seg["end"]),
            }

    speakers = []
    for sp in sorted(speaker_samples.keys()):
        current = face_map.get(sp, {})
        speakers.append({
            "speaker": sp,
            "sample_start": speaker_samples[sp]["start"],
            "sample_end": speaker_samples[sp]["end"],
            "current_position_id": current.get("position_id"),
            "confidence": current.get("confidence"),
            "source": current.get("source"),
            "needs_manual": current.get("needs_manual", False),
        })

    low_conf = [s for s in speakers if s.get("needs_manual")]
    return {
        "positions": positions,
        "speakers": speakers,
        "needs_calibration": bool(low_conf),
    }


@router.post("/{project}/{episode}/bind-override")
async def bind_override(project: str, episode: str, body: BindOverride):
    """Sobrescribe el binding con el mapping manual del user."""
    validate_path_params(project, episode)
    if not body.mapping:
        raise HTTPException(400, "mapping vacío")

    from scripts.core.speaker_bind import override_speaker_binding
    try:
        result = override_speaker_binding(
            project=project, episode=episode,
            root=ROOT, mapping=body.mapping,
        )
    except Exception as e:
        raise HTTPException(500, f"Error aplicando override: {e}")

    return {"status": "ok", "binding": result}
