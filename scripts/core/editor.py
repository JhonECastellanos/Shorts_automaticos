"""Edición de borradores (drafts) de shorts — paso 'edit' del pipeline.

Filosofía v3.4:
- Edit genera BORRADORES rápidos de baja resolución (540x960, CRF 30, preset ultrafast).
  Sirven para que el usuario vea el resultado aproximado con cortes de cámara y
  ajuste rangos si hace falta ANTES de renderizar el final.
- Export (paso manual disparado desde UI) genera los shorts FINALES en alta
  resolución (1080x1920, CRF 18, preset slow) y los distribuye a OCAMO.

**Clave de rendimiento**: cada borrador se renderiza en UNA sola llamada ffmpeg
usando `filter_complex` que encadena todos los cortes de cámara inline. Antes
generábamos N segmentos con N ffmpeg y después concat — 10-20× más lento.

Con el approach nuevo, un borrador de 90s con 4 cambios de cámara se renderiza
en ~5-8 segundos en CPU (libx264 ultrafast).

Borradores en `output/drafts/short_XX.mp4`, finales en `output/short_XX/short_XX.mp4`.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

from .camera_switcher import compute_camera_segments
from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir,
    save_json, load_json, find_video, find_ffmpeg_exe, find_ffprobe_exe,
    dominant_speaker_in_range, speaker_turns_in_range,
)

logger = logging.getLogger(__name__)

# Parámetros de borrador (low-res, fast encoder)
_DRAFT_WIDTH = 540
_DRAFT_HEIGHT = 960
_DRAFT_CRF = 30
_DRAFT_PRESET = "ultrafast"


def edit_shorts(
    *,
    project: str,
    episode: str,
    root: Path,
    force: bool = False,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    ep_dir = get_episode_dir(root, project, episode)
    analysis_dir = ep_dir / "analysis"
    diar_dir = ep_dir / "diarization"
    output_dir = ep_dir / "output"
    drafts_dir = output_dir / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)

    moments_path = analysis_dir / "moments.json"
    if not moments_path.exists():
        raise FileNotFoundError(f"No hay moments.json en {analysis_dir}")

    moments = load_json(moments_path)
    if not moments:
        on_progress("edit", "moments.json vacío — nada que editar", 100)
        return []

    segments = load_json(diar_dir / "speaker_segments.json") if (diar_dir / "speaker_segments.json").exists() else []

    on_progress("edit", f"Enriqueciendo {len(moments)} moments con speaker turns", 3)
    enriched = []
    for m in moments:
        dominant = dominant_speaker_in_range(segments, m["start"], m["end"]) if segments else "UNKNOWN"
        turns = speaker_turns_in_range(segments, m["start"], m["end"]) if segments else []
        enriched.append({**m, "dominant_speaker": dominant, "speaker_turns": turns})

    diar_dir.mkdir(parents=True, exist_ok=True)
    save_json(diar_dir / "moments_with_speaker.json", enriched)

    settings = load_settings(root)
    camera_cfg = settings.get("camera_switching", {})
    zones_path = ep_dir / "calibration" / "speaker_zones.json"
    speaker_zones = load_json(zones_path) if zones_path.exists() else {}

    video_path = find_video(ep_dir / "input")
    video_info = _probe_video(video_path)

    total = len(enriched)
    results: list[dict] = []
    for i, moment in enumerate(enriched):
        short_num = int(moment.get("rank", i + 1))
        out_file = drafts_dir / f"short_{short_num:02d}.mp4"

        if out_file.exists() and not force:
            results.append({"file": str(out_file), "skipped": True, "short_num": short_num})
            on_progress("edit", f"Borrador short_{short_num:02d} ya existe", 3 + int(((i + 1) / total) * 92))
            continue

        pct = 3 + int((i / max(total, 1)) * 92)
        on_progress("edit", f"Renderizando borrador short_{short_num:02d} ({i+1}/{total})", pct)

        speaker_turns = moment.get("speaker_turns", [])
        use_multispeaker = (
            camera_cfg.get("enabled", False)
            and len(speaker_turns) > 1
            and len(speaker_zones) > 1
        )

        try:
            if use_multispeaker:
                cam_segments = compute_camera_segments(
                    speaker_turns=speaker_turns,
                    speaker_zones=speaker_zones,
                    start=moment["start"],
                    end=moment["end"],
                    camera_cfg=camera_cfg,
                )
                # Colapsar segmentos muy cortos (< 0.3s) al anterior para reducir complejidad
                cam_segments = _merge_tiny_segments(cam_segments, min_dur=0.3)
                if len(cam_segments) <= 1:
                    cx = cam_segments[0]["center_x"] if cam_segments else 0.5
                    _render_draft_single_crop(
                        video_path=video_path,
                        start=moment["start"], end=moment["end"],
                        out_file=out_file, center_x=cx, video_info=video_info,
                    )
                else:
                    _render_draft_filtercomplex(
                        video_path=video_path,
                        moment_start=moment["start"], moment_end=moment["end"],
                        cam_segments=cam_segments,
                        out_file=out_file, video_info=video_info,
                    )
            else:
                speaker = moment.get("dominant_speaker")
                zone = speaker_zones.get(speaker, {}) if speaker else {}
                cx = zone.get("cx", 0.5)
                _render_draft_single_crop(
                    video_path=video_path,
                    start=moment["start"], end=moment["end"],
                    out_file=out_file, center_x=cx, video_info=video_info,
                )
        except Exception as exc:
            logger.warning("Draft %d falló: %s", short_num, exc)
            on_progress("edit", f"⚠ Borrador short_{short_num:02d} falló: {str(exc)[:150]}", pct)
            continue

        meta = {
            "short_num": short_num,
            "start": moment["start"], "end": moment["end"],
            "duration": round(moment["end"] - moment["start"], 2),
            "score": moment.get("score"), "topic": moment.get("topic"),
            "hook": moment.get("hook"), "dominant_speaker": moment.get("dominant_speaker"),
            "is_draft": True,
        }
        save_json(drafts_dir / f"short_{short_num:02d}.json", meta)
        results.append({"file": str(out_file), "skipped": False, "short_num": short_num})

    on_progress("edit", f"{len(results)} borradores low-res generados en output/drafts/", 100)
    return results


def _merge_tiny_segments(cam_segments: list[dict], min_dur: float = 0.3) -> list[dict]:
    """Funde segmentos < min_dur con el anterior para no fragmentar el render."""
    if not cam_segments:
        return []
    out = [dict(cam_segments[0])]
    for seg in cam_segments[1:]:
        dur = seg["end"] - seg["start"]
        if dur < min_dur:
            out[-1]["end"] = seg["end"]
        else:
            out.append(dict(seg))
    return out


def _probe_video(video_path: Path) -> dict:
    import json
    cmd = [
        find_ffprobe_exe(), "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    fps_parts = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and float(fps_parts[1]) > 0 else 30.0
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
    }


_SEEK_BUFFER = 2.0  # segundos antes del start para absorber offset de fast-seek al keyframe


def _render_draft_single_crop(
    *, video_path: Path, start: float, end: float, out_file: Path,
    center_x: float, video_info: dict,
) -> None:
    """Renderiza borrador con crop estático. UN solo ffmpeg.

    Fix sync: fast-seek cerca del start (con buffer de 2s) y slow-seek exacto
    dentro del buffer. Así video + audio caen en el mismo instante.
    """
    src_w = video_info["width"]
    src_h = video_info["height"]
    scale_factor = src_h / _DRAFT_HEIGHT
    crop_w = int(_DRAFT_WIDTH * scale_factor)
    crop_x = int(center_x * src_w - crop_w / 2)
    crop_x = max(0, min(crop_x, src_w - crop_w))

    duration = end - start
    vf = f"crop={crop_w}:{src_h}:{crop_x}:0,scale={_DRAFT_WIDTH}:{_DRAFT_HEIGHT}"

    fast_seek = max(0, start - _SEEK_BUFFER)
    inner_seek = start - fast_seek

    cmd = [
        find_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(fast_seek), "-i", str(video_path),
        "-ss", str(inner_seek),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", _DRAFT_PRESET, "-crf", str(_DRAFT_CRF),
        "-c:a", "aac", "-b:a", "96k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg draft static err: {res.stderr[-400:]}")


def _render_draft_filtercomplex(
    *, video_path: Path, moment_start: float, moment_end: float,
    cam_segments: list[dict], out_file: Path, video_info: dict,
) -> None:
    """Renderiza borrador con cambios de cámara en UN solo ffmpeg + filter_complex.

    Fix micro-retroceso: el **audio fluye continuo** (un único `atrim` global),
    y solo el **video** se trocea por segmentos de cámara (`trim+crop+concat`).
    Antes tanto video como audio iban por trim+concat, lo que introducía un
    micro-gap perceptible como retroceso en cada cambio de cámara porque el
    concat del audio no queda perfectamente alineado muestra-a-muestra.

    Además usa fast-seek con buffer + rangos absolutos post-seek para precisión.
    """
    src_w = video_info["width"]
    src_h = video_info["height"]
    scale_factor = src_h / _DRAFT_HEIGHT
    crop_w = int(_DRAFT_WIDTH * scale_factor)

    duration = moment_end - moment_start
    fast_seek = max(0, moment_start - _SEEK_BUFFER)
    inner_offset = moment_start - fast_seek

    n = len(cam_segments)
    parts: list[str] = []
    # Video: split en N, trim+crop+scale por segmento, concat.
    parts.append(f"[0:v]split={n}" + "".join(f"[v{i}]" for i in range(n)))

    concat_inputs: list[str] = []
    for i, seg in enumerate(cam_segments):
        rel_start = max(0.0, seg["start"] - moment_start)
        rel_end = min(duration, seg["end"] - moment_start)
        if rel_end <= rel_start:
            continue
        abs_start = rel_start + inner_offset
        abs_end = rel_end + inner_offset

        cx = seg["center_x"]
        crop_x = int(cx * src_w - crop_w / 2)
        crop_x = max(0, min(crop_x, src_w - crop_w))

        parts.append(
            f"[v{i}]trim=start={abs_start:.3f}:end={abs_end:.3f},setpts=PTS-STARTPTS,"
            f"crop={crop_w}:{src_h}:{crop_x}:0,scale={_DRAFT_WIDTH}:{_DRAFT_HEIGHT}"
            f"[vc{i}]"
        )
        concat_inputs.append(f"[vc{i}]")

    if not concat_inputs:
        raise RuntimeError("No hay segmentos válidos para el borrador")

    n_valid = len(concat_inputs)
    if n_valid == 1:
        # Un solo segmento: copy-through el stream, sin concat (evita overhead).
        parts.append(f"{concat_inputs[0]}null[vout]")
    else:
        parts.append(f"{''.join(concat_inputs)}concat=n={n_valid}:v=1:a=0[vout]")

    # Audio: un único atrim global del rango completo. Continuo, sin cortes.
    parts.append(
        f"[0:a]atrim=start={inner_offset:.3f}:end={inner_offset + duration:.3f},"
        f"asetpts=PTS-STARTPTS[aout]"
    )

    filter_complex = ";".join(parts)

    cmd = [
        find_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(fast_seek), "-i", str(video_path),
        "-t", str(duration + _SEEK_BUFFER * 2),  # buffer al final para el atrim
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", _DRAFT_PRESET, "-crf", str(_DRAFT_CRF),
        "-c:a", "aac", "-b:a", "96k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg draft filtercomplex err: {res.stderr[-500:]}")


__all__ = ["edit_shorts"]
