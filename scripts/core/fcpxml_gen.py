"""Generación de FCPXML para importar timeline en DaVinci Resolve / FCP."""

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir,
    find_video, find_ffprobe_exe, load_json, save_json,
)


def generate_fcpxml(
    *,
    project: str,
    episode: str,
    fps: float | None = None,
    root: Path,
    on_progress: ProgressCallback = noop_progress,
) -> str:
    settings = load_settings(root)
    ep_dir = get_episode_dir(root, project, episode)
    analysis_dir = ep_dir / "analysis"

    ranked_path = analysis_dir / "moments_ranked.json"
    moments_path = analysis_dir / "moments.json"
    if ranked_path.exists():
        moments = load_json(ranked_path)
    elif moments_path.exists():
        moments = load_json(moments_path)
    else:
        raise FileNotFoundError(f"No se encontró moments.json. Ejecuta analyze primero.")
    moments = sorted(
        moments,
        key=lambda moment: (int(moment.get("rank", 10_000)), float(moment.get("start", 0))),
    )

    video_path = find_video(ep_dir / "input")

    if fps is None:
        video_info = _get_video_info(video_path)
        fps = video_info["fps"]

    on_progress("fcpxml", f"Generando FCPXML ({len(moments)} clips, {fps:.2f} fps)", 10)

    # Construir el FCPXML
    fcpxml = _build_fcpxml(
        project_name=f"{project}_{episode}",
        video_path=video_path,
        moments=moments,
        fps=fps,
    )

    out_path = ep_dir / "output" / f"{project}_{episode}_timeline.fcpxml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fcpxml, encoding="utf-8")

    on_progress("fcpxml", f"FCPXML guardado: {out_path.name}", 100)
    return str(out_path)


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
    return {"fps": fps, "width": int(stream["width"]), "height": int(stream["height"])}


def _build_fcpxml(
    *,
    project_name: str,
    video_path: Path,
    moments: list[dict],
    fps: float,
) -> str:
    # Calcular frame duration rational
    fps_num, fps_den = _fps_to_rational(fps)
    frame_dur = f"{fps_den}/{fps_num}s"

    root = ET.Element("fcpxml", version="1.10")
    resources = ET.SubElement(root, "resources")

    # Format resource
    fmt_id = "r1"
    ET.SubElement(resources, "format", id=fmt_id, name=f"FFVideoFormat1080p{fps:.0f}",
                  frameDuration=frame_dur, width="1920", height="1080")

    # Asset resource
    asset_id = "r2"
    asset = ET.SubElement(resources, "asset", id=asset_id, name=video_path.stem,
                          src=f"file://{video_path.as_posix()}", format=fmt_id,
                          hasVideo="1", hasAudio="1")

    # Library → Event → Project → Sequence
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=project_name)
    proj = ET.SubElement(event, "project", name=project_name)

    # Calcular duración total de la secuencia
    total_dur = sum(m["end"] - m["start"] for m in moments)
    total_frames = int(total_dur * fps)
    seq_dur = f"{total_frames * fps_den}/{fps_num}s"

    sequence = ET.SubElement(proj, "sequence", format=fmt_id, duration=seq_dur,
                             tcStart="0/1s", tcFormat="NDF")
    spine = ET.SubElement(sequence, "spine")

    # Añadir clips al spine
    timeline_offset = 0
    for i, moment in enumerate(moments):
        duration = moment["end"] - moment["start"]
        dur_frames = int(duration * fps)
        clip_dur = f"{dur_frames * fps_den}/{fps_num}s"

        start_frames = int(moment["start"] * fps)
        clip_start = f"{start_frames * fps_den}/{fps_num}s"

        offset_frames = int(timeline_offset * fps)
        clip_offset = f"{offset_frames * fps_den}/{fps_num}s"

        clip = ET.SubElement(spine, "asset-clip",
                             name=moment.get("topic", f"Short {i + 1}"),
                             ref=asset_id,
                             offset=clip_offset,
                             start=clip_start,
                             duration=clip_dur,
                             format=fmt_id,
                             tcFormat="NDF")

        # Añadir marker con info del short
        note_text = (
            f"Score: {moment.get('score', '?')} | "
            f"Hook: {moment.get('hook', '')} | "
            f"{moment.get('reason', '')}"
        )
        marker = ET.SubElement(clip, "marker", start=clip_start, duration=frame_dur,
                               value=note_text)
        timeline_offset += duration

    # Formatear XML con indentación
    ET.indent(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
    return xml_str


def _fps_to_rational(fps: float) -> tuple[int, int]:
    common = {
        23.976: (24000, 1001),
        24.0: (24, 1),
        25.0: (25, 1),
        29.97: (30000, 1001),
        30.0: (30, 1),
        50.0: (50, 1),
        59.94: (60000, 1001),
        60.0: (60, 1),
    }
    for rate, rational in common.items():
        if abs(fps - rate) < 0.1:
            return rational
    return (int(round(fps)), 1)
