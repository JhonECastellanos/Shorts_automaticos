"""Exportación de shorts con ffmpeg: crop dinámico 9:16, subtítulos, tracking."""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .camera_switcher import compute_camera_segments
from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir,
    get_public_episode_dir, sync_public_original,
    find_video, find_ffmpeg_exe, find_ffprobe_exe, save_json, load_json,
    srt_time_to_seconds, seconds_to_srt_time,
)

logger = logging.getLogger(__name__)


def _resolve_encoding_params(settings: dict, export_cfg: dict) -> dict:
    """Resuelve parámetros de encoding según config de processing.

    Retorna dict con: video_codec, crf_or_cq, preset, extra_args.
    """
    processing = settings.get("processing", {})
    device = processing.get("device", "auto")
    encoder = processing.get("ffmpeg_encoder", "auto")
    preset = processing.get("ffmpeg_preset", "auto")

    # Determinar si usar NVENC
    use_nvenc = False
    if encoder == "h264_nvenc":
        use_nvenc = True
    elif encoder == "auto" and device in ("auto", "gpu"):
        use_nvenc = _check_nvenc_available()

    if use_nvenc:
        resolved_preset = preset if preset != "auto" else "p4"
        crf_val = export_cfg.get("crf", 18)
        return {
            "video_codec": "h264_nvenc",
            "quality_args": ["-cq", str(crf_val), "-b:v", "0"],
            "preset_args": ["-preset", resolved_preset],
        }
    else:
        resolved_preset = preset if preset != "auto" else export_cfg.get("preset", "slow")
        return {
            "video_codec": "libx264",
            "quality_args": ["-crf", str(export_cfg.get("crf", 18))],
            "preset_args": ["-preset", resolved_preset],
        }


_nvenc_cache: bool | None = None


def _check_nvenc_available() -> bool:
    """Verifica si h264_nvenc REALMENTE funciona.

    Tres niveles de check:
    1. ¿Está listado en ffmpeg? (soporte compilado)
    2. ¿Existe `nvcuda.dll`/`nvEncodeAPI64.dll` en Windows System32?
       (driver NVIDIA instalado). Sin esta DLL, NVENC siempre falla en runtime.
    3. Smoke test: codifica 1 frame real 1280x720. Frames pequeños no fuerzan
       la inicialización completa del encoder y pueden dar falso-positivo.

    Resultado cacheado en _nvenc_cache.
    """
    global _nvenc_cache
    if _nvenc_cache is not None:
        return _nvenc_cache
    try:
        # (2) Check determinista primero — si no hay DLL, ni preguntamos
        import sys as _sys
        if _sys.platform == "win32":
            import os as _os
            sys32 = _os.path.join(_os.environ.get("WINDIR", r"C:\Windows"), "System32")
            nvcuda = _os.path.join(sys32, "nvcuda.dll")
            nvenc_api = _os.path.join(sys32, "nvEncodeAPI64.dll")
            if not (_os.path.exists(nvcuda) and _os.path.exists(nvenc_api)):
                _nvenc_cache = False
                logger.info("NVENC no disponible (sin nvcuda.dll/nvEncodeAPI64.dll) — usando CPU (libx264)")
                return False

        ffmpeg = find_ffmpeg_exe()

        # (1) Listado
        listed = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_nvenc" not in listed.stdout:
            _nvenc_cache = False
            logger.info("NVENC no disponible (no listado en ffmpeg) — usando CPU (libx264)")
            return False

        # (3) Smoke test realista: 1280x720, 10 frames
        smoke = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=size=1280x720:duration=0.5:rate=30",
                "-c:v", "h264_nvenc", "-preset", "p4", "-frames:v", "10",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=20,
        )
        _nvenc_cache = smoke.returncode == 0
        if _nvenc_cache:
            logger.info("NVENC detectado y funcional — usando GPU para encoding")
        else:
            reason = (smoke.stderr[:300] if smoke.stderr else f"returncode={smoke.returncode}").strip()
            logger.info("NVENC listado pero NO funcional (%s) — fallback a libx264", reason)
    except Exception as exc:
        _nvenc_cache = False
        logger.info("NVENC check falló (%s) — usando CPU (libx264)", exc)
    return _nvenc_cache


def export_shorts(
    *,
    project: str,
    episode: str,
    index: int | None = None,
    no_subtitles: bool = False,
    force: bool = False,
    skip_distribute: bool = False,
    root: Path,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    settings = load_settings(root)
    crop_cfg = settings["crop"]
    export_cfg = settings["export"]
    camera_cfg = settings.get("camera_switching", {})
    enc = _resolve_encoding_params(settings, export_cfg)

    ep_dir = get_episode_dir(root, project, episode)
    input_dir = ep_dir / "input"
    output_dir = ep_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = find_video(input_dir)
    video_info = _get_video_info(video_path)

    # Carpeta de salida pública (OCAMO)
    output_cfg = settings.get("output", {})
    public_root = get_public_episode_dir(root, project, episode, create=True)
    if public_root:
        public_root.mkdir(parents=True, exist_ok=True)

    # Cargar moments (con speaker info si existe, sino los originales)
    diarization_dir = ep_dir / "diarization"
    analysis_dir = ep_dir / "analysis"
    moments_with_speaker = diarization_dir / "moments_with_speaker.json"
    ranked_path = analysis_dir / "moments_ranked.json"
    moments_path = analysis_dir / "moments.json"

    if moments_with_speaker.exists():
        moments = load_json(moments_with_speaker)
    elif ranked_path.exists():
        moments = load_json(ranked_path)
    elif moments_path.exists():
        moments = load_json(moments_path)
    else:
        raise FileNotFoundError("No se encontró moments.json. Ejecuta analyze primero.")

    ordered_moments = [
        {
            **moment,
            "_short_num": int(moment.get("rank", position)),
        }
        for position, moment in enumerate(moments, 1)
    ]
    ordered_moments.sort(key=lambda moment: (moment["_short_num"], moment.get("start", 0)))

    # Cargar speaker zones si existen
    zones_path = ep_dir / "calibration" / "speaker_zones.json"
    speaker_zones = load_json(zones_path) if zones_path.exists() else {}

    # Cargar SRT para subtítulos
    srt_path = None
    if not no_subtitles:
        srt_files = list((ep_dir / "transcripts").glob("*.srt"))
        if srt_files:
            srt_path = srt_files[0]

    # Filtrar por índice si se especifica
    if index is not None:
        selected = next((moment for moment in ordered_moments if moment["_short_num"] == index), None)
        if selected is None:
            raise ValueError(f"Short #{index} no encontrado para exportar")
        ordered_moments = [selected]

    results = []
    for i, moment in enumerate(ordered_moments):
        short_num = int(moment["_short_num"])
        short_dir = output_dir / f"short_{short_num:02d}"
        short_dir.mkdir(exist_ok=True)
        out_file = short_dir / f"short_{short_num:02d}.mp4"

        if out_file.exists() and not force:
            results.append({"file": str(out_file), "score": moment.get("score"), "skipped": True})
            on_progress("export", f"short_{short_num:02d} ya existe, omitido", None)
            continue

        pct = (i / max(len(ordered_moments), 1)) * 90
        on_progress("export", f"Exportando short_{short_num:02d}...", pct)

        # Determinar si usar camera switching multispeaker
        speaker_turns = moment.get("speaker_turns", [])
        use_multispeaker = (
            camera_cfg.get("enabled", False)
            and len(speaker_turns) > 1
            and len(speaker_zones) > 1
        )

        # Generar sub-SRT si hay subtítulos
        sub_srt = None
        if srt_path:
            sub_srt = short_dir / f"short_{short_num:02d}.srt"
            _extract_sub_srt(srt_path, moment["start"], moment["end"], sub_srt)

        if use_multispeaker:
            cam_segments = compute_camera_segments(
                speaker_turns=speaker_turns,
                speaker_zones=speaker_zones,
                start=moment["start"],
                end=moment["end"],
                camera_cfg=camera_cfg,
            )
            _export_short_multispeaker(
                video_path=video_path,
                start=moment["start"],
                end=moment["end"],
                out_file=out_file,
                crop_cfg=crop_cfg,
                export_cfg=export_cfg,
                camera_cfg=camera_cfg,
                cam_segments=cam_segments,
                video_info=video_info,
                sub_srt=sub_srt if not no_subtitles else None,
                enc=enc,
            )
        else:
            # Fallback: crop estático (1 speaker o camera_switching desactivado)
            speaker = moment.get("dominant_speaker")
            zone = speaker_zones.get(speaker, {}) if speaker else {}
            cx = zone.get("center_x", 0.5)
            cy = zone.get("center_y", 0.4)
            _export_short(
                video_path=video_path,
                start=moment["start"],
                end=moment["end"],
                out_file=out_file,
                crop_cfg=crop_cfg,
                export_cfg=export_cfg,
                center_x=cx,
                center_y=cy,
                video_info=video_info,
                sub_srt=sub_srt if not no_subtitles else None,
                enc=enc,
            )

        # Guardar metadata del short
        meta = {
            "short_num": short_num,
            "start": moment["start"],
            "end": moment["end"],
            "duration": round(moment["end"] - moment["start"], 2),
            "score": moment.get("score"),
            "topic": moment.get("topic"),
            "hook": moment.get("hook"),
            "dominant_speaker": moment.get("dominant_speaker"),
        }
        save_json(short_dir / "metadata.json", meta)
        results.append({"file": str(out_file), "score": moment.get("score"), "skipped": False})

        # Copiar short a carpeta pública (OCAMO) — skip si editor-only
        if public_root and not skip_distribute:
            shutil.copy2(str(out_file), str(public_root / f"short_{short_num:02d}.mp4"))

    # Copiar video original a carpeta pública (solo en flujo de distribución)
    if public_root and output_cfg.get("copy_original", True) and not skip_distribute:
        sync_public_original(root, project, episode, video_path, on_progress, "export")

    on_progress("export", f"{len(results)} shorts procesados", 100)
    return results


def distribute_shorts(
    *,
    project: str,
    episode: str,
    root: Path,
    indices: list[int] | None = None,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    """Copia los shorts ya renderizados a la carpeta pública (OCAMO).

    No re-renderiza: lee ``output/short_XX/short_XX.mp4`` y los copia.
    Si ``indices`` se pasa, solo distribuye esos shorts (para export selectivo).
    """
    settings = load_settings(root)
    ep_dir = get_episode_dir(root, project, episode)
    output_dir = ep_dir / "output"
    public_root = get_public_episode_dir(root, project, episode, create=True)
    output_cfg = settings.get("output", {})

    if not public_root:
        on_progress("export", "Ruta pública OCAMO no configurada", 100)
        return []

    public_root.mkdir(parents=True, exist_ok=True)
    short_dirs = sorted(output_dir.glob("short_*"))
    results: list[dict] = []
    total = len(short_dirs)
    for i, sd in enumerate(short_dirs):
        if not sd.is_dir():
            continue
        try:
            short_num = int(sd.name.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if indices and short_num not in indices:
            continue
        mp4 = sd / f"short_{short_num:02d}.mp4"
        if not mp4.exists():
            continue
        dst = public_root / f"short_{short_num:02d}.mp4"
        shutil.copy2(str(mp4), str(dst))
        pct = 5 + int((i / max(total, 1)) * 85)
        on_progress("export", f"Distribuyendo short_{short_num:02d}", pct)
        results.append({"file": str(dst), "short_num": short_num})

    # Sincronizar video original (solo si es export completo, no selectivo)
    if not indices and output_cfg.get("copy_original", True):
        try:
            video_path = find_video(ep_dir / "input")
            sync_public_original(root, project, episode, video_path, on_progress, "export")
        except FileNotFoundError:
            pass

    on_progress("export", f"{len(results)} shorts distribuidos", 100)
    return results


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
    }


def _extract_sub_srt(full_srt: Path, start: float, end: float, out_srt: Path) -> None:
    content = full_srt.read_text(encoding="utf-8")
    blocks = content.strip().split("\n\n")
    extracted = []
    counter = 1

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        time_line = lines[1]
        parts = time_line.split(" --> ")
        if len(parts) != 2:
            continue
        sub_start = srt_time_to_seconds(parts[0])
        sub_end = srt_time_to_seconds(parts[1])

        if sub_end <= start or sub_start >= end:
            continue

        new_start = max(0, sub_start - start)
        new_end = min(end - start, sub_end - start)

        text = "\n".join(lines[2:])
        extracted.append(
            f"{counter}\n"
            f"{seconds_to_srt_time(new_start)} --> {seconds_to_srt_time(new_end)}\n"
            f"{text}"
        )
        counter += 1

    out_srt.write_text("\n\n".join(extracted), encoding="utf-8")


def _export_short(
    *,
    video_path: Path,
    start: float,
    end: float,
    out_file: Path,
    crop_cfg: dict,
    export_cfg: dict,
    center_x: float,
    center_y: float,
    video_info: dict,
    sub_srt: Path | None,
    enc: dict | None = None,
) -> None:
    duration = end - start
    ow = crop_cfg["output_width"]
    oh = crop_cfg["output_height"]
    src_w = video_info["width"]
    src_h = video_info["height"]

    # Calcular crop source: escalar para que el height sea oh
    scale_factor = src_h / oh
    crop_w = int(ow * scale_factor)
    crop_h = src_h

    # Centrar el crop horizontalmente según center_x
    crop_x = int(center_x * src_w - crop_w / 2)
    crop_x = max(0, min(crop_x, src_w - crop_w))

    vf_parts = [f"crop={crop_w}:{crop_h}:{crop_x}:0", f"scale={ow}:{oh}"]

    # Subtítulos
    if sub_srt and sub_srt.exists():
        srt_escaped = str(sub_srt).replace("\\", "/").replace(":", "\\:")
        font = export_cfg.get("subtitle_font", "Arial")
        fontsize = export_cfg.get("subtitle_fontsize", 28)
        color = export_cfg.get("subtitle_color", "white")
        outline = export_cfg.get("subtitle_outline_color", "black")
        outline_w = export_cfg.get("subtitle_outline_width", 2)
        margin_v = export_cfg.get("subtitle_margin_v", 80)

        vf_parts.append(
            f"subtitles='{srt_escaped}':force_style="
            f"'FontName={font},FontSize={fontsize},"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"Outline={outline_w},MarginV={margin_v}'"
        )

    vf = ",".join(vf_parts)

    # Usar encoding params resueltos si están disponibles
    if enc:
        codec_args = ["-c:v", enc["video_codec"]] + enc["quality_args"] + enc["preset_args"]
    else:
        codec_args = ["-c:v", export_cfg["video_codec"], "-crf", str(export_cfg["crf"]), "-preset", export_cfg.get("preset", "slow")]

    cmd = [
        find_ffmpeg_exe(), "-y",
        "-ss", str(max(0, start - 2)), "-i", str(video_path),
        "-ss", str(min(2, start)), "-t", str(duration),
        "-vf", vf,
        *codec_args,
        "-c:a", export_cfg["audio_codec"],
        "-b:a", export_cfg["audio_bitrate"],
        "-movflags", "+faststart",
        str(out_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg export error:\n{result.stderr[-2000:]}")


# ── Multi-speaker export con zoom in/out ─────────────────────────

def _export_short_multispeaker(
    *,
    video_path: Path,
    start: float,
    end: float,
    out_file: Path,
    crop_cfg: dict,
    export_cfg: dict,
    camera_cfg: dict,
    cam_segments: list[dict],
    video_info: dict,
    sub_srt: Path | None,
    enc: dict | None = None,
) -> None:
    """Exporta un short con cambios de cámara por speaker usando zoom in/out."""
    ow = crop_cfg["output_width"]
    oh = crop_cfg["output_height"]
    src_w = video_info["width"]
    src_h = video_info["height"]
    fps = video_info.get("fps", 30.0)
    ffmpeg = find_ffmpeg_exe()

    scale_factor = src_h / oh
    crop_w = int(ow * scale_factor)
    crop_h = src_h

    transition_type = camera_cfg.get("transition_type", "zoom")
    trans_ms = camera_cfg.get("transition_duration_ms", 250 if transition_type == "tiktok_zoom" else 400)
    trans_frames = max(1, int(fps * trans_ms / 1000))
    use_zoom = transition_type in ("zoom", "tiktok_zoom")
    zoom_style = "tiktok" if transition_type == "tiktok_zoom" else "smooth"

    tmp_dir = out_file.parent / "_tmp_segments"
    tmp_dir.mkdir(exist_ok=True)
    segment_files: list[Path] = []

    try:
        for idx, seg in enumerate(cam_segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_dur = seg_end - seg_start
            if seg_dur <= 0:
                continue

            cx = seg["center_x"]
            crop_x = int(cx * src_w - crop_w / 2)
            crop_x = max(0, min(crop_x, src_w - crop_w))

            total_frames = int(seg_dur * fps)
            is_first = idx == 0
            is_last = idx == len(cam_segments) - 1

            # Construir filtro de video
            vf_parts = [f"crop={crop_w}:{crop_h}:{crop_x}:0", f"scale={ow}:{oh}"]

            # Zoom in/out al speaker activo (zoom, tiktok_zoom)
            if use_zoom and len(cam_segments) > 1 and total_frames > trans_frames * 2:
                zoom_expr = _build_zoom_expr(
                    total_frames=total_frames,
                    trans_frames=trans_frames,
                    is_first=is_first,
                    is_last=is_last,
                    style=zoom_style,
                )
                vf_parts.append(
                    f"zoompan=z='{zoom_expr}'"
                    f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                    f":d={total_frames}:s={ow}x{oh}:fps={fps}"
                )
                # zoompan resets duration, remove scale (zoompan outputs at s=)
                vf_parts = [f"crop={crop_w}:{crop_h}:{crop_x}:0", vf_parts[-1]]

            vf = ",".join(vf_parts)

            seg_file = tmp_dir / f"seg_{idx:03d}.mp4"
            if enc:
                codec_args = ["-c:v", enc["video_codec"]] + enc["quality_args"] + enc["preset_args"]
            else:
                codec_args = ["-c:v", export_cfg["video_codec"], "-crf", str(export_cfg["crf"]), "-preset", export_cfg.get("preset", "medium")]
            cmd = [
                ffmpeg, "-y",
                "-ss", str(max(0, seg_start - 2)), "-i", str(video_path),
                "-ss", str(min(2, seg_start)), "-t", str(seg_dur),
                "-vf", vf,
                *codec_args,
                "-c:a", export_cfg["audio_codec"],
                "-b:a", export_cfg["audio_bitrate"],
                str(seg_file),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg segment {idx} error:\n{res.stderr[-2000:]}")
            segment_files.append(seg_file)

        if not segment_files:
            raise RuntimeError("No se generaron segmentos de cámara")

        if len(segment_files) == 1:
            # Un solo segmento → mover directamente
            _concat_or_move_single(segment_files[0], out_file, sub_srt, export_cfg, ffmpeg)
        else:
            _concat_segments(segment_files, out_file, sub_srt, export_cfg, ffmpeg, tmp_dir)

    finally:
        # Limpiar temporales
        for f in segment_files:
            f.unlink(missing_ok=True)
        concat_txt = tmp_dir / "concat.txt"
        concat_txt.unlink(missing_ok=True)
        no_sub = tmp_dir / "no_sub.mp4"
        no_sub.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def _build_zoom_expr(
    total_frames: int,
    trans_frames: int,
    is_first: bool,
    is_last: bool,
    style: str = "smooth",
) -> str:
    """Construye expresión de zoom para zoompan filter.

    Estilos:
      - smooth: 1.0→1.08, lineal, 400ms (sutil)
      - tiktok: 1.0→1.15, ease-out cuadrático, 250ms (agresivo, snappy)

    Comportamiento:
      - Segmento intermedio: zoom in → hold → zoom out
      - Primer segmento: hold → zoom out al final
      - Último segmento: zoom in al inicio → hold
    """
    if style == "tiktok":
        zoom_max = 1.15
    else:
        zoom_max = 1.08
    hold_zoom = 1.0
    delta = zoom_max - hold_zoom

    if is_first and is_last:
        return str(hold_zoom)

    # Funciones de easing para ffmpeg expressions
    # t = progreso normalizado (0..1)
    if style == "tiktok":
        # ease-out cuadrático: f(t) = 1 - (1-t)^2 = 2t - t^2
        def ease_in(t_expr: str) -> str:
            return f"({t_expr}*{t_expr})"  # ease-in: t^2
        def ease_out(t_expr: str) -> str:
            return f"(2*{t_expr}-{t_expr}*{t_expr})"  # ease-out: 2t-t^2
    else:
        # lineal
        def ease_in(t_expr: str) -> str:
            return t_expr
        def ease_out(t_expr: str) -> str:
            return t_expr

    t_in = f"(on/{trans_frames})"       # 0→1 durante zoom in
    t_out = f"((on-{total_frames - trans_frames})/{trans_frames})"  # 0→1 durante zoom out

    zoom_in_expr = f"{hold_zoom}+{delta}*{ease_out(t_in)}"   # 1.0 → max, ease-out (rápido al inicio)
    zoom_out_expr = f"{zoom_max}-{delta}*{ease_in(t_out)}"   # max → 1.0, ease-in (lento al inicio)

    if not is_first and not is_last:
        # zoom in → hold → zoom out
        expr = (
            f"if(lt(on,{trans_frames}),"
            f"{zoom_in_expr},"
            f"if(gt(on,{total_frames - trans_frames}),"
            f"{zoom_out_expr},"
            f"{zoom_max}))"
        )
    elif is_first:
        # hold → zoom out al final
        out_start = total_frames - trans_frames
        expr = (
            f"if(gt(on,{out_start}),"
            f"{zoom_out_expr},"
            f"{zoom_max})"
        )
    else:
        # zoom in al inicio → hold
        expr = (
            f"if(lt(on,{trans_frames}),"
            f"{zoom_in_expr},"
            f"{zoom_max})"
        )

    return expr


def _concat_or_move_single(
    seg_file: Path, out_file: Path,
    sub_srt: Path | None, export_cfg: dict, ffmpeg: str,
) -> None:
    """Un solo segmento: aplicar subtítulos si los hay y mover."""
    if sub_srt and sub_srt.exists():
        _apply_subtitles(seg_file, out_file, sub_srt, export_cfg, ffmpeg)
    else:
        shutil.move(str(seg_file), str(out_file))


def _concat_segments(
    segment_files: list[Path], out_file: Path,
    sub_srt: Path | None, export_cfg: dict, ffmpeg: str,
    tmp_dir: Path,
) -> None:
    """Concatenar múltiples segmentos y aplicar subtítulos."""
    concat_txt = tmp_dir / "concat.txt"
    concat_txt.write_text(
        "\n".join(f"file '{f.name}'" for f in segment_files),
        encoding="utf-8",
    )

    if sub_srt and sub_srt.exists():
        # Concat → temp → subtítulos → final
        no_sub = tmp_dir / "no_sub.mp4"
        cmd_concat = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy", "-movflags", "+faststart",
            str(no_sub),
        ]
        res = subprocess.run(cmd_concat, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg concat error:\n{res.stderr[-2000:]}")
        _apply_subtitles(no_sub, out_file, sub_srt, export_cfg, ffmpeg)
    else:
        cmd_concat = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy", "-movflags", "+faststart",
            str(out_file),
        ]
        res = subprocess.run(cmd_concat, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg concat error:\n{res.stderr[-2000:]}")


def _apply_subtitles(
    input_file: Path, out_file: Path,
    sub_srt: Path, export_cfg: dict, ffmpeg: str,
) -> None:
    """Aplica subtítulos a un video ya exportado."""
    srt_escaped = str(sub_srt).replace("\\", "/").replace(":", "\\:")
    font = export_cfg.get("subtitle_font", "Arial")
    fontsize = export_cfg.get("subtitle_fontsize", 28)
    outline_w = export_cfg.get("subtitle_outline_width", 2)
    margin_v = export_cfg.get("subtitle_margin_v", 80)

    vf = (
        f"subtitles='{srt_escaped}':force_style="
        f"'FontName={font},FontSize={fontsize},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"Outline={outline_w},MarginV={margin_v}'"
    )
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_file),
        "-vf", vf,
        "-c:v", export_cfg.get("video_codec", "libx264"),
        "-crf", str(export_cfg.get("crf", 18)),
        "-preset", export_cfg.get("preset", "medium"),
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitles error:\n{res.stderr[-2000:]}")


def export_preview(
    *,
    project: str,
    episode: str,
    index: int,
    root: Path,
) -> dict:
    """Genera un video borrador rápido (baja calidad, 480p) con subtítulos y camera switching para validar."""
    settings = load_settings(root)
    crop_cfg = settings["crop"]
    export_cfg = settings["export"]
    camera_cfg = settings.get("camera_switching", {})

    ep_dir = get_episode_dir(root, project, episode)
    input_dir = ep_dir / "input"
    output_dir = ep_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = find_video(input_dir)
    video_info = _get_video_info(video_path)

    # Cargar moments
    analysis_dir = ep_dir / "analysis"
    ranked_path = analysis_dir / "moments_ranked.json"
    moments_path = analysis_dir / "moments.json"
    diarization_dir = ep_dir / "diarization"
    moments_with_speaker = diarization_dir / "moments_with_speaker.json"

    if moments_with_speaker.exists():
        moments = load_json(moments_with_speaker)
    elif ranked_path.exists():
        moments = load_json(ranked_path)
    elif moments_path.exists():
        moments = load_json(moments_path)
    else:
        raise FileNotFoundError("No se encontró moments.json.")

    moment = None
    for i, m in enumerate(moments):
        if m.get("rank", i + 1) == index:
            moment = m
            break
    if moment is None:
        raise ValueError(f"Short #{index} no encontrado")

    short_dir = output_dir / f"short_{index:02d}"
    short_dir.mkdir(exist_ok=True)
    out_file = short_dir / f"preview_{index:02d}.mp4"

    start = moment["start"]
    end_t = moment["end"]
    duration = end_t - start

    # Cargar speaker zones y SRT
    zones_path = ep_dir / "calibration" / "speaker_zones.json"
    speaker_zones = load_json(zones_path) if zones_path.exists() else {}

    srt_path = None
    srt_files = list((ep_dir / "transcripts").glob("*.srt"))
    if srt_files:
        srt_path = srt_files[0]

    # Extraer sub-SRT para este short
    sub_srt = None
    if srt_path:
        sub_srt = short_dir / f"preview_{index:02d}.srt"
        _extract_sub_srt(srt_path, start, end_t, sub_srt)

    src_w = video_info["width"]
    src_h = video_info["height"]
    fps = video_info.get("fps", 30.0)

    # Preview dimensions (480p vertical)
    preview_w = 270
    preview_h = 480

    scale_factor = src_h / crop_cfg["output_height"]
    crop_w = int(crop_cfg["output_width"] * scale_factor)
    ffmpeg = find_ffmpeg_exe()

    # Determinar si usar camera switching multispeaker
    speaker_turns = moment.get("speaker_turns", [])
    use_multispeaker = (
        camera_cfg.get("enabled", False)
        and len(speaker_turns) > 1
        and len(speaker_zones) > 1
    )

    if use_multispeaker:
        cam_segments = compute_camera_segments(
            speaker_turns=speaker_turns,
            speaker_zones=speaker_zones,
            start=start,
            end=end_t,
            camera_cfg=camera_cfg,
        )
        _export_preview_multispeaker(
            video_path=video_path,
            start=start,
            end=end_t,
            out_file=out_file,
            crop_w=crop_w,
            src_w=src_w,
            src_h=src_h,
            preview_w=preview_w,
            preview_h=preview_h,
            cam_segments=cam_segments,
            sub_srt=sub_srt,
            export_cfg=export_cfg,
            ffmpeg=ffmpeg,
        )
    else:
        # Static crop (1 speaker)
        speaker = moment.get("dominant_speaker")
        zone = speaker_zones.get(speaker, {}) if speaker else {}
        cx = zone.get("center_x", 0.5)
        crop_x = int(cx * src_w - crop_w / 2)
        crop_x = max(0, min(crop_x, src_w - crop_w))

        vf_parts = [f"crop={crop_w}:{src_h}:{crop_x}:0", f"scale={preview_w}:{preview_h}"]

        if sub_srt and sub_srt.exists():
            srt_escaped = str(sub_srt).replace("\\", "/").replace(":", "\\:")
            font = export_cfg.get("subtitle_font", "Arial")
            fontsize = max(12, export_cfg.get("subtitle_fontsize", 28) // 2)
            outline_w = max(1, export_cfg.get("subtitle_outline_width", 2) // 2)
            margin_v = max(10, export_cfg.get("subtitle_margin_v", 80) // 4)
            vf_parts.append(
                f"subtitles='{srt_escaped}':force_style="
                f"'FontName={font},FontSize={fontsize},"
                f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                f"Outline={outline_w},MarginV={margin_v}'"
            )

        vf = ",".join(vf_parts)

        cmd = [
            ffmpeg, "-y",
            "-ss", str(max(0, start - 2)), "-i", str(video_path),
            "-ss", str(min(2, start)), "-t", str(duration),
            "-vf", vf,
            "-c:v", "libx264",
            "-crf", "35",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-b:a", "64k",
            "-movflags", "+faststart",
            str(out_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg preview error:\n{result.stderr[-2000:]}")

    return {"file": str(out_file), "short_num": index}


def _export_preview_multispeaker(
    *,
    video_path: Path,
    start: float,
    end: float,
    out_file: Path,
    crop_w: int,
    src_w: int,
    src_h: int,
    preview_w: int,
    preview_h: int,
    cam_segments: list[dict],
    sub_srt: Path | None,
    export_cfg: dict,
    ffmpeg: str,
) -> None:
    """Preview multispeaker: cut directo entre speakers (sin zoom, rápido)."""
    tmp_dir = out_file.parent / "_tmp_preview"
    tmp_dir.mkdir(exist_ok=True)
    segment_files: list[Path] = []

    try:
        for idx, seg in enumerate(cam_segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_dur = seg_end - seg_start
            if seg_dur <= 0:
                continue

            cx = seg["center_x"]
            crop_x = int(cx * src_w - crop_w / 2)
            crop_x = max(0, min(crop_x, src_w - crop_w))

            vf = f"crop={crop_w}:{src_h}:{crop_x}:0,scale={preview_w}:{preview_h}"

            seg_file = tmp_dir / f"pseg_{idx:03d}.mp4"
            cmd = [
                ffmpeg, "-y",
                "-ss", str(max(0, seg_start - 2)), "-i", str(video_path),
                "-ss", str(min(2, seg_start)), "-t", str(seg_dur),
                "-vf", vf,
                "-c:v", "libx264",
                "-crf", "35",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "64k",
                str(seg_file),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg preview segment {idx} error:\n{res.stderr[-2000:]}")
            segment_files.append(seg_file)

        if not segment_files:
            raise RuntimeError("No se generaron segmentos de preview")

        if len(segment_files) == 1:
            # Un solo segmento
            if sub_srt and sub_srt.exists():
                _apply_preview_subtitles(segment_files[0], out_file, sub_srt, export_cfg, ffmpeg, preview_w)
            else:
                shutil.move(str(segment_files[0]), str(out_file))
        else:
            # Concatenar y aplicar subs
            concat_txt = tmp_dir / "concat.txt"
            concat_txt.write_text(
                "\n".join(f"file '{f.name}'" for f in segment_files),
                encoding="utf-8",
            )

            if sub_srt and sub_srt.exists():
                no_sub = tmp_dir / "no_sub.mp4"
                cmd_concat = [
                    ffmpeg, "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_txt),
                    "-c", "copy", "-movflags", "+faststart",
                    str(no_sub),
                ]
                res = subprocess.run(cmd_concat, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"ffmpeg preview concat error:\n{res.stderr[-2000:]}")
                _apply_preview_subtitles(no_sub, out_file, sub_srt, export_cfg, ffmpeg, preview_w)
            else:
                cmd_concat = [
                    ffmpeg, "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_txt),
                    "-c", "copy", "-movflags", "+faststart",
                    str(out_file),
                ]
                res = subprocess.run(cmd_concat, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"ffmpeg preview concat error:\n{res.stderr[-2000:]}")

    finally:
        for f in segment_files:
            f.unlink(missing_ok=True)
        for tmp in (tmp_dir / "concat.txt", tmp_dir / "no_sub.mp4"):
            tmp.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def _apply_preview_subtitles(
    input_file: Path, out_file: Path,
    sub_srt: Path, export_cfg: dict, ffmpeg: str,
    preview_w: int,
) -> None:
    """Aplica subtítulos escalados a un preview de baja resolución."""
    srt_escaped = str(sub_srt).replace("\\", "/").replace(":", "\\:")
    font = export_cfg.get("subtitle_font", "Arial")
    fontsize = max(12, export_cfg.get("subtitle_fontsize", 28) // 2)
    outline_w = max(1, export_cfg.get("subtitle_outline_width", 2) // 2)
    margin_v = max(10, export_cfg.get("subtitle_margin_v", 80) // 4)

    vf = (
        f"subtitles='{srt_escaped}':force_style="
        f"'FontName={font},FontSize={fontsize},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"Outline={outline_w},MarginV={margin_v}'"
    )
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_file),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "35",
        "-preset", "ultrafast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg preview subtitles error:\n{res.stderr[-2000:]}")
