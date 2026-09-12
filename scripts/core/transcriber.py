"""Extracción de audio y transcripción con faster-whisper."""

import json
import os
from pathlib import Path

# Evitar que la descarga de HuggingFace se cuelgue indefinidamente (2 min máximo).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir,
    find_video, run_ffmpeg, save_json, seconds_to_srt_time,
)


def transcribe(
    *,
    project: str,
    episode: str,
    lang: str | None = None,
    force: bool = False,
    root: Path,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    settings = load_settings(root)
    cfg = settings["whisper"]

    ep_dir = get_episode_dir(root, project, episode)
    input_dir = ep_dir / "input"
    audio_dir = ep_dir / "audio"
    transcripts_dir = ep_dir / "transcripts"

    for d in (audio_dir, transcripts_dir):
        d.mkdir(parents=True, exist_ok=True)

    srt_glob = list(transcripts_dir.glob("*.srt"))
    if srt_glob and not force:
        on_progress("transcribe", f"Transcripción ya existe: {srt_glob[0].name}", 100)
        return {"skipped": True, "srt_file": srt_glob[0].name}

    video_path = find_video(input_dir)
    audio_path = _extract_audio(video_path, audio_dir, on_progress)

    device, compute_type = _resolve_device(cfg)
    on_progress("transcribe", f"Cargando modelo {cfg['model']} en {device}...", 15)

    from faster_whisper import WhisperModel
    model = WhisperModel(
        cfg["model"],
        device=device,
        compute_type=compute_type,
        download_root=str(root / "models"),
    )

    language = lang or cfg.get("language") or None
    on_progress("transcribe", "Transcribiendo audio...", 20)

    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=cfg.get("word_timestamps", True),
        vad_filter=cfg.get("vad_filter", True),
    )

    total_duration = info.duration if hasattr(info, "duration") and info.duration else None

    segments = []
    for seg in segments_gen:
        segments.append(seg)
        ts = seconds_to_srt_time(seg.end)
        if total_duration and total_duration > 0:
            pct = 20 + (seg.end / total_duration) * 70  # 20% → 90%
        else:
            pct = None
        on_progress("transcribe", f"Transcribiendo... [{ts}]", pct)

    on_progress("transcribe", "Generando archivos de salida...", 90)

    stem = video_path.stem
    srt_content = _build_srt(segments, word_level=cfg.get("word_timestamps", True))
    transcript_content = _build_transcript(segments)

    srt_path = transcripts_dir / f"{stem}.srt"
    txt_path = transcripts_dir / f"{stem}_transcript.txt"
    srt_path.write_text(srt_content, encoding="utf-8")
    txt_path.write_text(transcript_content, encoding="utf-8")

    meta = {
        "audio_file": audio_path.name,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds": round(info.duration, 2) if hasattr(info, "duration") else None,
        "segments_count": len(segments),
        "model": cfg["model"],
        "device": device,
        "srt_file": srt_path.name,
        "transcript_file": txt_path.name,
    }
    save_json(transcripts_dir / "transcription_meta.json", meta)

    on_progress("transcribe", f"Transcripción completada: {len(segments)} segmentos", 100)
    return meta


def _extract_audio(video_path: Path, audio_dir: Path, cb: ProgressCallback) -> Path:
    audio_path = audio_dir / (video_path.stem + ".wav")
    if audio_path.exists():
        cb("transcribe", f"Audio ya existe: {audio_path.name}", 10)
        return audio_path

    cb("transcribe", f"Extrayendo audio: {video_path.name}", 5)
    run_ffmpeg([
        "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", "-vn",
        str(audio_path),
    ])
    cb("transcribe", f"Audio extraído: {audio_path.name}", 10)
    return audio_path


def _resolve_device(cfg: dict) -> tuple[str, str]:
    device_cfg = cfg.get("device", "auto")
    compute_cfg = cfg.get("compute_type", "auto")

    if device_cfg == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = device_cfg

    if compute_cfg == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = compute_cfg

    return device, compute_type


def _build_srt(segments, word_level: bool) -> str:
    lines = []
    idx = 1
    for segment in segments:
        if word_level and hasattr(segment, "words") and segment.words:
            for word in segment.words:
                start = seconds_to_srt_time(word.start)
                end = seconds_to_srt_time(word.end)
                lines.append(f"{idx}\n{start} --> {end}\n{word.word.strip()}\n")
                idx += 1
        else:
            start = seconds_to_srt_time(segment.start)
            end = seconds_to_srt_time(segment.end)
            lines.append(f"{idx}\n{start} --> {end}\n{segment.text.strip()}\n")
            idx += 1
    return "\n".join(lines)


def _build_transcript(segments) -> str:
    lines = []
    for segment in segments:
        ts = f"[{seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}]"
        lines.append(f"{ts} {segment.text.strip()}")
    return "\n".join(lines)
