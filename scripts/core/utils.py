"""Utilidades compartidas entre módulos core."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import yaml

ProgressCallback = Callable[[str, str, float | None], None]


def noop_progress(step: str, msg: str, pct: float | None = None) -> None:
    pass


def load_settings(root: Path) -> dict:
    """Carga settings.yaml y superpone los secretos desde variables de entorno.

    Los secretos nunca deben estar en el YAML versionado; se leen del entorno
    (o del .env) y se inyectan en las secciones correspondientes del dict.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv(root / ".env", override=False)

    data = yaml.safe_load((root / "config" / "settings.yaml").read_text(encoding="utf-8"))

    # Limpieza legacy: en v3 no usamos openroute. Si alguien dejó la key en YAML
    # por una migración vieja, la quitamos aquí para no contaminar la config.
    if "diarization" in data:
        data["diarization"].pop("openroute_api_key", None)

    return data


def get_episode_dir(root: Path, project: str, episode: str, create: bool = True) -> Path:
    ep_dir = root / "projects" / project / episode
    if create:
        for sub in ("input", "audio", "transcripts", "analysis", "diarization", "calibration", "output"):
            (ep_dir / sub).mkdir(parents=True, exist_ok=True)
    return ep_dir


def get_public_root(root: Path) -> Path | None:
    output_cfg = load_settings(root).get("output", {})
    public_root = output_cfg.get("root")
    return Path(public_root) if public_root else None


def get_public_episode_dir(root: Path, project: str, episode: str, create: bool = False) -> Path | None:
    public_root = get_public_root(root)
    if not public_root:
        return None
    public_dir = public_root / project / episode
    if create:
        public_dir.mkdir(parents=True, exist_ok=True)
    return public_dir


def sync_public_original(
    root: Path,
    project: str,
    episode: str,
    video_path: Path,
    on_progress: ProgressCallback = noop_progress,
    step: str = "download",
) -> Path | None:
    public_dir = get_public_episode_dir(root, project, episode, create=True)
    if not public_dir:
        return None

    dest = public_dir / video_path.name
    if not dest.exists() or dest.stat().st_size != video_path.stat().st_size:
        on_progress(step, f"Copiando video original a {public_dir}...", None)
        shutil.copy2(str(video_path), str(dest))
    return dest


def sync_public_transcripts(
    root: Path,
    project: str,
    episode: str,
    transcripts_dir: Path,
    on_progress: ProgressCallback = noop_progress,
    step: str = "transcribe",
) -> None:
    """Copia los archivos de transcripción a la carpeta pública."""
    public_dir = get_public_episode_dir(root, project, episode, create=True)
    if not public_dir:
        return

    # Copiar archivos .srt y .txt
    for pattern in ("*.srt", "*.txt"):
        for transcript_file in transcripts_dir.glob(pattern):
            dest = public_dir / transcript_file.name
            if not dest.exists():
                on_progress(step, f"Copiando transcripción {transcript_file.name} a {public_dir}...", None)
                shutil.copy2(str(transcript_file), str(dest))


def find_video(input_dir: Path) -> Path:
    for ext in ("mp4", "mkv", "mov", "avi", "webm", "m4v"):
        matches = list(input_dir.glob(f"*.{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No se encontró ningún video en {input_dir}")


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).parent)
    winget_base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_base.exists():
        for candidate in winget_base.glob("Gyan.FFmpeg*/**/bin"):
            if (candidate / "ffmpeg.exe").exists():
                return str(candidate)
    return None


def find_ffmpeg_exe() -> str:
    loc = find_ffmpeg()
    if loc:
        exe = Path(loc) / "ffmpeg.exe"
        if exe.exists():
            return str(exe)
        exe = Path(loc) / "ffmpeg"
        if exe.exists():
            return str(exe)
    return "ffmpeg"


def find_ffprobe_exe() -> str:
    loc = find_ffmpeg()
    if loc:
        exe = Path(loc) / "ffprobe.exe"
        if exe.exists():
            return str(exe)
        exe = Path(loc) / "ffprobe"
        if exe.exists():
            return str(exe)
    return "ffprobe"


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = [find_ffmpeg_exe()] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr[-3000:]}")
    return result


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def seconds_to_srt_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_time_to_seconds(t: str) -> float:
    h, m, rest = t.strip().split(":")
    s, ms = rest.replace(",", ".").split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + float(f"0.{ms}")


# ── Helpers de speaker en rangos temporales (compartidos entre validator/ranking/exporter) ─

def dominant_speaker_in_range(segments: list[dict], start: float, end: float) -> str:
    """Retorna el speaker con más tiempo de habla en el rango [start, end]."""
    from collections import defaultdict
    time_per_speaker: dict[str, float] = defaultdict(float)
    for seg in segments:
        overlap_start = max(seg["start"], start)
        overlap_end = min(seg["end"], end)
        if overlap_end > overlap_start:
            time_per_speaker[seg["speaker"]] += overlap_end - overlap_start
    if not time_per_speaker:
        return "UNKNOWN"
    return max(time_per_speaker, key=time_per_speaker.get)


def speaker_turns_in_range(segments: list[dict], start: float, end: float) -> list[dict]:
    """Retorna lista de turnos de speaker en el rango [start, end], mergeando consecutivos."""
    turns = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        turns.append({
            "speaker": seg["speaker"],
            "start": max(seg["start"], start),
            "end": min(seg["end"], end),
        })
    merged: list[dict] = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"]:
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    return merged
