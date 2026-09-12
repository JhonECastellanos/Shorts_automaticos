"""Descarga de videos con yt-dlp o registro de archivos locales."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp

from .utils import (
    ProgressCallback, noop_progress, get_episode_dir, find_ffmpeg, save_json, load_json,
    sync_public_original,
)


def download(
    *,
    project: str,
    episode: str,
    url: str | None = None,
    local_file: str | None = None,
    root: Path,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    ep_dir = get_episode_dir(root, project, episode)
    input_dir = ep_dir / "input"

    if local_file:
        return _register_local(Path(local_file), input_dir, ep_dir, on_progress)
    if url:
        return _download_url(url, input_dir, ep_dir, on_progress)
    raise ValueError("Debes especificar url o local_file")


def _register_local(src: Path, input_dir: Path, ep_dir: Path, cb: ProgressCallback) -> dict:
    if not src.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {src}")
    size_mb = src.stat().st_size / (1024 * 1024)
    cb("download", f"Registrando archivo local: {src.name} ({size_mb:.1f} MB)", 0)
    dest = input_dir / src.name
    if dest.resolve() != src.resolve():
        cb("download", f"Copiando archivo: {src.name}", 20)
        shutil.copy2(str(src), str(dest))
        cb("download", f"Archivo copiado: {dest.name}", 50)
    else:
        cb("download", f"Archivo ya en destino: {dest.name}", 50)
    meta = {
        "source": "local",
        "original_path": str(src),
        "filename": dest.name,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_metadata(ep_dir, meta)
    cb("download", "Sincronizando original público...", 70)
    sync_public_original(ep_dir.parent.parent.parent, ep_dir.parent.name, ep_dir.name, dest, cb, "download")
    cb("download", f"Archivo registrado: {dest.name}", 100)
    return meta


def _find_cookies_file() -> Path | None:
    """Busca un archivo cookies.txt en ubicaciones estándar del proyecto."""
    candidates = [
        Path(__file__).parent.parent.parent / "config" / "youtube_cookies.txt",
        Path(__file__).parent.parent.parent / "config" / "cookies.txt",
        Path(__file__).parent.parent.parent / "youtube_cookies.txt",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _download_url(url: str, input_dir: Path, ep_dir: Path, cb: ProgressCallback) -> dict:
    ffmpeg_loc = find_ffmpeg()
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(input_dir / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "writethumbnail": False,
        "quiet": True,
        "no_warnings": True,
        "writeinfojson": False,
        # yt-dlp 2026+ requiere JS runtime + EJS solver para YouTube
        "js_runtimes": {"node": {}},
        "remote_components": "ejs:github",
    }
    if ffmpeg_loc:
        ydl_opts["ffmpeg_location"] = ffmpeg_loc

    # Cookies: archivo explícito si existe (opcional, no requerido)
    cookies_file = _find_cookies_file()
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)
        cb("download", f"Usando cookies: {cookies_file.name}", 5)

    cb("download", f"Descargando: {url}", 10)

    def _hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = 10 + (downloaded / total) * 80
                cb("download", f"Descargando... {downloaded // (1024*1024)} MB / {total // (1024*1024)} MB", pct)

    ydl_opts["progress_hooks"] = [_hook]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info and "entries" in info:
            info = info["entries"][0]

    candidates = list(input_dir.glob("*.mp4"))
    if not candidates:
        candidates = list(input_dir.glob(f"*.{info.get('ext', 'mp4')}"))
    video_path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    if not video_path:
        raise RuntimeError(f"No se encontró el archivo descargado para: {url}")

    cb("download", f"Merge ffmpeg completado: {video_path.name}", 95)

    meta = {
        "source": "yt-dlp",
        "url": url,
        "title": info.get("title"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "filename": video_path.name,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_metadata(ep_dir, meta)
    sync_public_original(ep_dir.parent.parent.parent, ep_dir.parent.name, ep_dir.name, video_path, cb, "download")
    cb("download", f"Descarga completada: {video_path.name}", 100)
    return meta


def _save_metadata(ep_dir: Path, data: dict) -> None:
    meta_path = ep_dir / "input" / "metadata.json"
    existing = {}
    if meta_path.exists():
        existing = load_json(meta_path)
    existing.update(data)
    save_json(meta_path, existing)
