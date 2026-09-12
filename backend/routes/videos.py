"""Rutas de videos: listar proyectos, episodios, info de video."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from ..models import VideoInfo, ProjectListItem, EpisodeItem, EpisodePipelineState
from ..services.job_manager import list_jobs
from . import validate_path_params
from scripts.core.utils import get_public_episode_dir, get_public_root

router = APIRouter()

ROOT = Path(__file__).parent.parent.parent
_VIDEO_EXTS = ("mp4", "mkv", "mov", "avi", "webm", "m4v")
_DEFAULT_STEPS = [
    "download",
    "transcribe",
    "diarize",
    "face_positions",
    "speaker_bind",
    "analyze",
    "edit",
    "rank",
    "export",
]

_STEP_ALIASES = {
    "trend": "analyze",
    "fcpxml": "export",
    "doc_export": "export",
    # Legacy v2 → v3.5
    "detect": "face_positions",
    "calibrate": "speaker_bind",
    "guion": "analyze",
    "validate": "analyze",
    "ranking": "rank",
    # Legacy v3.0-3.4 → v3.5
    "face_track": "face_positions",
    "asd": "speaker_bind",
    "cmia_bind": "speaker_bind",
}


def _parse_iso(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _normalize_step(step: str | None) -> str | None:
    if not isinstance(step, str) or not step:
        return None
    normalized = _STEP_ALIASES.get(step, step)
    return normalized if normalized in _DEFAULT_STEPS else None


def _normalize_steps(steps: list[str] | None, *, fallback_to_default: bool = True) -> list[str]:
    if not isinstance(steps, list):
        return list(_DEFAULT_STEPS) if fallback_to_default else []

    result: list[str] = []
    for step in steps:
        normalized = _normalize_step(step)
        if normalized and normalized not in result:
            result.append(normalized)
    if result:
        return result
    return list(_DEFAULT_STEPS) if fallback_to_default else []


def _has_video(ep_dir: Path) -> bool:
    input_dir = ep_dir / "input"
    if not input_dir.exists():
        return False
    return any(any(input_dir.glob(f"*.{ext}")) for ext in _VIDEO_EXTS)


def _has_public_video(project: str, episode: str) -> bool:
    public_dir = get_public_episode_dir(ROOT, project, episode, create=False)
    if not public_dir or not public_dir.exists():
        return False
    return any(any(public_dir.glob(f"*.{ext}")) for ext in _VIDEO_EXTS)


def _display_project_name(project: str) -> str:
    public_root = get_public_root(ROOT)
    if public_root and public_root.exists():
        for child in public_root.iterdir():
            if child.is_dir() and child.name.lower() == project.lower():
                return child.name
    return project


def _count_shorts(ep_dir: Path) -> int:
    for fname in ("moments_ranked.json", "moments.json"):
        p = ep_dir / "analysis" / fname
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return len(data) if isinstance(data, list) else 0
            except Exception:
                pass
    return 0


def _has_transcript(ep_dir: Path) -> bool:
    return bool(list((ep_dir / "transcripts").glob("*.srt")))


_ACTIVE_JOB_STATUSES = {"pending", "running", "paused", "stopped", "awaiting_export"}


def _live_job(project: str, episode: str) -> dict | None:
    matches = [
        job for job in list_jobs()
        if (job.get("project") or "").lower() == project.lower()
        and (job.get("episode") or "").lower() == episode.lower()
    ]
    if not matches:
        return None
    # Most recently updated first
    matches.sort(key=lambda j: j.get("updated_at") or "", reverse=True)
    newest = matches[0]
    # Only return if the newest job is still active
    if newest.get("status") in _ACTIVE_JOB_STATUSES:
        return newest
    return None


def _has_live_job(project: str, episode: str) -> bool:
    job = _live_job(project, episode)
    return bool(job and job.get("status") in {"pending", "running", "paused"})


def _read_status(ep_dir: Path) -> dict:
    status_path = ep_dir / "pipeline_status.json"
    if not status_path.exists():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _episode_status(project: str, ep_dir: Path) -> str:
    live_job = _live_job(project, ep_dir.name)
    if live_job and isinstance(live_job.get("status"), str):
        return live_job["status"]

    status = _read_status(ep_dir).get("status")
    if isinstance(status, str) and status:
        if status in {"pending", "running", "paused"} and not _has_live_job(project, ep_dir.name):
            return "failed"
        return status

    if _count_shorts(ep_dir) > 0:
        return "completed"
    if _has_transcript(ep_dir):
        return "transcribed"
    return "ready"


def _last_elapsed_seconds(ep_dir: Path) -> float | None:
    elapsed = _read_status(ep_dir).get("elapsed_seconds")
    return float(elapsed) if isinstance(elapsed, (int, float)) else None


def _episode_runtime(project: str, ep_dir: Path) -> tuple[str | None, float | None, str | None, str | None, bool, bool, bool, bool]:
    live_job = _live_job(project, ep_dir.name)
    if live_job:
        status = str(live_job.get("status") or "")
        return (
            _normalize_step(str(live_job.get("current_step"))) if live_job.get("current_step") else None,
            float(live_job["progress"]) if isinstance(live_job.get("progress"), (int, float)) else None,
            str(live_job.get("id")) if live_job.get("id") else None,
            str(live_job.get("updated_at")) if live_job.get("updated_at") else None,
            status in {"pending", "running"},
            status in {"paused", "stopped"},
            status in {"pending", "running", "paused"},
            status in {"pending", "running", "paused"},
        )

    status = _read_status(ep_dir)
    return (
        _normalize_step(str(status.get("current_step"))) if status.get("current_step") else None,
        float(status["progress"]) if isinstance(status.get("progress"), (int, float)) else None,
        None,
        str(status.get("updated_at")) if status.get("updated_at") else None,
        False,
        False,
        False,
        False,
    )


def _infer_steps_completed(status: str, steps: list[str]) -> list[str]:
    if status == "completed":
        return list(steps)
    if status == "transcribed":
        completed = []
        for step in steps:
            completed.append(step)
            if step == "transcribe":
                break
        return completed
    return []


def _episode_pipeline_state(project: str, episode: str, ep_dir: Path) -> EpisodePipelineState:
    live_job = _live_job(project, episode)
    if live_job:
        status = str(live_job.get("status") or "ready")
        steps = _normalize_steps(live_job.get("steps"))
        steps_completed = _normalize_steps(live_job.get("steps_completed"), fallback_to_default=False)
        current_step = _normalize_step(str(live_job.get("current_step"))) if live_job.get("current_step") else None
        progress = float(live_job["progress"]) if isinstance(live_job.get("progress"), (int, float)) else None
        if progress is None and status == "completed":
            progress = 100.0
        return EpisodePipelineState(
            job_id=str(live_job.get("id")) if live_job.get("id") else None,
            project=project,
            episode=episode,
            steps=steps,
            status=status,
            current_step=current_step,
            progress=progress,
            steps_completed=steps_completed,
            error=str(live_job.get("error")) if live_job.get("error") else None,
            created_at=str(live_job.get("created_at")) if live_job.get("created_at") else None,
            updated_at=str(live_job.get("updated_at")) if live_job.get("updated_at") else None,
            started_at=str(live_job.get("started_at")) if live_job.get("started_at") else None,
            finished_at=str(live_job.get("finished_at")) if live_job.get("finished_at") else None,
            elapsed_seconds=float(live_job["elapsed_seconds"]) if isinstance(live_job.get("elapsed_seconds"), (int, float)) else None,
            step_durations={str(key): float(value) for key, value in dict(live_job.get("step_durations") or {}).items()},
            can_pause=status in {"pending", "running"},
            can_resume=status in {"paused", "stopped", "awaiting_export", "failed", "cancelled"},
            can_stop=status in {"pending", "running", "paused"},
            can_cancel=status in {"pending", "running", "paused", "awaiting_export"},
        )

    persisted = _read_status(ep_dir)
    status = _episode_status(project, ep_dir)
    steps = _normalize_steps([str(step) for step in persisted.get("steps", []) if isinstance(step, str)])
    steps_completed_raw = persisted.get("steps_completed")
    steps_completed = _normalize_steps(
        [str(step) for step in steps_completed_raw if isinstance(step, str)],
        fallback_to_default=False,
    ) if isinstance(steps_completed_raw, list) else _infer_steps_completed(status, steps)
    progress = float(persisted["progress"]) if isinstance(persisted.get("progress"), (int, float)) else None
    if progress is None and status == "completed":
        progress = 100.0

    step_durations_raw = persisted.get("step_durations")
    step_durations = {
        str(key): float(value)
        for key, value in dict(step_durations_raw or {}).items()
        if isinstance(value, (int, float))
    }

    persisted_job_id = str(persisted["job_id"]) if persisted.get("job_id") else None

    return EpisodePipelineState(
        job_id=persisted_job_id,
        project=project,
        episode=episode,
        steps=steps,
        status=status,
        current_step=_normalize_step(str(persisted.get("current_step"))) if persisted.get("current_step") else None,
        progress=progress,
        steps_completed=steps_completed,
        error=str(persisted.get("error")) if persisted.get("error") else None,
        created_at=str(persisted.get("created_at")) if persisted.get("created_at") else None,
        updated_at=str(persisted.get("updated_at")) if persisted.get("updated_at") else None,
        started_at=str(persisted.get("started_at")) if persisted.get("started_at") else None,
        finished_at=str(persisted.get("finished_at")) if persisted.get("finished_at") else None,
        elapsed_seconds=float(persisted["elapsed_seconds"]) if isinstance(persisted.get("elapsed_seconds"), (int, float)) else None,
        step_durations=step_durations,
        can_pause=False,
        can_resume=status in {"failed", "cancelled", "stopped", "awaiting_export"},
        can_stop=False,
        can_cancel=False,
    )


def _episode_sort_key(episode: EpisodeItem) -> tuple[int, float, str]:
    status_priority = {
        "running": 0,
        "paused": 1,
        "pending": 2,
        "stopped": 3,
        "failed": 4,
        "completed": 5,
        "transcribed": 6,
        "ready": 7,
        "cancelled": 8,
    }
    return (
        status_priority.get(episode.status, 99),
        -_parse_iso(episode.updated_at),
        episode.name.lower(),
    )


def _project_sort_key(project: ProjectListItem) -> tuple[int, float, str]:
    if not project.episodes:
        return (99, 0.0, project.project.lower())
    best = sorted(project.episodes, key=_episode_sort_key)[0]
    return (_episode_sort_key(best)[0], -_parse_iso(best.updated_at), project.project.lower())


_MAX_PROJECTS = 10


def _existing_project_names() -> list[str]:
    """Lista nombres de carpetas-proyecto que tienen al menos 1 episodio con video."""
    projects_dir = ROOT / "projects"
    if not projects_dir.exists():
        return []
    names: list[str] = []
    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir() or proj_dir.name.startswith("."):
            continue
        # solo contar proyectos "reales" (con episodios con video)
        for ep_dir in proj_dir.iterdir():
            if ep_dir.is_dir() and not ep_dir.name.startswith(".") and _has_video(ep_dir):
                names.append(proj_dir.name)
                break
    return names


@router.get("/projects/limits")
async def project_limits():
    """Devuelve el máximo permitido y cuántos proyectos ya existen."""
    existing = _existing_project_names()
    return {
        "max_projects": _MAX_PROJECTS,
        "current_count": len(existing),
        "remaining_slots": max(0, _MAX_PROJECTS - len(existing)),
        "at_limit": len(existing) >= _MAX_PROJECTS,
    }


@router.get("/projects/suggest-name")
async def suggest_project_name():
    """Sugiere el siguiente nombre numérico disponible (1..10).

    Si ya existen "1", "2", "3", el siguiente será "4". Si hay huecos ("1", "3"),
    sugiere "2" primero. Si se alcanzó el máximo, retorna null + flag at_limit.
    """
    existing_names = set(_existing_project_names())
    for n in range(1, _MAX_PROJECTS + 1):
        if str(n) not in existing_names:
            return {
                "suggested_name": str(n),
                "at_limit": False,
                "current_count": len(existing_names),
                "max_projects": _MAX_PROJECTS,
            }
    return {
        "suggested_name": None,
        "at_limit": True,
        "current_count": len(existing_names),
        "max_projects": _MAX_PROJECTS,
    }


@router.get("/projects", response_model=list[ProjectListItem])
async def list_projects():
    projects_dir = ROOT / "projects"
    if not projects_dir.exists():
        return []

    result = []
    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir() or proj_dir.name.startswith("."):
            continue
        episodes = []
        for ep_dir in sorted(proj_dir.iterdir()):
            if not ep_dir.is_dir() or ep_dir.name.startswith("."):
                continue
            # Incluir episodios en pipeline activo aunque el video todavía no
            # esté completo (p.ej. durante download). Solo filtramos episodios
            # sin video Y sin job asociado (carpeta huérfana).
            pipeline = _episode_pipeline_state(proj_dir.name, ep_dir.name, ep_dir)
            has_video = _has_video(ep_dir)
            has_any_progress = bool(pipeline.job_id) or bool(pipeline.steps_completed)
            if not has_video and not has_any_progress:
                continue
            has_public_video = _has_public_video(proj_dir.name, ep_dir.name)
            episodes.append(EpisodeItem(
                name=ep_dir.name,
                shorts_count=_count_shorts(ep_dir),
                has_transcript=_has_transcript(ep_dir),
                has_public_video=has_public_video,
                status=pipeline.status,
                steps=pipeline.steps,
                steps_completed=pipeline.steps_completed,
                last_elapsed_seconds=pipeline.elapsed_seconds,
                current_step=pipeline.current_step,
                progress=pipeline.progress,
                job_id=pipeline.job_id,
                updated_at=pipeline.updated_at,
                started_at=pipeline.started_at,
                step_durations=pipeline.step_durations,
                can_pause=pipeline.can_pause,
                can_resume=pipeline.can_resume,
                can_stop=pipeline.can_stop,
                can_cancel=pipeline.can_cancel,
            ))
        if episodes:
            result.append(ProjectListItem(project=_display_project_name(proj_dir.name), episodes=sorted(episodes, key=_episode_sort_key)))
    return sorted(result, key=_project_sort_key)


@router.delete("/{project}/{episode}")
async def delete_project(project: str, episode: str):
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    if not ep_dir.exists():
        raise HTTPException(404, "Proyecto no encontrado")

    if _episode_status(project, ep_dir) in {"pending", "running"}:
        raise HTTPException(400, "No se puede eliminar un proyecto en ejecución")

    public_dir = get_public_episode_dir(ROOT, project, episode, create=False)
    shutil.rmtree(ep_dir, ignore_errors=True)

    project_dir = ROOT / "projects" / project
    if project_dir.exists() and not any(project_dir.iterdir()):
        project_dir.rmdir()

    if public_dir and public_dir.exists():
        shutil.rmtree(public_dir, ignore_errors=True)
        public_project_dir = public_dir.parent
        if public_project_dir.exists() and not any(public_project_dir.iterdir()):
            public_project_dir.rmdir()

    return {"status": "deleted"}


@router.get("/{project}/{episode}/pipeline-status", response_model=EpisodePipelineState)
async def get_pipeline_status(project: str, episode: str):
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    if not ep_dir.exists():
        raise HTTPException(404, "Proyecto no encontrado")
    return _episode_pipeline_state(project, episode, ep_dir)


@router.get("/{project}/{episode}", response_model=VideoInfo)
async def get_video_info(project: str, episode: str):
    validate_path_params(project, episode)
    meta_path = ROOT / "projects" / project / episode / "input" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, "Video no encontrado")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return VideoInfo(
        filename=meta.get("filename", ""),
        source=meta.get("source"),
        title=meta.get("title"),
        duration=meta.get("duration"),
        url=meta.get("url"),
    )


@router.api_route("/{project}/{episode}/stream", methods=["GET", "HEAD"])
async def stream_video(project: str, episode: str, request: Request):
    """Sirve el video master con soporte de Range requests para seeking rápido.

    Acepta HEAD para que el frontend pueda probar existencia sin disparar
    errores 404 ruidosos en la consola del browser cuando el video aún se
    está descargando.
    """
    validate_path_params(project, episode)
    input_dir = ROOT / "projects" / project / episode / "input"
    if not input_dir.exists():
        raise HTTPException(404, "Directorio de input no encontrado")

    video_path = None
    for ext in ("mp4", "mkv", "mov", "avi", "webm", "m4v"):
        matches = list(input_dir.glob(f"*.{ext}"))
        if matches:
            video_path = matches[0]
            break
    if not video_path:
        raise HTTPException(404, "No se encontró video en el directorio de input")

    file_size = os.path.getsize(video_path)
    media_type = f"video/{video_path.suffix.lstrip('.')}"

    range_header = request.headers.get("range")
    if range_header:
        # Parse Range: bytes=START-END
        range_spec = range_header.replace("bytes=", "").strip()
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_range():
            chunk_size = 1024 * 1024  # 1MB chunks
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "public, max-age=300",
            },
        )

    # Sin Range header: devolver archivo completo con Accept-Ranges
    return FileResponse(
        str(video_path),
        media_type=media_type,
        filename=video_path.name,
        headers={"Accept-Ranges": "bytes"},
    )
