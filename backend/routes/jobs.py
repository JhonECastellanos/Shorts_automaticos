"""Rutas de jobs: crear, listar, status, cancelar, SSE progress."""

import asyncio
import json
import queue

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models import JobCreate, JobResponse, JobUpdateConfig, ProgressEvent
from ..services.job_manager import (
    create_job, get_job, list_jobs, cancel_job, pause_job, resume_job, stop_job,
    subscribe_progress, unsubscribe_progress,
)

router = APIRouter()


_MAX_PROJECTS = 10


@router.post("", response_model=JobResponse)
async def create(req: JobCreate):
    needs_source = "download" in [s.value if hasattr(s, "value") else s for s in req.steps]
    if needs_source and not req.url and not req.local_file:
        raise HTTPException(400, "Debes especificar url o local_file para el paso de descarga")

    # Límite de 10 proyectos simultáneos. Solo aplica si se está creando un proyecto
    # NUEVO (si el usuario continúa un proyecto existente, no cuenta como creación).
    if needs_source:
        from .videos import _existing_project_names
        existing = _existing_project_names()
        if req.project not in existing and len(existing) >= _MAX_PROJECTS:
            raise HTTPException(
                400,
                f"Límite alcanzado: {_MAX_PROJECTS} proyectos máximo. "
                f"Existen {len(existing)}. Elimina uno antes de crear otro.",
            )

    try:
        job = create_job(req)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return job


@router.get("", response_model=list[JobResponse])
async def list_all():
    return list_jobs()


@router.get("/{job_id}", response_model=JobResponse)
async def get(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    return job


@router.patch("/{job_id}/config", response_model=JobResponse)
async def patch_config(job_id: str, req: JobUpdateConfig):
    from ..services.job_manager import update_job_config
    job = update_job_config(job_id, req.model_dump(exclude_unset=True))
    if not job:
        raise HTTPException(404, "Job no encontrado")
    return job


@router.post("/{job_id}/cancel")
async def cancel(job_id: str):
    if not cancel_job(job_id):
        raise HTTPException(400, "No se puede cancelar este job")
    return {"status": "cancelled"}


@router.post("/{job_id}/pause", response_model=JobResponse)
async def pause(job_id: str):
    job = pause_job(job_id)
    if not job:
        raise HTTPException(400, "No se puede pausar este job")
    return job


@router.post("/{job_id}/resume", response_model=JobResponse)
async def resume(job_id: str):
    job = resume_job(job_id)
    if not job:
        raise HTTPException(400, f"No se puede reanudar el job {job_id[:8]} — estado incompatible o no encontrado")
    return job


@router.post("/{job_id}/stop", response_model=JobResponse)
async def stop(job_id: str):
    job = stop_job(job_id)
    if not job:
        raise HTTPException(400, "No se puede detener este job")
    return job


@router.get("/{job_id}/progress")
async def progress_sse(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")

    # If the job is already in a terminal state, close immediately
    terminal_statuses = {"completed", "failed", "cancelled", "stopped", "awaiting_export"}
    if job.get("status") in terminal_statuses:
        async def done_stream():
            event = ProgressEvent(
                job_id=job_id, step="done",
                message=f"Job {job['status']}", percent=None,
            )
            yield f"data: {json.dumps(event.model_dump())}\n\n"
        return StreamingResponse(done_stream(), media_type="text/event-stream")

    q = queue.Queue()
    subscribe_progress(job_id, q)

    async def event_stream():
        try:
            while True:
                try:
                    event = q.get_nowait()
                    data = json.dumps(event.model_dump())
                    yield f"data: {data}\n\n"
                    if event.step == "done":
                        break
                except queue.Empty:
                    await asyncio.sleep(0.3)
                    # Send heartbeat
                    yield ": heartbeat\n\n"
        finally:
            unsubscribe_progress(job_id, q)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
