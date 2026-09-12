"""Repositorio de jobs: operaciones CRUD sobre la base de datos.

Todas las funciones reciben una Session como primer argumento y son
completamente agnósticas de cómo se gestiona el ciclo de vida de la sesión.
El caller (normalmente get_db()) es responsable del commit/rollback.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Job, JobEvent, ShortSnapshot


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Jobs ─────────────────────────────────────────────────────────

def create_job(db: Session, job_data: dict) -> Job:
    """Persiste un job nuevo en la BD."""
    job = Job(
        id=job_data["id"],
        project=job_data["project"],
        episode=job_data["episode"],
        url=job_data.get("url"),
        local_file=job_data.get("local_file"),
        ai_model=job_data.get("ai_model"),
        diarization_provider=job_data.get("diarization_provider"),
        min_duration=job_data.get("min_duration"),
        max_duration=job_data.get("max_duration"),
        steps=list(job_data.get("steps", [])),
        status=job_data.get("status", "pending"),
        current_step=job_data.get("current_step"),
        progress=job_data.get("progress"),
        steps_completed=list(job_data.get("steps_completed", [])),
        error=job_data.get("error"),
        step_durations=dict(job_data.get("step_durations", {})),
        elapsed_seconds=job_data.get("elapsed_seconds"),
        created_at=_now(),
        updated_at=_now(),
        started_at=None,
        finished_at=None,
    )
    db.add(job)
    return job


def get_job(db: Session, job_id: str) -> Optional[Job]:
    return db.get(Job, job_id)


def list_jobs(db: Session) -> list[Job]:
    return list(db.execute(select(Job)).scalars().all())


def update_job_fields(db: Session, job_id: str, fields: dict) -> Optional[Job]:
    """Actualiza campos arbitrarios de un job. Agrega `updated_at` automáticamente."""
    job = db.get(Job, job_id)
    if not job:
        return None
    fields_with_ts = {**fields, "updated_at": _now()}
    for key, value in fields_with_ts.items():
        if hasattr(job, key):
            setattr(job, key, value)
    return job


def get_jobs_by_project_episode(db: Session, project: str, episode: str) -> list[Job]:
    stmt = select(Job).where(Job.project == project, Job.episode == episode)
    return list(db.execute(stmt).scalars().all())


def get_active_jobs(db: Session) -> list[Job]:
    """Devuelve los jobs en estado activo (pending / running / paused)."""
    stmt = select(Job).where(Job.status.in_(["pending", "running", "paused"]))
    return list(db.execute(stmt).scalars().all())


# ── Job Events ───────────────────────────────────────────────────

def add_event(
    db: Session,
    job_id: str,
    step: str,
    message: str,
    percent: Optional[float],
) -> JobEvent:
    """Persiste un evento de progreso para el job."""
    event = JobEvent(
        job_id=job_id,
        step=step,
        message=message,
        percent=percent,
        occurred_at=_now(),
    )
    db.add(event)
    return event


def get_events(db: Session, job_id: str) -> list[JobEvent]:
    stmt = select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id)
    return list(db.execute(stmt).scalars().all())


# ── Short Snapshots ───────────────────────────────────────────────────

def create_snapshot(db: Session, data: dict) -> ShortSnapshot:
    """Guarda un snapshot de short para aprendizaje."""
    snap = ShortSnapshot(
        project=data["project"],
        episode=data["episode"],
        short_index=data["short_index"],
        title=data.get("title"),
        start=data["start"],
        end=data["end"],
        duration=data["duration"],
        score=data.get("score"),
        speakers=data.get("speakers"),
        speaker_zones=data.get("speaker_zones"),
        camera_segments=data.get("camera_segments"),
        extra_data=data.get("extra_data"),
        snapshot_reason=data.get("snapshot_reason"),
        created_at=_now(),
    )
    db.add(snap)
    return snap


def list_snapshots(
    db: Session,
    project: str,
    episode: str,
) -> list[ShortSnapshot]:
    stmt = (
        select(ShortSnapshot)
        .where(ShortSnapshot.project == project, ShortSnapshot.episode == episode)
        .order_by(ShortSnapshot.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())
