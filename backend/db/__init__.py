"""Capa de persistencia con SQLAlchemy."""

from .base import Base, check_connection, create_tables, get_db, get_engine, init_engine
from .models import Job, JobEvent, ShortSnapshot
from .repository import (
    add_event,
    create_job,
    create_snapshot,
    get_active_jobs,
    get_events,
    get_job,
    get_jobs_by_project_episode,
    list_jobs,
    list_snapshots,
    update_job_fields,
)

__all__ = [
    "Base",
    "Job",
    "JobEvent",
    "ShortSnapshot",
    "add_event",
    "check_connection",
    "create_job",
    "create_snapshot",
    "create_tables",
    "get_active_jobs",
    "get_db",
    "get_engine",
    "get_events",
    "get_job",
    "get_jobs_by_project_episode",
    "init_engine",
    "list_jobs",
    "list_snapshots",
    "update_job_fields",
]
