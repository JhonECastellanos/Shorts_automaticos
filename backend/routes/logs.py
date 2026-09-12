"""Rutas de logs: log del job y WebSocket para streaming en tiempo real."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..services.job_manager import get_job

_log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{job_id}")
async def get_logs(job_id: str):
    job = get_job(job_id)
    in_memory_log = job.get("log", []) if job else []

    # If in-memory log has data, return it directly
    if in_memory_log:
        return {"job_id": job_id, "log": in_memory_log}

    # Fallback: read from DB (covers recovered jobs with empty log and
    # jobs that only exist in DB after a server restart)
    try:
        from ..db import get_db, get_events
        with get_db() as db:
            rows = get_events(db, job_id)
            if rows:
                log = [{"step": r.step, "msg": r.message, "pct": r.percent} for r in rows]
                # Hydrate in-memory log so subsequent requests are fast
                if job is not None:
                    job["log"] = log
                return {"job_id": job_id, "log": log}
    except Exception as exc:
        _log.debug("No se pudo leer eventos de BD para %s: %s", job_id, exc)

    if job:
        return {"job_id": job_id, "log": []}

    raise HTTPException(404, "Job no encontrado")
