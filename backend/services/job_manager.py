"""Gestión de jobs: ejecución asíncrona del pipeline en background threads."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logging

import shutil

from ..models import JobCreate, JobStatus, PipelineStep, ProgressEvent
from scripts.core.utils import get_episode_dir, get_public_episode_dir, save_json

# In-memory store — fuente de verdad en tiempo real.
# La BD actúa como capa de persistencia duradera (dual-write).
_jobs: dict[str, dict] = {}
_progress_subscribers: dict[str, list] = defaultdict(list)
_lock = threading.Lock()

ROOT = Path(__file__).parent.parent.parent
_log = logging.getLogger(__name__)


class JobInterrupted(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_job_time(value: Any) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _persist_job_status(job: dict) -> None:
    ep_dir = get_episode_dir(ROOT, job["project"], job["episode"], create=True)
    save_json(ep_dir / "pipeline_status.json", {
        "job_id": job["id"],
        "project": job["project"],
        "episode": job["episode"],
        "steps": job.get("steps", []),
        "status": job["status"],
        "current_step": job["current_step"],
        "progress": job["progress"],
        "steps_completed": job["steps_completed"],
        "error": job["error"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "elapsed_seconds": job.get("elapsed_seconds"),
        "step_durations": job.get("step_durations", {}),
    })
    _db_persist_job(job)


# ── Helpers de persistencia en BD (fallan silenciosamente) ───────

def _iso_to_dt(value: Any) -> datetime | None:
    """Convierte string ISO a datetime para columnas DateTime de SQLAlchemy."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _db_create_job_safe(job: dict) -> None:
    """Crea el registro del job en BD. Falla silenciosamente si la BD no está disponible."""
    try:
        from ..db import get_db, create_job as _repo_create
        with get_db() as db:
            _repo_create(db, job)
    except Exception as exc:
        _log.debug("No se pudo crear job en BD: %s", exc)


def _db_persist_job(job: dict) -> None:
    """Actualiza los campos de estado del job en BD. Falla silenciosamente."""
    try:
        from ..db import get_db, update_job_fields as _repo_update
        fields = {
            "status":          job.get("status"),
            "current_step":    job.get("current_step"),
            "progress":        job.get("progress"),
            "steps_completed": list(job.get("steps_completed", [])),
            "step_durations":  dict(job.get("step_durations", {})),
            "error":           job.get("error"),
            "elapsed_seconds": job.get("elapsed_seconds"),
            "updated_at":      datetime.now(timezone.utc),
            "started_at":      _iso_to_dt(job.get("started_at")),
            "finished_at":     _iso_to_dt(job.get("finished_at")),
        }
        with get_db() as db:
            _repo_update(db, job["id"], fields)
    except Exception as exc:
        _log.debug("No se pudo actualizar job en BD: %s", exc)


def _db_persist_event(job_id: str, step: str, msg: str, pct: float | None) -> None:
    """Persiste un evento de progreso en BD. Falla silenciosamente."""
    try:
        from ..db import get_db, add_event as _repo_add_event
        with get_db() as db:
            _repo_add_event(db, job_id, step, msg, pct)
    except Exception as exc:
        _log.debug("No se pudo persistir evento en BD: %s", exc)


def _refresh_elapsed(job: dict) -> None:
    elapsed = float(job.get("_elapsed_base", 0.0))
    started_monotonic = job.get("_started_monotonic")
    if started_monotonic is not None:
        elapsed += time.monotonic() - started_monotonic
    job["elapsed_seconds"] = round(elapsed, 1)


def _start_elapsed(job: dict) -> None:
    if job.get("_started_monotonic") is None:
        job["_started_monotonic"] = time.monotonic()
    _refresh_elapsed(job)


def _freeze_elapsed(job: dict) -> None:
    started_monotonic = job.get("_started_monotonic")
    if started_monotonic is not None:
        job["_elapsed_base"] = float(job.get("_elapsed_base", 0.0)) + (time.monotonic() - started_monotonic)
        job["_started_monotonic"] = None
    _refresh_elapsed(job)


def _active_step_total(job: dict) -> float | None:
    active_step = job.get("_active_step")
    if not active_step:
        return None

    total = float(job.get("_step_elapsed_base", 0.0))
    started_monotonic = job.get("_step_started_monotonic")
    if started_monotonic is not None:
        total += time.monotonic() - started_monotonic
    return round(total, 1)


def _start_active_step(job: dict) -> None:
    if job.get("_active_step") and job.get("_step_started_monotonic") is None:
        job["_step_started_monotonic"] = time.monotonic()
    _refresh_active_step_duration(job)


def _freeze_active_step(job: dict) -> None:
    started_monotonic = job.get("_step_started_monotonic")
    if started_monotonic is not None:
        job["_step_elapsed_base"] = float(job.get("_step_elapsed_base", 0.0)) + (time.monotonic() - started_monotonic)
        job["_step_started_monotonic"] = None
    _refresh_active_step_duration(job)


def _refresh_active_step_duration(job: dict) -> None:
    active_step = job.get("_active_step")
    total = _active_step_total(job)
    if not active_step or total is None:
        return

    current = float(job.setdefault("step_durations", {}).get(active_step, 0.0))
    if total > current:
        job["step_durations"][active_step] = total


def _close_active_step(job: dict, *, finalize: bool) -> None:
    active_step = job.get("_active_step")
    if not active_step:
        return

    _freeze_active_step(job)
    if finalize:
        job["_step_elapsed_base"] = 0.0
        job["_step_started_monotonic"] = None


def _job_sort_key(job: dict) -> tuple[int, float, str, str]:
    status_priority = {
        JobStatus.RUNNING.value: 0,
        JobStatus.PAUSED.value: 1,
        JobStatus.PENDING.value: 2,
        JobStatus.STOPPED.value: 3,
        JobStatus.FAILED.value: 4,
        JobStatus.COMPLETED.value: 5,
        JobStatus.CANCELLED.value: 6,
    }
    return (
        status_priority.get(str(job.get("status")), 99),
        -_parse_job_time(job.get("updated_at")),
        str(job.get("project", "")).lower(),
        str(job.get("episode", "")).lower(),
    )


def _remaining_steps(job: dict) -> list[str]:
    completed = set(job.get("steps_completed", []))
    current_step = job.get("current_step")
    include_steps = current_step is None or current_step in completed
    steps_to_run: list[str] = []

    for step_name in job.get("steps", []):
        if not include_steps and step_name == current_step:
            include_steps = True
        if not include_steps:
            continue
        if step_name in completed:
            continue
        steps_to_run.append(step_name)

    return steps_to_run


def _control_checkpoint(job: dict) -> None:
    control = job.get("_control_cond")
    if control is None:
        return

    while True:
        with control:
            if job.get("_stop_requested"):
                raise JobInterrupted(str(job.get("_interrupt_status") or JobStatus.STOPPED.value))
            if not job.get("_pause_requested"):
                return
            control.wait(timeout=0.5)


def _snapshot_before_cleanup(project: str, episode: str, reason: str) -> None:
    """Guarda snapshots de los shorts actuales antes de limpiar para aprendizaje."""
    import json as _json
    ep_dir = get_episode_dir(ROOT, project, episode)

    # Cargar datos existentes
    moments_path = ep_dir / "diarization" / "moments_with_speaker.json"
    if not moments_path.exists():
        moments_path = ep_dir / "analysis" / "moments_ranked.json"
    if not moments_path.exists():
        moments_path = ep_dir / "analysis" / "moments.json"
    if not moments_path.exists():
        return  # nada que guardar

    zones_path = ep_dir / "calibration" / "speaker_zones.json"
    speaker_zones = None
    if zones_path.exists():
        speaker_zones = _json.loads(zones_path.read_text(encoding="utf-8"))

    moments = _json.loads(moments_path.read_text(encoding="utf-8"))

    try:
        from ..db import get_db, create_snapshot
        with get_db() as db:
            for i, m in enumerate(moments):
                create_snapshot(db, {
                    "project": project.upper(),
                    "episode": episode,
                    "short_index": int(m.get("rank", i + 1)),
                    "title": m.get("topic"),
                    "start": m["start"],
                    "end": m["end"],
                    "duration": round(m["end"] - m["start"], 2),
                    "score": m.get("score"),
                    "speakers": [t["speaker"] for t in m.get("speaker_turns", [])],
                    "speaker_zones": speaker_zones,
                    "camera_segments": m.get("speaker_turns"),
                    "extra_data": {
                        "dominant_speaker": m.get("dominant_speaker"),
                        "hook": m.get("hook"),
                    },
                    "snapshot_reason": reason,
                })
            db.commit()
            _log.info("Guardados %d snapshots antes de cleanup (%s/%s)", len(moments), project, episode)
    except Exception as exc:
        _log.warning("No se pudieron guardar snapshots: %s", exc)


_PIPELINE_ORDER_V3 = [
    "download", "transcribe", "diarize", "face_positions",
    "speaker_bind", "analyze", "edit", "rank", "export",
]

# Alias legacy → paso v3 (para jobs viejos en BD que todavía usen nombres antiguos).
_STEP_ALIASES: dict[str, str] = {
    # Pipeline v2 (muy viejo)
    "detect": "face_positions",
    "calibrate": "speaker_bind",
    "guion": "analyze",       # guion era pseudo-diarización LLM
    "validate": "analyze",    # validate era segundo LLM
    "ranking": "rank",
    "fcpxml": "export",
    "doc_export": "export",
    # Pipeline v3.0-v3.4 (reciente)
    "face_track": "face_positions",
    "asd": "speaker_bind",    # ASD se fusionó en el binding simple
    "cmia_bind": "speaker_bind",
}


def _canonical_step(step: str) -> str:
    return _STEP_ALIASES.get(step, step)


# Artefactos que NUNCA deben borrarse por cleanup del pipeline. Son
# datos pesados/costosos de regenerar: video fuente, audio WAV extraído,
# transcripción whisper. Se mantienen a través de todos los reinicios
# hasta que el usuario elimine el proyecto completo desde la UI.
_PROTECTED_PATHS = frozenset({"input", "audio", "transcripts"})


def _cleanup_from_step(
    project: str, episode: str, from_step: str,
    *, only_steps: set[str] | None = None,
) -> None:
    """Elimina artefactos de un paso y todos los posteriores al reiniciar.

    Garantiza que los datos costosos (input/video, audio/wav, transcripts/) no
    se toquen — aunque el usuario reinicie desde 'download' o 'transcribe', los
    archivos se mantienen (los pasos tienen su propio skip-if-exists).

    Si `only_steps` se pasa, se limpian SOLO los artefactos de esos steps
    (intersección con los posteriores a `from_step`). Útil cuando el user
    lanza un job parcial como `steps=[edit, rank]` — no queremos borrar
    moments.json de analyze aunque esté "después" en el orden canónico.
    """
    from_step = _canonical_step(from_step)
    try:
        step_idx = _PIPELINE_ORDER_V3.index(from_step)
    except ValueError:
        return

    posteriors = set(_PIPELINE_ORDER_V3[step_idx:])
    steps_to_clean = posteriors & only_steps if only_steps else posteriors
    ep_dir = get_episode_dir(ROOT, project, episode)

    # Guardar snapshots si se van a perder datos de análisis/binding/edit
    if steps_to_clean & {"analyze", "diarize", "face_positions", "speaker_bind", "edit", "rank"}:
        _snapshot_before_cleanup(project, episode, f"re-{from_step}")

    # Mapa de step → archivos/carpetas a eliminar (relativas a ep_dir)
    _ARTIFACTS: dict[str, list[str]] = {
        "diarize": [
            "diarization/speaker_segments.json",
        ],
        "face_positions": [
            "faces/face_positions.json",
            "faces/thumbnails",
            # Legacy v3.0-v3.4
            "faces/face_tracks.json",
            "faces/asd_scores.json",
            # Legacy v2
            "calibration/face_slots.json",
            "calibration/thumbnails",
        ],
        "speaker_bind": [
            "diarization/speaker_face_map.json",
            "calibration/speaker_zones.json",
        ],
        "analyze": [
            "analysis/moments.json",
            "analysis/moments_partial.json",
            "analysis/moments_ranked.json",
            "diarization/moments_with_speaker.json",
            # Legacy
            "guion",
            "validation",
        ],
        "edit": [
            # Borradores low-res generados por el paso edit
            "output/drafts",
            # Si también hay shorts finales renderizados (de una iteración previa),
            # los borramos porque el nuevo edit va a regenerar borradores.
            "output",
        ],
        "rank": [
            "analysis/moments_ranked.json",
            "output/guion_validado.docx",
            "output/guion_shorts.docx",
        ],
        # Export limpia TODO output (drafts + finales) porque los finales
        # se regenerarán al volver a exportar.
        "export": ["output"],
    }

    for step in steps_to_clean:
        for artifact in _ARTIFACTS.get(step, []):
            # Safety net: nunca borrar input/, audio/, transcripts/ por cleanup del pipeline.
            top = artifact.split("/", 1)[0]
            if top in _PROTECTED_PATHS:
                _log.warning("Cleanup: rechazado borrado de artefacto protegido '%s'", artifact)
                continue
            if "*" in artifact:
                # glob
                parent = ep_dir / Path(artifact).parent
                pattern = Path(artifact).name
                if parent.is_dir():
                    for f in parent.glob(pattern):
                        f.unlink(missing_ok=True)
                continue
            target = ep_dir / artifact
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink(missing_ok=True)

    # Limpiar carpeta pública OCAMO del episodio
    if steps_to_clean & {"analyze", "edit", "rank", "export"}:
        public_dir = get_public_episode_dir(ROOT, project, episode)
        if public_dir and public_dir.exists():
            for f in public_dir.glob("short_*.mp4"):
                f.unlink(missing_ok=True)

    _log.info("Cleanup desde '%s' para %s/%s completado", from_step, project, episode)


def create_job(req: JobCreate) -> dict:
    for existing in _jobs.values():
        if existing.get("project") != req.project or existing.get("episode") != req.episode:
            continue
            
        if existing.get("status") in {
            JobStatus.PENDING.value,
            JobStatus.RUNNING.value,
            JobStatus.PAUSED.value,
        }:
            stop_job(existing["id"])
            time.sleep(0.5)

    job_id = str(uuid.uuid4())[:8]
    now = _now_iso()

    # Reordenar steps según orden canónico v3 (safety net) y normalizar aliases legacy.
    requested_steps: set[str] = set()
    for s in req.steps:
        requested_steps.add(_canonical_step(s.value))
    ordered_steps = [s for s in _PIPELINE_ORDER_V3 if s in requested_steps]

    # Limpiar artefactos SOLO de los pasos que el user va a ejecutar. Antes
    # borraba TODOS los posteriores, lo que rompía casos como `steps=[edit]`
    # donde analyze NO se pidió — se perdía moments.json por cleanup.
    # Fix: cleanup limitado a los steps presentes en la request.
    first_step = ordered_steps[0] if ordered_steps else "download"
    if first_step not in ("download", "transcribe"):
        _cleanup_from_step(
            req.project, req.episode, first_step,
            only_steps=set(ordered_steps),
        )

    job = {
        "id": job_id,
        "project": req.project,
        "episode": req.episode,
        "url": req.url,
        "local_file": req.local_file,
        "ai_model": req.ai_model,
        "diarization_provider": req.diarization_provider,
        "min_duration": req.min_duration,
        "max_duration": req.max_duration,
        "steps": ordered_steps,
        "status": JobStatus.PENDING.value,
        "current_step": None,
        "progress": 0,
        "steps_completed": [],
        "error": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": None,
        "step_durations": {},
        "log": [],
        "_started_monotonic": None,
        "_elapsed_base": 0.0,
        "_step_started_monotonic": None,
        "_step_elapsed_base": 0.0,
        "_active_step": None,
        "_pause_requested": False,
        "_stop_requested": False,
        "_interrupt_status": None,
        "_control_cond": threading.Condition(),
        "_thread": None,
    }
    with _lock:
        _jobs[job_id] = job
    _db_create_job_safe(job)   # crear registro en BD antes de la primera persist
    _persist_job_status(job)

    thread = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    job["_thread"] = thread
    thread.start()
    return job


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def update_job_config(job_id: str, config: dict) -> dict | None:
    job = _jobs.get(job_id)
    if not job:
        return None
    for k, v in config.items():
        if v is not None:
            job[k] = v
            if k == 'ai_model':
                _emit_progress(job_id, "analyze", f"⚙️ Modelo de análisis cambiado a: {v}", job.get("progress"))
    job["updated_at"] = _now_iso()
    _persist_job_status(job)
    return job


def list_jobs() -> list[dict]:
    return sorted(_jobs.values(), key=_job_sort_key)


def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if not job or job["status"] not in (
        JobStatus.PENDING.value,
        JobStatus.RUNNING.value,
        JobStatus.PAUSED.value,
    ):
        return False

    job["_stop_requested"] = True
    job["_pause_requested"] = False
    job["_interrupt_status"] = JobStatus.CANCELLED.value
    job["status"] = JobStatus.CANCELLED.value
    _freeze_active_step(job)
    _freeze_elapsed(job)
    job["finished_at"] = _now_iso()
    job["updated_at"] = _now_iso()
    _persist_job_status(job)

    control = job.get("_control_cond")
    if control is not None:
        with control:
            control.notify_all()
    return True


def pause_job(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    if not job or job["status"] not in (JobStatus.PENDING.value, JobStatus.RUNNING.value):
        return None

    job["_pause_requested"] = True
    job["status"] = JobStatus.PAUSED.value
    _freeze_active_step(job)
    _freeze_elapsed(job)
    job["finished_at"] = None
    job["updated_at"] = _now_iso()
    _persist_job_status(job)
    return job


def resume_job(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    if not job:
        # El job no está en memoria (p.ej. backend reiniciado). Intentamos
        # recuperarlo de la BD antes de rechazar la solicitud.
        try:
            from ..db import get_db, get_job as db_get_job
            with get_db() as db:
                db_job = db_get_job(db, job_id)
                if db_job:
                    job = _job_dict_from_db(db_job)
                    # Si estaba activo al morir el servidor, lo dejamos como failed
                    stale = {JobStatus.PENDING.value, JobStatus.RUNNING.value, JobStatus.PAUSED.value}
                    if job["status"] in stale:
                        job["status"] = JobStatus.FAILED.value
                        job["error"] = "Servidor reiniciado — estado de ejecución perdido"
                        job["finished_at"] = _now_iso()
                    with _lock:
                        _jobs[job_id] = job
                    _log.info("Job %s recuperado de BD para resume (status=%s)", job_id[:8], job["status"])
        except Exception as exc:
            _log.warning("No se pudo recuperar job %s de BD para resume: %s", job_id[:8], exc)

    if not job:
        _log.warning("resume_job: job %s no encontrado en memoria ni en BD", job_id[:8])
        return None

    if job["status"] == JobStatus.PAUSED.value:
        job["_pause_requested"] = False
        job["_stop_requested"] = False
        job["_interrupt_status"] = None
        job["status"] = JobStatus.RUNNING.value
        job["finished_at"] = None
        job["log"] = []  # Clear stale logs so SSE historical load is fresh
        _start_elapsed(job)
        _start_active_step(job)
        job["updated_at"] = _now_iso()
        _persist_job_status(job)

        control = job.get("_control_cond")
        if control is not None:
            with control:
                control.notify_all()
        return job

    if job["status"] not in (JobStatus.STOPPED.value, JobStatus.AWAITING_EXPORT.value,
                              JobStatus.FAILED.value, JobStatus.CANCELLED.value):
        return None

    thread = job.get("_thread")
    if isinstance(thread, threading.Thread) and thread.is_alive():
        # El thread puede estar terminando su limpieza (e.g. justo después de cancel).
        # Esperamos hasta 3 s para que finalice antes de rechazar el resume.
        thread.join(timeout=3.0)
        if thread.is_alive():
            return None

    job["_pause_requested"] = False
    job["_stop_requested"] = False
    job["_interrupt_status"] = None
    job["status"] = JobStatus.RUNNING.value
    job["finished_at"] = None
    job["log"] = []  # Clear stale logs so SSE historical load is fresh
    _start_elapsed(job)
    _start_active_step(job)
    job["updated_at"] = _now_iso()
    _persist_job_status(job)

    new_thread = threading.Thread(target=_run_pipeline, args=(job_id, True), daemon=True)
    job["_thread"] = new_thread
    new_thread.start()
    return job


def stop_job(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    if not job or job["status"] not in (
        JobStatus.PENDING.value,
        JobStatus.RUNNING.value,
        JobStatus.PAUSED.value,
    ):
        return None

    job["_stop_requested"] = True
    job["_pause_requested"] = False
    job["_interrupt_status"] = JobStatus.STOPPED.value
    job["status"] = JobStatus.STOPPED.value
    _freeze_active_step(job)
    _freeze_elapsed(job)
    job["finished_at"] = _now_iso()
    job["updated_at"] = _now_iso()
    _persist_job_status(job)

    control = job.get("_control_cond")
    if control is not None:
        with control:
            control.notify_all()
    return job


def subscribe_progress(job_id: str, queue):
    _progress_subscribers[job_id].append(queue)


def unsubscribe_progress(job_id: str, queue):
    subs = _progress_subscribers.get(job_id, [])
    if queue in subs:
        subs.remove(queue)


def _emit_progress(job_id: str, step: str, msg: str, pct: float | None):
    event = ProgressEvent(job_id=job_id, step=step, message=msg, percent=pct)
    job = _jobs.get(job_id)
    if job:
        if step != "done":
            job["current_step"] = step
            job["progress"] = pct
        _refresh_elapsed(job)
        _refresh_active_step_duration(job)
        job["updated_at"] = _now_iso()
        job["log"].append({"step": step, "msg": msg, "pct": pct})
        if step != "done":
            _db_persist_event(job_id, step, msg, pct)
        _persist_job_status(job)

    for q in _progress_subscribers.get(job_id, []):
        try:
            q.put_nowait(event)
        except Exception:
            pass


def _run_pipeline(job_id: str, resume: bool = False):
    job = _jobs[job_id]
    if not job.get("started_at"):
        job["started_at"] = _now_iso()

    job["status"] = JobStatus.RUNNING.value
    job["finished_at"] = None
    job["error"] = None
    job["_stop_requested"] = False
    _start_elapsed(job)
    _start_active_step(job)
    job["updated_at"] = _now_iso()
    _persist_job_status(job)

    def on_progress(step, msg, pct):
        _control_checkpoint(job)
        _emit_progress(job_id, step, msg, pct)

    step_map = {
        "download": _step_download,
        "transcribe": _step_transcribe,
        "diarize": _step_diarize,
        "face_positions": _step_face_positions,
        "speaker_bind": _step_speaker_bind,
        "analyze": _step_analyze,
        "edit": _step_edit,
        "rank": _step_rank,
        "export": _step_export,
    }

    try:
        for raw_step_name in _remaining_steps(job):
            # Normalizar aliases legacy (detect→face_track, guion→analyze, etc.)
            step_name = _canonical_step(raw_step_name)
            _control_checkpoint(job)
            if job.get("_stop_requested"):
                raise JobInterrupted(JobStatus.STOPPED.value)

            func = step_map.get(step_name)
            if func:
                is_resumed_step = job.get("_active_step") == step_name and float(job.get("_step_elapsed_base", 0.0)) > 0
                if job.get("_active_step") != step_name:
                    job["_active_step"] = step_name
                    job["_step_elapsed_base"] = 0.0
                    job["_step_started_monotonic"] = None
                    job["progress"] = 0

                _start_active_step(job)
                on_progress(
                    step_name,
                    f"{'Reanudando' if is_resumed_step and resume else 'Iniciando'} {step_name}...",
                    job.get("progress") if is_resumed_step else 0,
                )
                func(job, on_progress)
                _close_active_step(job, finalize=True)
                if step_name not in job["steps_completed"]:
                    job["steps_completed"].append(step_name)

                # Tras completar "rank", SIEMPRE pausar para revisión manual. En v3
                # el export es 100% manual (desde UI: modal post-rank o selección).
                # Antes se pausaba solo si 'export' estaba en los steps pendientes,
                # pero eso rompía el modal si el user lanzaba `steps=["edit","rank"]`.
                if step_name == "rank":
                    job["_active_step"] = None
                    job["_step_elapsed_base"] = 0.0
                    job["_step_started_monotonic"] = None
                    job["status"] = JobStatus.AWAITING_EXPORT.value
                    job["progress"] = 100
                    _freeze_elapsed(job)
                    job["updated_at"] = _now_iso()
                    _persist_job_status(job)
                    _emit_progress(job_id, "awaiting_export", "Ranking completo — revisa los borradores y exporta cuando estés listo", 100)
                    return  # Detener pipeline, export se ejecuta manualmente

                job["_active_step"] = None
                job["_step_elapsed_base"] = 0.0
                job["_step_started_monotonic"] = None
                _refresh_elapsed(job)
                _persist_job_status(job)

        job["status"] = JobStatus.COMPLETED.value
        job["progress"] = 100
        _freeze_elapsed(job)
        job["finished_at"] = _now_iso()
        _persist_job_status(job)
    except JobInterrupted as exc:
        _freeze_active_step(job)
        _freeze_elapsed(job)
        job["status"] = exc.reason
        job["finished_at"] = _now_iso()
        _persist_job_status(job)
    except Exception as exc:
        _freeze_active_step(job)
        _freeze_elapsed(job)
        job["status"] = JobStatus.FAILED.value
        job["finished_at"] = _now_iso()
        job["error"] = f"{exc}\n{traceback.format_exc()}"
        _persist_job_status(job)
    finally:
        _refresh_elapsed(job)
        if job["status"] in (
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.STOPPED.value,
        ) and not job.get("finished_at"):
            job["finished_at"] = _now_iso()
        job["updated_at"] = _now_iso()
        _persist_job_status(job)
        if job["status"] not in (JobStatus.PAUSED.value, JobStatus.AWAITING_EXPORT.value):
            _emit_progress(job_id, "done", f"Job {job['status']}", 100 if job["status"] == JobStatus.COMPLETED.value else None)


def _step_download(job: dict, on_progress):
    # DEBUG: log to file to trace execution
    from pathlib import Path as _P
    _dbg = _P(__file__).parent.parent.parent / "debug_download.log"
    with open(_dbg, "a") as _f:
        _f.write(f"=== _step_download ENTERED ===\n")
        _f.write(f"job_id={job.get('id')}, url={job.get('url')}\n")
    try:
        from scripts.core.downloader import download
        with open(_dbg, "a") as _f:
            _f.write(f"download imported OK\n")
        download(
            project=job["project"], episode=job["episode"],
            url=job.get("url"), local_file=job.get("local_file"),
            root=ROOT, on_progress=on_progress,
        )
        with open(_dbg, "a") as _f:
            _f.write(f"download() returned OK\n")
    except Exception as _exc:
        with open(_dbg, "a") as _f:
            import traceback as _tb
            _f.write(f"EXCEPTION in _step_download: {type(_exc).__name__}: {_exc}\n")
            _f.write(_tb.format_exc())
        raise


def _step_transcribe(job: dict, on_progress):
    from scripts.core.transcriber import transcribe
    from scripts.core.utils import sync_public_transcripts
    from pathlib import Path

    transcribe(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=on_progress,
    )

    # Copiar transcripciones a carpeta pública
    ep_dir = ROOT / "projects" / job["project"] / job["episode"]
    transcripts_dir = ep_dir / "transcripts"
    if transcripts_dir.exists():
        sync_public_transcripts(ROOT, job["project"], job["episode"], transcripts_dir, on_progress, "transcribe")


def _step_diarize(job: dict, on_progress):
    """Diarización pyannote real: VAD + embeddings + clustering + OSD."""
    from scripts.core.diarizer import diarize
    diarize(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=on_progress,
        provider=job.get("diarization_provider") or "pyannote",
    )


def _step_face_positions(job: dict, on_progress):
    """Detección de posiciones faciales fijas (centroides 2D)."""
    from scripts.core.face_positions import detect_face_positions
    detect_face_positions(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=on_progress,
    )


def _step_speaker_bind(job: dict, on_progress):
    """Asocia SPEAKER_xx ↔ face position usando calibración de primeros minutos."""
    from scripts.core.speaker_bind import bind_speakers_to_positions
    bind_speakers_to_positions(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=on_progress,
    )


def _step_analyze(job: dict, on_progress):
    """ÚNICO paso LLM: elige los mejores momentos con transcript anotado por speaker."""
    from scripts.core.analyzer import analyze
    analyze(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=on_progress,
        model=job.get("ai_model"),
        min_duration=job.get("min_duration"),
        max_duration=job.get("max_duration"),
        force=True,
    )


def _step_edit(job: dict, on_progress):
    """Genera los shorts verticales 9:16 aplicando camera switching por CMIA."""
    from scripts.core.editor import edit_shorts
    edit_shorts(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=on_progress,
    )


def _step_rank(job: dict, on_progress):
    """Ranking heurístico + DOCX de guion estructurado sobre los shorts ya renderizados."""
    from scripts.core.ranker import rank_shorts
    rank_shorts(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=on_progress,
        min_duration=job.get("min_duration"),
        max_duration=job.get("max_duration"),
    )


def _generate_guion_validado_docx(
    *,
    project: str,
    episode: str,
    moments: list[dict],
    speaker_zones: dict,
    segments: list[dict],
    cross_check: dict,
    out_path,
) -> None:
    """Genera DOCX con guion validado: participantes + shorts + speaker turns."""
    from pathlib import Path
    from scripts.core.utils import seconds_to_srt_time

    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
    except ImportError:
        _log.warning("python-docx no instalado, omitiendo guion_validado.docx")
        return

    doc = Document()

    # ── Título ──
    title = doc.add_heading(f"GUION VALIDADO — {project.upper()} / {episode}", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── Resumen de validación ──
    summary = cross_check.get("summary", {})
    if summary:
        p = doc.add_paragraph()
        p.add_run("Confianza: ").bold = True
        p.add_run(f"{summary.get('confidence', 'N/A')} ({summary.get('confidence_score', 0):.0%})")
        p.add_run("  |  ")
        p.add_run("Speakers: ").bold = True
        p.add_run(f"{summary.get('total_speakers', 0)}")
        p.add_run("  |  ")
        p.add_run("Face slots: ").bold = True
        p.add_run(f"{summary.get('face_slots_detected', 0)}")

    # ── Tabla de participantes ──
    doc.add_heading("Participantes", level=2)
    speakers_data = cross_check.get("speakers", {})
    if speakers_data:
        table = doc.add_table(rows=1, cols=5)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Light Grid Accent 1"
        hdr_cells = table.rows[0].cells
        for i, txt in enumerate(["Speaker", "Segmentos", "Duración", "% Tiempo", "Posición X"]):
            hdr_cells[i].text = txt
            for paragraph in hdr_cells[i].paragraphs:
                for run in paragraph.runs:
                    run.bold = True
                    run.font.size = Pt(9)

        for sp_name, sp_data in speakers_data.items():
            row_cells = table.add_row().cells
            row_cells[0].text = sp_name
            row_cells[1].text = str(sp_data.get("segments", 0))
            dur = sp_data.get("duration_sec", 0)
            row_cells[2].text = f"{int(dur // 60)}m {int(dur % 60)}s"
            row_cells[3].text = f"{sp_data.get('pct_time', 0):.1f}%"
            cx = sp_data.get("face_center_x")
            row_cells[4].text = f"{cx:.3f}" if cx is not None else "N/A"

    # ── Shorts con speaker turns ──
    doc.add_heading("Shorts", level=2)
    for m in moments:
        rank = m.get("rank", "?")
        score = m.get("score", 0)
        topic = m.get("topic", "Sin título")
        start_ts = seconds_to_srt_time(m["start"]).split(",")[0]
        end_ts = seconds_to_srt_time(m["end"]).split(",")[0]
        duration = m["end"] - m["start"]

        doc.add_heading(f"SHORT #{rank}: {topic}", level=3)
        p_meta = doc.add_paragraph()
        run = p_meta.add_run(f"Score: {score:.1f}  |  {start_ts} — {end_ts}  |  {duration:.0f}s")
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(100, 100, 100)

        turns = m.get("speaker_turns", [])
        if turns:
            prev_speaker = None
            for turn in turns:
                speaker = turn["speaker"]
                turn_start = seconds_to_srt_time(turn["start"]).split(",")[0]

                if prev_speaker is not None and speaker != prev_speaker:
                    p_cam = doc.add_paragraph()
                    run_cam = p_cam.add_run(f"  [{turn_start}] → CAMBIO CÁMARA → {speaker}")
                    run_cam.font.size = Pt(8)
                    run_cam.font.color.rgb = RGBColor(180, 60, 60)
                    run_cam.bold = True

                p_turn = doc.add_paragraph()
                run_ts = p_turn.add_run(f"[{turn_start}] ")
                run_ts.font.size = Pt(8)
                run_ts.font.color.rgb = RGBColor(120, 120, 120)
                run_name = p_turn.add_run(f"{speaker}: ")
                run_name.bold = True
                run_name.font.size = Pt(10)

                prev_speaker = speaker
        else:
            doc.add_paragraph("(sin turnos de speaker)")

        doc.add_paragraph()

    # ── Issues de validación ──
    issues = cross_check.get("issues", [])
    if issues:
        doc.add_heading("Observaciones de validación", level=2)
        for issue in issues:
            severity_icon = {"warning": "⚠️", "info": "ℹ️", "error": "❌"}.get(
                issue.get("severity", "info"), "•"
            )
            doc.add_paragraph(f"{severity_icon} {issue['message']}")

    doc.save(str(out_path))
    _log.info("guion_validado.docx generado: %s", out_path)


def _restore_speaker_zones_from_snapshot(project: str, episode: str) -> None:
    """Si speaker_zones.json no existe, intenta restaurarlo desde snapshots guardados."""
    import json as _json
    ep_dir = get_episode_dir(ROOT, project, episode)
    zones_path = ep_dir / "calibration" / "speaker_zones.json"
    if zones_path.exists():
        return  # ya existe, calibrate() lo leerá directamente

    try:
        from ..db import get_db, list_snapshots
        with get_db() as db:
            snapshots = list_snapshots(db, project.upper(), episode)
            for snap in snapshots:  # ordenados por created_at DESC
                if snap.speaker_zones:
                    zones_path.parent.mkdir(parents=True, exist_ok=True)
                    zones_path.write_text(
                        _json.dumps(snap.speaker_zones, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    _log.info(
                        "speaker_zones restaurado desde snapshot (created_at=%s)",
                        snap.created_at,
                    )
                    return
    except Exception as exc:
        _log.warning("No se pudo restaurar speaker_zones desde snapshots: %s", exc)


def _step_export(job: dict, on_progress):
    """Empaquetado final: sincroniza OCAMO, genera DOCX por short y FCPXML.

    El render ya lo hizo `edit`. Este paso solo distribuye/empaqueta.
    Se dispara manualmente (desde la UI) después de que `rank` deja el job
    en AWAITING_EXPORT. Soporta también export selectivo por índice cuando
    la UI pida exportar un short concreto (ver `export_shorts_selection`).
    """
    from scripts.core.exporter import distribute_shorts
    from scripts.core.doc_exporter import export_documents
    from scripts.core.fcpxml_gen import generate_fcpxml

    def _phase_progress(label: str, start_pct: float, end_pct: float):
        span = max(0.0, end_pct - start_pct)

        def _callback(_step: str, message: str, pct: float | None):
            scaled_pct = None
            if pct is not None:
                bounded = max(0.0, min(float(pct), 100.0))
                scaled_pct = round(start_pct + (bounded / 100.0) * span, 1)
            on_progress("export", f"{label}: {message}", scaled_pct)

        return _callback

    on_progress("export", "Empaquetando shorts, documentos y timeline...", 0)
    distribute_shorts(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=_phase_progress("Distribución", 0, 60),
        indices=job.get("export_indices"),
    )
    export_documents(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=_phase_progress("Docs", 60, 85),
    )
    generate_fcpxml(
        project=job["project"], episode=job["episode"],
        root=ROOT, on_progress=_phase_progress("Timeline", 85, 100),
    )
    on_progress("export", "Exportación final completada", 100)


# ── Recuperación de estado desde BD al iniciar el servidor ────────

def _job_dict_from_db(db_job) -> dict:
    """Reconstruye el dict interno de un job a partir de un registro ORM (sin thread activo)."""
    def _dt_iso(dt):
        if dt is None:
            return None
        return dt.isoformat() if isinstance(dt, datetime) else str(dt)

    return {
        "id":                    db_job.id,
        "project":               db_job.project,
        "episode":               db_job.episode,
        "url":                   db_job.url,
        "local_file":            db_job.local_file,
        "ai_model":              db_job.ai_model,
        "diarization_provider":  db_job.diarization_provider,
        "min_duration":          db_job.min_duration,
        "max_duration":          db_job.max_duration,
        "steps":                 list(db_job.steps or []),
        "status":                db_job.status,
        "current_step":          db_job.current_step,
        "progress":              db_job.progress,
        "steps_completed":       list(db_job.steps_completed or []),
        "error":                 db_job.error,
        "step_durations":        dict(db_job.step_durations or {}),
        "elapsed_seconds":       db_job.elapsed_seconds,
        "created_at":            _dt_iso(db_job.created_at),
        "updated_at":            _dt_iso(db_job.updated_at),
        "started_at":            _dt_iso(db_job.started_at),
        "finished_at":           _dt_iso(db_job.finished_at),
        # Entradas de log se cargan por separado vía /api/logs
        "log":                   [],
        # Campos internos inertes (no hay thread vivo)
        "_started_monotonic":    None,
        "_elapsed_base":         0.0,
        "_step_started_monotonic": None,
        "_step_elapsed_base":    0.0,
        "_active_step":          None,
        "_pause_requested":      False,
        "_stop_requested":       False,
        "_interrupt_status":     None,
        "_control_cond":         threading.Condition(),
        "_thread":               None,
    }


def recover_jobs_from_db() -> int:
    """Carga los jobs persistidos en BD al arrancar el servidor.

    - Jobs ACTIVOS (pending/running/paused): se marcan como failed porque
      sus threads de ejecución ya no existen tras el reinicio.
    - Jobs históricos (completed/failed/cancelled/stopped): se cargan
      en _jobs para que la UI pueda mostrar el historial sin consultas extra.

    Devuelve el número de jobs recuperados. Si la BD no está disponible,
    devuelve 0 sin propagar el error.
    """
    try:
        from ..db import get_db, list_jobs as db_list_jobs, update_job_fields as db_update
    except Exception:
        return 0

    recovered = 0
    try:
        with get_db() as db:
            db_jobs = db_list_jobs(db)
            stale_statuses = {
                JobStatus.PENDING.value,
                JobStatus.RUNNING.value,
                JobStatus.PAUSED.value,
            }
            now = datetime.now(timezone.utc)
            for db_job in db_jobs:
                job = _job_dict_from_db(db_job)

                if db_job.status in stale_statuses:
                    job["status"] = JobStatus.FAILED.value
                    job["error"] = "Servidor reiniciado — estado de ejecución perdido"
                    job["finished_at"] = now.isoformat()
                    job["updated_at"] = now.isoformat()
                    db_update(db, db_job.id, {
                        "status":      JobStatus.FAILED.value,
                        "error":       "Servidor reiniciado — estado de ejecución perdido",
                        "finished_at": now,
                        "updated_at":  now,
                    })

                with _lock:
                    if db_job.id not in _jobs:
                        _jobs[db_job.id] = job
                recovered += 1

        _log.info("Jobs recuperados desde BD: %d", recovered)
    except Exception as exc:
        _log.warning("No se pudieron recuperar jobs desde BD: %s", exc)

    return recovered
