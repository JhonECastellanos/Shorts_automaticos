"""Rutas de shorts: listar, detalle, eliminar, exportar."""

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..models import ShortResponse, ShortUpdateTime
from . import validate_path_params

router = APIRouter()

ROOT = Path(__file__).parent.parent.parent


def _get_public_root() -> Path | None:
    cfg = get_settings().output_cfg()
    root = cfg.get("root")
    return Path(root) if root else None


@router.get("/{project}/{episode}", response_model=list[ShortResponse])
async def list_shorts(project: str, episode: str):
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    analysis_dir = ep_dir / "analysis"

    p = analysis_dir / "moments_ranked.json"
    if not p.exists():
        p = analysis_dir / "moments.json"
    if not p.exists():
        return []

    moments = json.loads(p.read_text(encoding="utf-8"))
    output_dir = ep_dir / "output"
    result = []

    public_root = _get_public_root()
    drafts_dir = output_dir / "drafts"
    for i, m in enumerate(moments):
        short_num = m.get("rank", i + 1)
        ocamo_file = (public_root / project / episode / f"short_{short_num:02d}.mp4") if public_root else None
        projects_file = output_dir / f"short_{short_num:02d}" / f"short_{short_num:02d}.mp4"
        draft_file = drafts_dir / f"short_{short_num:02d}.mp4"

        # Preferencia: final HD (OCAMO > local output) > borrador low-res > nada
        if ocamo_file and ocamo_file.exists():
            file_url, has_video, is_draft = f"/ocamo/{project}/{episode}/short_{short_num:02d}.mp4", True, False
        elif projects_file.exists():
            file_url, has_video, is_draft = f"/projects/{project}/{episode}/output/short_{short_num:02d}/short_{short_num:02d}.mp4", True, False
        elif draft_file.exists():
            file_url, has_video, is_draft = f"/projects/{project}/{episode}/output/drafts/short_{short_num:02d}.mp4", True, True
        else:
            file_url, has_video, is_draft = None, False, False

        result.append(ShortResponse(
            index=short_num,
            start=m["start"],
            end=m["end"],
            duration=round(m["end"] - m["start"], 2),
            score=m.get("score"),
            trend_score=m.get("trend_score"),
            keyword_score=m.get("keyword_score"),
            hook_score=m.get("hook_score"),
            duration_score=m.get("duration_score"),
            base_score=m.get("base_score"),
            topic=m.get("topic"),
            hook=m.get("hook"),
            reason=m.get("reason"),
            dominant_speaker=m.get("dominant_speaker"),
            rank=short_num,
            file=file_url,
            has_video=has_video,
            is_draft=is_draft,
            categories=m.get("categories", []),
        ))

    return sorted(result, key=lambda short: short.rank or short.index)


@router.get("/{project}/{episode}/{index}")
async def get_short_detail(project: str, episode: str, index: int):
    validate_path_params(project, episode)
    meta_path = ROOT / "projects" / project / episode / "output" / f"short_{index:02d}" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, "Short no encontrado")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@router.patch("/{project}/{episode}/{index}")
async def update_short_time(project: str, episode: str, index: int, req: ShortUpdateTime):
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    analysis_dir = ep_dir / "analysis"
    
    p = analysis_dir / "moments_ranked.json"
    if not p.exists():
        p = analysis_dir / "moments.json"
    if not p.exists():
        raise HTTPException(404, "Shorts no encontrados")
    
    try:
        moments = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(500, "Error al leer json")
        
    updated = False
    for i, m in enumerate(moments):
        if m.get("rank", i + 1) == index:
            m["start"] = req.start
            m["end"] = req.end
            updated = True
            break
            
    if not updated:
        raise HTTPException(404, "Short no encontrado en esta posición")
        
    p.write_text(json.dumps(moments, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "updated"}


@router.delete("/{project}/{episode}/{index}")
async def delete_short(project: str, episode: str, index: int):
    """Elimina un short del JSON de momentos y borra su carpeta de output."""
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    analysis_dir = ep_dir / "analysis"

    p = analysis_dir / "moments_ranked.json"
    if not p.exists():
        p = analysis_dir / "moments.json"
    if not p.exists():
        raise HTTPException(404, "Shorts no encontrados")

    moments = json.loads(p.read_text(encoding="utf-8"))
    original_len = len(moments)
    moments = [m for i, m in enumerate(moments) if m.get("rank", i + 1) != index]

    if len(moments) == original_len:
        raise HTTPException(404, "Short no encontrado en esta posición")

    p.write_text(json.dumps(moments, indent=2, ensure_ascii=False), encoding="utf-8")

    # Borrar carpeta de output del short si existe
    short_dir = ep_dir / "output" / f"short_{index:02d}"
    if short_dir.exists():
        shutil.rmtree(short_dir)

    return {"status": "deleted", "remaining": len(moments)}


def _active_job_for(project: str, episode: str):
    """Devuelve el job en awaiting_export/running del proyecto/episodio, si hay."""
    from ..services.job_manager import list_jobs
    for j in list_jobs():
        if j.get("project") != project or j.get("episode") != episode:
            continue
        if j.get("status") in ("awaiting_export", "running", "paused"):
            return j
    return None


def _emit_export_progress(project: str, episode: str, msg: str, exported_count: int, total_count: int) -> None:
    """Publica progreso del step 'export' al job activo del episodio, para que
    el pipeline-bar verde y el pct del paso reflejen la exportación en curso."""
    from ..services.job_manager import _emit_progress
    job = _active_job_for(project, episode)
    if not job:
        return
    pct = int((exported_count / max(1, total_count)) * 100)
    _emit_progress(job["id"], "export", msg, pct)


def _count_exported_finals(ep_dir: Path) -> int:
    output_dir = ep_dir / "output"
    if not output_dir.exists():
        return 0
    count = 0
    for sd in output_dir.glob("short_*"):
        if not sd.is_dir():
            continue
        mp4 = sd / f"{sd.name}.mp4"
        if mp4.exists():
            count += 1
    return count


def _total_shorts(ep_dir: Path) -> int:
    import json
    ranked = ep_dir / "analysis" / "moments_ranked.json"
    base = ep_dir / "analysis" / "moments.json"
    p = ranked if ranked.exists() else base
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return 0


@router.post("/{project}/{episode}/{index}/export")
async def export_single_short(project: str, episode: str, index: int):
    """Exporta un short individual en alta resolución."""
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    analysis_dir = ep_dir / "analysis"

    p = analysis_dir / "moments_ranked.json"
    if not p.exists():
        p = analysis_dir / "moments.json"
    if not p.exists():
        raise HTTPException(404, "Shorts no encontrados")

    from scripts.core.exporter import export_shorts
    from scripts.core.doc_exporter import export_documents

    total = _total_shorts(ep_dir)
    # Emite progreso inicial
    already = _count_exported_finals(ep_dir)
    _emit_export_progress(project, episode, f"Exportando short_{index:02d}...", already, total)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: export_shorts(
                project=project, episode=episode,
                index=index, force=True, root=ROOT,
            ),
        )
        # Genera doc para el short individual (best-effort)
        try:
            export_documents(project=project, episode=episode, root=ROOT)
        except Exception as exc:
            logger_msg = f"(docs err: {exc})"
            _emit_export_progress(project, episode, f"Short {index} exportado {logger_msg}", _count_exported_finals(ep_dir), total)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error exportando: {e}")

    exported = _count_exported_finals(ep_dir)
    _emit_export_progress(project, episode, f"Short {index} final listo ({exported}/{total})", exported, total)
    # Si exportaron todos, marcar export step como completed en el job
    if exported >= total and total > 0:
        _finalize_export_if_all_done(project, episode, total)

    return {"status": "exported", "results": result, "exported_count": exported, "total": total}


def _finalize_export_if_all_done(project: str, episode: str, total: int) -> None:
    """Si todos los shorts están exportados, marca el job como completed."""
    from ..services.job_manager import _emit_progress, _jobs, JobStatus
    job = _active_job_for(project, episode)
    if not job:
        return
    # Solo finalizamos si el status es awaiting_export (el flujo normal).
    if job.get("status") == JobStatus.AWAITING_EXPORT.value:
        job["status"] = JobStatus.COMPLETED.value
        job["progress"] = 100
        if "export" not in job.get("steps_completed", []):
            job.setdefault("steps_completed", []).append("export")
        from datetime import datetime, timezone
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        _emit_progress(job["id"], "export", f"Los {total} shorts fueron exportados", 100)


@router.post("/{project}/{episode}/{index}/preview")
async def preview_short(project: str, episode: str, index: int):
    """Genera un video borrador rápido (baja calidad) para validar."""
    validate_path_params(project, episode)

    from scripts.core.exporter import export_preview

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: export_preview(
                project=project,
                episode=episode,
                index=index,
                root=ROOT,
            ),
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error generando preview: {e}")

    return {"status": "preview_generated", "result": result}


@router.post("/{project}/{episode}/export-all")
async def export_all_shorts(project: str, episode: str):
    """Exporta TODOS los shorts en alta resolución + distribución + docs + fcpxml.

    Va short por short emitiendo progreso proporcional al job activo.
    El render fue en 'edit' a baja resolución (borrador); aquí generamos el
    archivo final high-res (1080x1920, CRF18, preset slow) y lo distribuimos.
    """
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    analysis_dir = ep_dir / "analysis"

    p = analysis_dir / "moments_ranked.json"
    if not p.exists():
        p = analysis_dir / "moments.json"
    if not p.exists():
        raise HTTPException(404, "Shorts no encontrados")

    import json
    moments = json.loads(p.read_text(encoding="utf-8"))
    indices_to_export = [int(m.get("rank", i + 1)) for i, m in enumerate(moments)]
    total = len(indices_to_export)

    from scripts.core.exporter import export_shorts, distribute_shorts
    from scripts.core.doc_exporter import export_documents
    from scripts.core.fcpxml_gen import generate_fcpxml

    def _run():
        exported = _count_exported_finals(ep_dir)
        for idx in indices_to_export:
            short_out = ep_dir / "output" / f"short_{idx:02d}" / f"short_{idx:02d}.mp4"
            if not short_out.exists():
                _emit_export_progress(project, episode, f"Renderizando final short_{idx:02d}...", exported, total)
                export_shorts(
                    project=project, episode=episode, root=ROOT,
                    index=idx, force=True, skip_distribute=True,
                )
                exported = _count_exported_finals(ep_dir)
                _emit_export_progress(project, episode, f"Short {idx} listo ({exported}/{total})", exported, total)
        # Distribuir a OCAMO + docs + fcpxml
        _emit_export_progress(project, episode, "Distribuyendo a OCAMO...", exported, total)
        results = distribute_shorts(project=project, episode=episode, root=ROOT, indices=indices_to_export)
        export_documents(project=project, episode=episode, root=ROOT)
        generate_fcpxml(project=project, episode=episode, root=ROOT)
        return results

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error exportando: {e}")

    exported = _count_exported_finals(ep_dir)
    if exported >= total and total > 0:
        _finalize_export_if_all_done(project, episode, total)

    return {"status": "exported", "results": result, "exported_count": exported, "total": total}


@router.post("/{project}/{episode}/export-selection")
async def export_selection(project: str, episode: str, payload: dict):
    """Exporta una lista específica de índices en alta resolución."""
    validate_path_params(project, episode)
    indices = payload.get("indices", [])
    if not isinstance(indices, list) or not indices:
        raise HTTPException(400, "indices debe ser lista no vacía")
    ep_dir = ROOT / "projects" / project / episode

    from scripts.core.exporter import export_shorts, distribute_shorts
    from scripts.core.doc_exporter import export_documents

    total = _total_shorts(ep_dir)

    def _run():
        exported = _count_exported_finals(ep_dir)
        for idx in indices:
            idx_int = int(idx)
            short_out = ep_dir / "output" / f"short_{idx_int:02d}" / f"short_{idx_int:02d}.mp4"
            if not short_out.exists():
                _emit_export_progress(project, episode, f"Renderizando final short_{idx_int:02d}...", exported, total)
                export_shorts(
                    project=project, episode=episode, root=ROOT,
                    index=idx_int, force=True, skip_distribute=True,
                )
                exported = _count_exported_finals(ep_dir)
                _emit_export_progress(project, episode, f"Short {idx_int} listo ({exported}/{total})", exported, total)
        results = distribute_shorts(project=project, episode=episode, root=ROOT, indices=[int(i) for i in indices])
        try:
            export_documents(project=project, episode=episode, root=ROOT)
        except Exception:
            pass
        return results

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(500, f"Error exportando: {e}")

    exported = _count_exported_finals(ep_dir)
    if exported >= total and total > 0:
        _finalize_export_if_all_done(project, episode, total)

    return {"status": "exported", "results": result, "exported_count": exported, "total": total}


@router.post("/{project}/{episode}/snapshot")
async def snapshot_shorts(project: str, episode: str, reason: str = "manual"):
    """Guarda un snapshot de todos los shorts actuales en la DB antes de reprocesar."""
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    analysis_dir = ep_dir / "analysis"

    p = analysis_dir / "moments_ranked.json"
    if not p.exists():
        p = analysis_dir / "moments.json"
    if not p.exists():
        raise HTTPException(404, "Shorts no encontrados")

    moments = json.loads(p.read_text(encoding="utf-8"))

    # Cargar speaker_zones si existen
    zones_path = ep_dir / "calibration" / "speaker_zones.json"
    speaker_zones = json.loads(zones_path.read_text(encoding="utf-8")) if zones_path.exists() else None

    from ..db import create_snapshot, get_db

    saved = 0
    with get_db() as db:
        for i, m in enumerate(moments):
            idx = m.get("rank", i + 1)
            create_snapshot(db, {
                "project": project,
                "episode": episode,
                "short_index": idx,
                "title": m.get("topic") or m.get("hook"),
                "start": m["start"],
                "end": m["end"],
                "duration": round(m["end"] - m["start"], 2),
                "score": m.get("score"),
                "speakers": m.get("speakers", []),
                "speaker_zones": speaker_zones,
                "camera_segments": m.get("camera_segments"),
                "extra_data": {k: v for k, v in m.items() if k not in ("start", "end", "score", "speakers", "camera_segments")},
                "snapshot_reason": reason,
            })
            saved += 1

    return {"status": "snapshot_saved", "count": saved}


@router.get("/{project}/{episode}/snapshots")
async def get_snapshots(project: str, episode: str):
    """Lista los snapshots guardados para comparación."""
    validate_path_params(project, episode)

    from ..db import get_db, list_snapshots

    with get_db() as db:
        snaps = list_snapshots(db, project, episode)
        return [
            {
                "id": s.id,
                "short_index": s.short_index,
                "title": s.title,
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "score": s.score,
                "snapshot_reason": s.snapshot_reason,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in snaps
        ]
