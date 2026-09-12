"""Ranking de shorts — paso 'rank' del pipeline.

Se ejecuta DESPUÉS de 'edit' (shorts ya renderizados). Aplica:
1. Constraints de cantidad/duración sobre los shorts generados.
2. Ranking heurístico basado en trend keywords, hooks, duración óptima y score IA.
3. Generación de ``output/guion_validado.docx`` — guion estructurado por short
   con timestamps de cambios de cámara, speaker y texto transcrito.

Produce:
    - ``analysis/moments_ranked.json``
    - ``output/guion_validado.docx``

Tras este paso el job queda en estado AWAITING_EXPORT hasta que el usuario
dispare el export manualmente desde la UI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .analyzer import _enforce_shorts_constraints, _rank_moments
from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir,
    save_json, load_json, seconds_to_srt_time,
)

logger = logging.getLogger(__name__)


def rank_shorts(
    *,
    project: str,
    episode: str,
    root: Path,
    min_duration: int | None = None,
    max_duration: int | None = None,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    settings = load_settings(root)
    cfg = settings["analysis"]

    ep_dir = get_episode_dir(root, project, episode)
    analysis_dir = ep_dir / "analysis"
    diar_dir = ep_dir / "diarization"
    output_dir = ep_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preferir moments_with_speaker (producido por edit) — ya tiene speaker_turns
    enriched_path = diar_dir / "moments_with_speaker.json"
    moments_path = enriched_path if enriched_path.exists() else analysis_dir / "moments.json"
    if not moments_path.exists():
        raise FileNotFoundError(f"No se encontró moments para rankear: {moments_path}")

    on_progress("rank", "Cargando moments...", 10)
    moments = load_json(moments_path)

    # Aplicar constraints (cantidad/duración/score)
    min_score = cfg.get("min_score", 3.5)
    min_dur = min_duration or cfg["min_duration_sec"]
    max_dur = max_duration or cfg["max_duration_sec"]
    moments = _enforce_shorts_constraints(
        moments,
        min_count=cfg.get("min_moments", 30),
        max_count=cfg.get("max_moments", 40),
        min_dur=min_dur,
        max_dur=max_dur,
        min_score=min_score,
        all_moments=moments,
    )
    on_progress("rank", f"{len(moments)} moments tras filtros", 35)

    # Ranking heurístico
    ranked = _rank_moments(moments)
    on_progress("rank", f"Ranking asignado a {len(ranked)} moments", 55)

    # Guardar moments_ranked.json
    ranked_path = analysis_dir / "moments_ranked.json"
    save_json(ranked_path, ranked)

    # Cargar datos complementarios para el DOCX
    segments = load_json(diar_dir / "speaker_segments.json") if (diar_dir / "speaker_segments.json").exists() else []
    face_map_path = diar_dir / "speaker_face_map.json"
    face_map = load_json(face_map_path) if face_map_path.exists() else {}

    on_progress("rank", "Generando guion_validado.docx...", 75)
    _generate_guion_docx(
        project=project, episode=episode,
        moments=ranked, segments=segments, face_map=face_map,
        out_path=output_dir / "guion_validado.docx",
    )
    on_progress("rank", f"{len(ranked)} shorts rankeados — guion_validado.docx listo", 100)
    return ranked


def _generate_guion_docx(
    *,
    project: str,
    episode: str,
    moments: list[dict],
    segments: list[dict],
    face_map: dict,
    out_path: Path,
) -> None:
    """Genera guion estructurado por short: timestamps, cambios de cámara, speaker, texto."""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
    except ImportError:
        logger.warning("python-docx no instalado — omitiendo guion DOCX")
        return

    doc = Document()

    title = doc.add_heading(f"GUION DE SHORTS — {project.upper()} / {episode}", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Resumen
    p = doc.add_paragraph()
    p.add_run("Total shorts: ").bold = True
    p.add_run(f"{len(moments)}")
    p.add_run("  |  ")
    p.add_run("Speakers: ").bold = True
    p.add_run(f"{len(face_map)}")
    if segments:
        total_dur = sum(s["end"] - s["start"] for s in segments)
        p.add_run("  |  ")
        p.add_run("Duración analizada: ").bold = True
        p.add_run(f"{int(total_dur // 60)}m {int(total_dur % 60)}s")

    # Tabla de participantes
    if face_map:
        doc.add_heading("Participantes (binding voz ↔ rostro)", level=2)
        table = doc.add_table(rows=1, cols=4)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Light Grid Accent 1"
        for i, txt in enumerate(["Speaker", "Track ID (rostro)", "Confianza", "Posición X"]):
            cell = table.rows[0].cells[i]
            cell.text = txt
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.bold = True
                    run.font.size = Pt(9)
        for speaker, info in sorted(face_map.items()):
            cells = table.add_row().cells
            cells[0].text = speaker
            cells[1].text = str(info.get("track_id", "?"))
            cells[2].text = f"{info.get('confidence', 0):.0%}"
            centroid = info.get("centroid", [0.5, 0.5])
            cells[3].text = f"{centroid[0]:.3f}"

    # Shorts
    doc.add_heading("Shorts (estructura por cambios de cámara)", level=2)
    for m in moments:
        rank = m.get("rank", "?")
        score = m.get("score", 0) or m.get("base_score", 0)
        topic = m.get("topic", "Sin título")
        start_ts = seconds_to_srt_time(m["start"]).split(",")[0]
        end_ts = seconds_to_srt_time(m["end"]).split(",")[0]
        dur = m["end"] - m["start"]

        doc.add_heading(f"SHORT #{rank}: {topic}", level=3)
        meta = doc.add_paragraph()
        meta_run = meta.add_run(
            f"Score: {score:.1f}  |  {start_ts} — {end_ts}  |  {dur:.0f}s  |  "
            f"Dominante: {m.get('dominant_speaker', '?')}"
        )
        meta_run.font.size = Pt(9)
        meta_run.font.color.rgb = RGBColor(100, 100, 100)

        if m.get("hook"):
            hook_p = doc.add_paragraph()
            hook_run = hook_p.add_run(f"🎯 Hook: {m['hook']}")
            hook_run.italic = True
            hook_run.font.size = Pt(10)

        # Cambios de cámara con textos del transcript por turno
        turns = m.get("speaker_turns", [])
        if turns:
            prev_speaker = None
            for turn in turns:
                speaker = turn.get("speaker", "?")
                t_start = seconds_to_srt_time(turn["start"]).split(",")[0]
                t_end = seconds_to_srt_time(turn.get("end", turn["start"])).split(",")[0]
                turn_dur = turn.get("end", turn["start"]) - turn["start"]

                if prev_speaker is not None and speaker != prev_speaker:
                    cam_p = doc.add_paragraph()
                    cam_run = cam_p.add_run(f"    ⤷ [{t_start}] CAMBIO DE CÁMARA → {speaker}")
                    cam_run.bold = True
                    cam_run.font.size = Pt(9)
                    cam_run.font.color.rgb = RGBColor(180, 60, 60)

                turn_p = doc.add_paragraph()
                ts_run = turn_p.add_run(f"[{t_start} → {t_end} · {turn_dur:.1f}s] ")
                ts_run.font.size = Pt(8)
                ts_run.font.color.rgb = RGBColor(120, 120, 120)
                name_run = turn_p.add_run(f"{speaker}: ")
                name_run.bold = True
                name_run.font.size = Pt(10)

                prev_speaker = speaker
        else:
            doc.add_paragraph("(sin turnos — posible segmento de un solo speaker)")

        doc.add_paragraph()

    # Pie: espacio para feedback de calibración
    doc.add_heading("Calibración — feedback", level=2)
    doc.add_paragraph(
        "Si detectas shorts donde la cámara muestra al speaker equivocado, anota:"
    )
    doc.add_paragraph("• SHORT # ___  →  Timestamp del error  →  Speaker esperado")
    doc.add_paragraph(
        "Con estos datos se puede recalibrar CMIA (cmia_bind) sin re-hacer diarización ni ASD."
    )

    doc.save(str(out_path))
    logger.info("guion_validado.docx generado: %s", out_path)
