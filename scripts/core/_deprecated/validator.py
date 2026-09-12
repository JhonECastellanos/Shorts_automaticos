"""Paso 'validate' del pipeline: validación + creación de shorts.

Se ejecuta DESPUÉS de detect + guion y ANTES de analyze.
Inputs: transcript + face_slots + guion.
1. Verifica coherencia de N speakers, cobertura temporal del transcript,
   y calidad de los face_slots detectados.
2. Usa IA (Gemini) cruzando TODOS los inputs para crear los 35 shorts
   con la mejor selección de momentos posible.
"""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .utils import (
    ProgressCallback, noop_progress, get_episode_dir, save_json, load_json,
    load_settings,
)

logger = logging.getLogger(__name__)


def validate(
    *,
    project: str,
    episode: str,
    force: bool = False,
    root: Path,
    model: str | None = None,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    """Validación + creación de shorts: transcript + face_slots + guion."""
    load_dotenv(root / ".env")
    ep_dir = get_episode_dir(root, project, episode)
    calibration_dir = ep_dir / "calibration"
    transcripts_dir = ep_dir / "transcripts"
    guion_dir = ep_dir / "guion"
    validation_dir = ep_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    report_path = validation_dir / "validation_report.json"
    moments_path = validation_dir / "moments.json"
    if report_path.exists() and moments_path.exists() and not force:
        on_progress("validate", "validation ya existe (report + moments)", 100)
        return load_json(report_path)

    # ── Cargar inputs ──
    face_slots_path = calibration_dir / "face_slots.json"
    face_slots = load_json(face_slots_path) if face_slots_path.exists() else []

    transcript_files = list(transcripts_dir.glob("*_transcript.txt"))
    transcript_text = ""
    if transcript_files:
        transcript_text = transcript_files[0].read_text(encoding="utf-8")

    srt_files = list(transcripts_dir.glob("*.srt"))
    srt_text = ""
    if srt_files:
        srt_text = srt_files[0].read_text(encoding="utf-8")

    guion_path = guion_dir / "guion.txt"
    guion_text = guion_path.read_text(encoding="utf-8").strip() if guion_path.exists() else ""

    on_progress("validate", f"Inputs: {len(face_slots)} face_slots, transcript {len(transcript_text):,} chars, guion {len(guion_text):,} chars", 5)

    # ── 1. Analizar speakers en guion ──
    on_progress("validate", "Analizando speakers en guion...", 8)
    guion_speakers = _extract_guion_speakers(guion_text)

    # ── 2. Coherencia N speakers vs N face_slots ──
    on_progress("validate", "Verificando coherencia speakers/face_slots...", 12)
    coherence = _check_speaker_coherence(
        guion_speakers=guion_speakers,
        n_face_slots=len(face_slots),
        face_slots=face_slots,
    )

    # ── 3. Cobertura temporal del transcript ──
    on_progress("validate", "Analizando cobertura temporal...", 18)
    coverage = _analyze_transcript_coverage(srt_text)

    # ── 4. Calidad de face_slots ──
    on_progress("validate", "Verificando calidad face_slots...", 22)
    face_quality = _analyze_face_slots_quality(face_slots)

    # ── 5. Crear shorts con IA cruzando TODOS los inputs ──
    on_progress("validate", "Creando shorts con IA (cruzando transcript + guion + face_slots)...", 25)
    settings = load_settings(root)
    cfg = settings.get("analysis", {})
    moments = _create_shorts_with_ai(
        transcript_text=transcript_text,
        srt_text=srt_text,
        guion_text=guion_text,
        guion_speakers=guion_speakers,
        face_slots=face_slots,
        coherence=coherence,
        coverage=coverage,
        cfg=cfg,
        model=model,
        root=root,
        on_progress=on_progress,
    )
    save_json(moments_path, moments)
    on_progress("validate", f"{len(moments)} shorts creados y guardados", 80)

    # ── 6. Generar DOCX de validación ──
    on_progress("validate", "Generando reporte DOCX...", 85)
    docx_path = validation_dir / "validation_report.docx"
    _generate_validation_docx(
        project=project,
        episode=episode,
        guion_speakers=guion_speakers,
        coherence=coherence,
        coverage=coverage,
        face_quality=face_quality,
        face_slots=face_slots,
        out_path=docx_path,
    )

    # ── 7. Reporte final ──
    on_progress("validate", "Generando reporte JSON...", 92)
    report = _build_report(
        guion_speakers=guion_speakers,
        coherence=coherence,
        coverage=coverage,
        face_quality=face_quality,
        n_face_slots=len(face_slots),
        n_shorts=len(moments),
    )
    save_json(report_path, report)
    on_progress("validate", f"Validación completa — {len(moments)} shorts, confianza: {report['confidence']}", 100)
    return report


# ── Extracción de speakers del guion ────────────────────────────────────

def _extract_guion_speakers(guion_text: str) -> list[str]:
    """Extrae nombres de speakers del guion (formato 'NOMBRE:' al inicio de línea)."""
    if not guion_text:
        return []
    pattern = re.compile(r"^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\d]{0,30}):", re.MULTILINE)
    speakers = sorted(set(m.group(1).strip() for m in pattern.finditer(guion_text)))
    return speakers


# ── Coherencia speakers / face_slots ─────────────────────────────────────

def _check_speaker_coherence(
    *,
    guion_speakers: list[str],
    n_face_slots: int,
    face_slots: list[dict],
) -> dict:
    """Verifica coherencia entre N speakers del guion y N face_slots detectados."""
    issues: list[dict] = []
    n_guion = len(guion_speakers)

    if n_guion == 0:
        issues.append({
            "type": "no_guion_speakers",
            "message": "No se detectaron speakers en el guion — la diarización usará detección automática",
            "severity": "warning",
        })
    elif n_face_slots == 0:
        issues.append({
            "type": "no_face_slots",
            "message": "No se detectaron face_slots — la diarización será solo por voz",
            "severity": "warning",
        })
    elif n_guion != n_face_slots:
        issues.append({
            "type": "speaker_mismatch",
            "message": f"El guion tiene {n_guion} speakers pero se detectaron {n_face_slots} posiciones faciales",
            "severity": "warning" if abs(n_guion - n_face_slots) <= 1 else "error",
        })

    # Verificar separación mínima entre face_slots
    if len(face_slots) >= 2:
        cxs = sorted(s["cx"] for s in face_slots)
        min_sep = min(cxs[i + 1] - cxs[i] for i in range(len(cxs) - 1))
        if min_sep < 0.08:
            issues.append({
                "type": "slots_too_close",
                "message": f"Face slots muy cercanos (separación mínima: {min_sep:.3f}) — puede confundir la diarización",
                "severity": "warning",
            })

    # Confianza
    if n_guion > 0 and n_face_slots > 0:
        if n_guion == n_face_slots:
            confidence = "alta"
            confidence_score = 0.95
        elif abs(n_guion - n_face_slots) == 1:
            confidence = "media"
            confidence_score = 0.7
        else:
            confidence = "baja"
            confidence_score = 0.4
    elif n_guion > 0 or n_face_slots > 0:
        confidence = "media"
        confidence_score = 0.5
    else:
        confidence = "baja"
        confidence_score = 0.2

    return {
        "n_guion_speakers": n_guion,
        "n_face_slots": n_face_slots,
        "match": n_guion == n_face_slots and n_guion > 0,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "guion_speakers": guion_speakers,
        "issues": issues,
    }


# ── Cobertura temporal del transcript ────────────────────────────────────

def _analyze_transcript_coverage(srt_text: str) -> dict:
    """Analiza cobertura temporal y gaps del SRT."""
    if not srt_text:
        return {"coverage_pct": 0, "gap_count": 0, "total_gap_sec": 0, "total_duration_sec": 0, "issues": []}

    ts_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2})[,\.](\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2})[,\.](\d{3})"
    )

    def _to_sec(hms: str, ms: str) -> float:
        parts = hms.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) + int(ms) / 1000

    entries = []
    for m in ts_pattern.finditer(srt_text):
        s = _to_sec(m.group(1), m.group(2))
        e = _to_sec(m.group(3), m.group(4))
        if e > s:
            entries.append((s, e))

    if not entries:
        return {"coverage_pct": 0, "gap_count": 0, "total_gap_sec": 0, "total_duration_sec": 0, "issues": []}

    entries.sort()
    total_start = entries[0][0]
    total_end = entries[-1][1]
    total_span = total_end - total_start
    total_covered = sum(e - s for s, e in entries)

    gaps = []
    for i in range(1, len(entries)):
        gap = entries[i][0] - entries[i - 1][1]
        if gap > 2.0:
            gaps.append(round(gap, 1))

    coverage = total_covered / total_span * 100 if total_span > 0 else 0
    issues = []
    if coverage < 70:
        issues.append(f"Cobertura temporal baja: {coverage:.1f}% — posibles problemas de transcripción")
    if len(gaps) > 20:
        issues.append(f"Muchos gaps en transcript: {len(gaps)} (>20 puede indicar audio ruidoso)")

    return {
        "coverage_pct": round(coverage, 1),
        "gap_count": len(gaps),
        "total_gap_sec": round(sum(gaps), 1),
        "total_duration_sec": round(total_span, 1),
        "largest_gaps_sec": sorted(gaps, reverse=True)[:5],
        "issues": issues,
    }


# ── Calidad de face_slots ────────────────────────────────────────────────

def _analyze_face_slots_quality(face_slots: list[dict]) -> dict:
    """Verifica distribución y calidad de los face_slots detectados."""
    issues = []

    if not face_slots:
        return {"n_slots": 0, "avg_width": 0, "spread": 0, "issues": ["Sin face_slots detectados"]}

    cxs = [s["cx"] for s in face_slots]
    widths = [s.get("w", 0.12) for s in face_slots]
    spread = max(cxs) - min(cxs) if len(cxs) > 1 else 0

    if spread < 0.15 and len(face_slots) > 1:
        issues.append(f"Face slots muy agrupados (spread: {spread:.3f}) — todos parecen estar en la misma posición")

    avg_w = sum(widths) / len(widths)
    if avg_w < 0.04:
        issues.append("Caras detectadas muy pequeñas — el video puede estar demasiado lejos")

    return {
        "n_slots": len(face_slots),
        "avg_width": round(avg_w, 4),
        "spread": round(spread, 4),
        "slot_positions": [round(cx, 3) for cx in cxs],
        "issues": issues,
    }


# ── DOCX de validación ───────────────────────────────────────────────────

def _generate_validation_docx(
    *,
    project: str,
    episode: str,
    guion_speakers: list[str],
    coherence: dict,
    coverage: dict,
    face_quality: dict,
    face_slots: list[dict],
    out_path: Path,
) -> None:
    """Genera DOCX con reporte de validación pre-diarización."""
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    title = doc.add_heading(f"VALIDACIÓN PRE-DIARIZACIÓN — {project.upper()} / {episode}", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── Coherencia ──
    doc.add_heading("Coherencia speakers / face_slots", level=2)
    p = doc.add_paragraph()
    p.add_run("Speakers en guion: ").bold = True
    p.add_run(f"{coherence['n_guion_speakers']} — {', '.join(guion_speakers) if guion_speakers else 'ninguno'}")
    p = doc.add_paragraph()
    p.add_run("Face slots detectados: ").bold = True
    p.add_run(str(coherence["n_face_slots"]))
    p = doc.add_paragraph()
    p.add_run("Confianza: ").bold = True
    p.add_run(f"{coherence['confidence']} ({coherence['confidence_score']:.0%})")

    # ── Cobertura ──
    doc.add_heading("Cobertura temporal del transcript", level=2)
    p = doc.add_paragraph()
    p.add_run("Cobertura: ").bold = True
    p.add_run(f"{coverage.get('coverage_pct', 0):.1f}%")
    p.add_run("  |  ")
    p.add_run("Duración total: ").bold = True
    dur_sec = coverage.get("total_duration_sec", 0)
    p.add_run(f"{int(dur_sec // 60)}m {int(dur_sec % 60)}s")
    p.add_run("  |  ")
    p.add_run("Gaps: ").bold = True
    p.add_run(f"{coverage.get('gap_count', 0)} ({coverage.get('total_gap_sec', 0):.1f}s)")

    # ── Face slots ──
    if face_slots:
        doc.add_heading("Face slots detectados", level=2)
        for i, slot in enumerate(face_slots):
            doc.add_paragraph(f"Slot {i}: cx={slot['cx']:.3f}, cy={slot.get('cy', 0):.3f}, w={slot.get('w', 0):.3f}")

    # ── Issues ──
    all_issues = coherence.get("issues", []) + [{"severity": "info", "message": m} for m in coverage.get("issues", [])] + [{"severity": "info", "message": m} for m in face_quality.get("issues", [])]
    if all_issues:
        doc.add_heading("Observaciones", level=2)
        for issue in all_issues:
            icon = {"warning": "⚠️", "info": "ℹ️", "error": "❌"}.get(
                issue.get("severity", "info"), "•"
            )
            doc.add_paragraph(f"{icon} {issue.get('message', issue) if isinstance(issue, dict) else issue}")

    doc.save(str(out_path))
    logger.info("Validation report DOCX: %s", out_path)


# ── Reporte de validación ────────────────────────────────────────────────

def _build_report(
    *,
    guion_speakers: list[str],
    coherence: dict,
    coverage: dict,
    face_quality: dict,
    n_face_slots: int,
    n_shorts: int = 0,
) -> dict:
    """Construye resumen de validación."""
    all_issues = (
        coherence.get("issues", [])
        + [{"severity": "info", "message": m} for m in coverage.get("issues", [])]
        + [{"severity": "info", "message": m} for m in face_quality.get("issues", [])]
    )

    return {
        "confidence": coherence["confidence"],
        "confidence_score": coherence["confidence_score"],
        "n_guion_speakers": len(guion_speakers),
        "guion_speakers": guion_speakers,
        "n_face_slots": n_face_slots,
        "n_shorts": n_shorts,
        "speaker_face_match": coherence["match"],
        "transcript_coverage_pct": coverage.get("coverage_pct", 0),
        "transcript_duration_sec": coverage.get("total_duration_sec", 0),
        "transcript_gaps": coverage.get("gap_count", 0),
        "face_spread": face_quality.get("spread", 0),
        "issues_count": len(all_issues),
        "issues": all_issues,
    }


# ── Prompt de IA para creación de shorts ─────────────────────────────────

_VALIDATE_SHORTS_PROMPT = """Eres un editor experto en contenido viral para TikTok, Reels y YouTube Shorts.
Tienes acceso a TODOS los datos del video ya procesados: transcripción, guion con speakers, y posiciones faciales.

Tu tarea: crear EXACTAMENTE {n_shorts} shorts óptimos cruzando toda la información disponible.

Estructura JSON requerida por cada momento:
[
  {{
    "start": <float, segundos desde inicio del video>,
    "end": <float, segundos desde inicio del video>,
    "score": <float 0-10, potencial viral>,
    "topic": <string, tema en máximo 5 palabras>,
    "hook": <string, frase gancho de apertura>,
    "summary": <string, resumen de 1-2 oraciones del contenido>,
    "reason": <string, por qué funciona como short>
  }}
]

RESTRICCIONES DE DURACIÓN (obligatorias):
- Cada segmento DEBE durar entre {min_dur} y {max_dur} segundos (end - start).
- Se permiten hasta +20 segundos adicionales ({max_dur_flex}s máximo) SOLO si es necesario para cerrar un contexto o idea.
- Si un momento interesante dura menos de {min_dur}s, extiéndelo incluyendo contexto antes/después.

REGLAS SEMÁNTICAS (obligatorias):
- NUNCA cortar a mitad de un argumento, explicación o historia.
- Cada segmento debe iniciar cuando un speaker comienza una idea completa.
- Respetar cambios de speaker: usar el inicio de una intervención como punto de corte natural.
- El segmento debe terminar tras una conclusión, remate o pausa natural del discurso.

INFORMACIÓN DE SPEAKERS Y CARAS:
{speaker_info}

CRITERIOS DE SELECCIÓN:
- Chiste, anécdota o historia con remate
- Tensión narrativa, conflicto o debate
- Pregunta → respuesta rápida
- Dato impactante, frase polémica o declaración fuerte
- Cambio emocional, risas, sorpresa
- Consejo práctico o enseñanza
- Momento de alta densidad informativa

REGLAS DE CANTIDAD:
- Genera EXACTAMENTE {n_shorts} shorts. Ni más, ni menos.
- Se permite solapamiento parcial entre shorts (hasta ~30%) si cubren perspectivas distintas.
- start y end deben ser timestamps válidos presentes en la transcripción.
- Responder ÚNICAMENTE con el array JSON, sin markdown ni texto adicional.
"""


def _create_shorts_with_ai(
    *,
    transcript_text: str,
    srt_text: str,
    guion_text: str,
    guion_speakers: list[str],
    face_slots: list[dict],
    coherence: dict,
    coverage: dict,
    cfg: dict,
    model: str | None,
    root: Path,
    on_progress: ProgressCallback,
) -> list[dict]:
    """Crea los shorts usando IA cruzando todos los inputs del pipeline."""
    from .analyzer import (
        _setup_gemini,
        _call_gemini_with_key_rotation,
        _call_custom_model,
        _parse_moments_response,
        _enforce_shorts_constraints,
        _chunk_transcript,
        _preprocess_transcript,
    )

    n_shorts = cfg.get("min_moments", 35)
    min_dur = cfg.get("min_duration_sec", 61)
    max_dur = cfg.get("max_duration_sec", 180)
    max_dur_flex = max_dur + 20
    min_score = cfg.get("min_score", 3.5)

    # Construir info de speakers para el prompt
    speaker_lines = []
    if guion_speakers:
        speaker_lines.append(f"Speakers identificados en el guion: {', '.join(guion_speakers)} ({len(guion_speakers)} personas).")
    if face_slots:
        speaker_lines.append(f"Posiciones faciales detectadas: {len(face_slots)} slots.")
        if coherence.get("match"):
            speaker_lines.append("✓ Coincidencia exacta entre speakers y posiciones faciales.")
        else:
            speaker_lines.append(f"⚠ {coherence.get('n_guion_speakers', 0)} speakers vs {coherence.get('n_face_slots', 0)} face_slots.")
    speaker_info = "\n".join(speaker_lines) if speaker_lines else "Sin información de speakers disponible."

    system = _VALIDATE_SHORTS_PROMPT.format(
        n_shorts=n_shorts,
        min_dur=min_dur,
        max_dur=max_dur,
        max_dur_flex=max_dur_flex,
        speaker_info=speaker_info,
    )

    # Setup modelo
    selected_model = model or cfg.get("default_model", "gemini-2.5-flash")
    custom_config = None
    api_keys: list[str] = []
    _active_key: str | None = None

    custom_models = cfg.get("custom_models", [])
    custom_match = [cm for cm in custom_models if cm.get("name") == selected_model]
    if custom_match:
        custom_config = custom_match[0]
        key_env = custom_config.get("api_key_env", "")
        if key_env:
            custom_config["api_key"] = os.getenv(key_env, custom_config.get("api_key", ""))
        on_progress("validate", f"Usando modelo custom: {selected_model}", 28)
    else:
        for i in range(1, 4):
            key = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if key and key != "your_gemini_api_key_here":
                api_keys.append(key)
        if not api_keys:
            raise RuntimeError("Ninguna GEMINI_API_KEY configurada en .env")

        if model:
            gemini_models = [selected_model]
        else:
            gemini_models = cfg.get("models", [cfg.get("model", "gemini-2.5-flash")])
            if isinstance(gemini_models, str):
                gemini_models = [gemini_models]
            if selected_model not in gemini_models:
                gemini_models.insert(0, selected_model)

        # Reutilizar setup de analyzer para probar keys+modelos
        _, selected_model_resolved, _active_key = _setup_gemini(api_keys, gemini_models, lambda s, m, p: on_progress("validate", m, p))
        if not selected_model_resolved:
            raise RuntimeError("Todas las APIs Gemini fallaron en validate. Verifica tus API keys.")
        selected_model = selected_model_resolved

    on_progress("validate", f"Modelo activo para shorts: {selected_model}", 30)

    # Preparar transcript
    clean_transcript = _preprocess_transcript(transcript_text)
    chunks = _chunk_transcript(
        clean_transcript,
        max_tokens=cfg.get("max_tokens_per_chunk", 28000),
        overlap_sec=cfg.get("chunk_overlap_sec", 120),
    )
    on_progress("validate", f"Transcript: {len(clean_transcript):,} chars, {len(chunks)} chunks", 32)

    # Procesar chunks
    all_moments: list[dict] = []
    for idx, chunk in enumerate(chunks, 1):
        pct = 32 + (idx / len(chunks)) * 35
        prompt = f"{system}\n\n---TRANSCRIPCIÓN---\n{chunk}\n---FIN TRANSCRIPCIÓN---"
        if guion_text:
            prompt += f"\n\n---GUION (speakers identificados)---\n{guion_text[:8000]}\n---FIN GUION---"
        try:
            if custom_config:
                raw = _call_custom_model(custom_config, prompt)
            else:
                raw = _call_gemini_with_key_rotation(
                    api_keys, selected_model, prompt, _active_key,
                    lambda s, m, p: on_progress("validate", m, p),
                )
            moments = _parse_moments_response(raw)
            all_moments.extend(moments)
            on_progress("validate", f"Chunk {idx}/{len(chunks)}: {len(moments)} momentos", pct)
        except Exception as e:
            on_progress("validate", f"Chunk {idx}/{len(chunks)}: error - {str(e)[:100]}", pct)
            logger.warning("Validate chunk %d error: %s", idx, e)

    on_progress("validate", f"Total raw: {len(all_moments)} momentos de {len(chunks)} chunks", 70)

    # Deduplicar y aplicar constraints
    from .analyzer import _merge_and_deduplicate
    final = _merge_and_deduplicate(all_moments, min_score)
    final = _enforce_shorts_constraints(
        final,
        min_count=n_shorts,
        max_count=n_shorts,
        min_dur=min_dur,
        max_dur=max_dur_flex,  # Usar duración flexible para no perder shorts por +20s
        min_score=min_score,
        all_moments=all_moments,
    )

    # Snap a word boundaries
    from .analyzer import _snap_to_word_boundaries, _snap_to_silences
    srt_dir = root / "projects" / "placeholder"  # not used, we pass srt directly
    transcripts_dir_actual = get_episode_dir(root, "", "").parent  # won't work, use ep_dir
    ep_dir = get_episode_dir(root, "", "")  # placeholder
    # Find SRT file from the actual episode dir (constructed earlier in validate())
    # We need to find the SRT in the calling context. Since we have srt_text,
    # let's find the srt files properly.
    try:
        # The ep_dir was computed in the caller — reconstruct it
        # We'll get it from the root + transcript path heuristic
        import tempfile
        if srt_text:
            # Write srt_text to a temp file for snapping
            with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as tf:
                tf.write(srt_text)
                temp_srt = Path(tf.name)
            final = _snap_to_word_boundaries(final, temp_srt)
            final = _snap_to_silences(final, temp_srt)
            temp_srt.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("No se pudo hacer snap a word boundaries: %s", e)

    if not final:
        on_progress("validate", "⚠️ IA no generó shorts válidos — el análisis los creará", 75)
        return []

    on_progress("validate", f"{len(final)} shorts finales tras filtros", 78)
    return final
