"""Generación de guion/screenplay a partir de la transcripción y momentos detectados.

Usa IA para inferir speakers del contexto y formatear como guion:
  PERSONAJE 1: "texto dicho"
  PERSONAJE 2: "texto dicho"
"""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir, save_json,
)

logger = logging.getLogger(__name__)

GUION_SYSTEM_PROMPT = """Eres un editor profesional de guiones para podcasts y videos.

Tu tarea es transformar la transcripción en un GUION con formato de screenplay/libreto.

REGLAS:
1. Identifica a cada persona que habla por contexto (presentaciones, cambios de voz, pausas, turnos de pregunta-respuesta).
2. Asigna un nombre a cada persona. Si se presentan, usa su nombre real. Si no, usa "Persona 1", "Persona 2", etc.
3. El formato de cada línea debe ser:
   NOMBRE: texto que dijo la persona
4. Mantén el orden cronológico exacto de la transcripción.
5. NO inventes diálogos. Solo reformatea lo que ya está en la transcripción.
6. Puedes limpiar muletillas excesivas ("eh", "este", "bueno" repetidos) pero mantén el tono original.
7. Agrupa las intervenciones continuas de un mismo hablante en un solo bloque.
8. Incluye acotaciones breves entre corchetes para contexto cuando sea útil: [risas], [pausa], [interrumpe].

FORMATO DE SALIDA:
Responde ÚNICAMENTE con el texto del guion. Sin markdown, sin explicaciones, solo el guion.

Ejemplo de formato:
RENZO: Bienvenidos al primer episodio del podcast...
JOHN: Mi nombre es John, tengo 29 años y estudio ingeniería...
MIGUEL: [risas] Yo también disfruto de leer mucho.
"""


def generate_guion(
    *,
    project: str,
    episode: str,
    root: Path,
    model: str | None = None,
    on_progress: ProgressCallback = noop_progress,
) -> str:
    """Genera un guion/screenplay a partir de la transcripción.

    Returns:
        Ruta al archivo guion.txt generado.
    """
    load_dotenv(root / ".env")
    settings = load_settings(root)
    cfg = settings["analysis"]

    ep_dir = get_episode_dir(root, project, episode)
    transcripts_dir = ep_dir / "transcripts"
    guion_dir = ep_dir / "guion"
    guion_dir.mkdir(parents=True, exist_ok=True)

    guion_path = guion_dir / "guion.txt"

    on_progress("guion", "Cargando transcripción...", 5)

    # Leer transcripción
    transcript_files = list(transcripts_dir.glob("*_transcript.txt"))
    if not transcript_files:
        raise FileNotFoundError(f"No se encontró transcript en {transcripts_dir}")

    transcript_text = transcript_files[0].read_text(encoding="utf-8")

    # Leer face_slots si existen (detect ya corrió antes de guion)
    face_context = ""
    face_slots_path = ep_dir / "calibration" / "face_slots.json"
    if face_slots_path.exists():
        try:
            face_slots = json.loads(face_slots_path.read_text(encoding="utf-8"))
            n_faces = len(face_slots)
            if n_faces > 0:
                face_context = (
                    f"\n\nINFORMACIÓN DE VIDEO: Se han detectado {n_faces} posiciones faciales "
                    f"distintas en el video (es decir, hay {n_faces} personas en pantalla). "
                    f"Asegúrate de identificar exactamente {n_faces} speakers."
                )
        except Exception:
            pass

    # Leer momentos si existen (para dar contexto a la IA)
    moments_path = ep_dir / "analysis" / "moments.json"
    moments_context = ""
    if moments_path.exists():
        try:
            moments = json.loads(moments_path.read_text(encoding="utf-8"))
            topics = [m.get("topic", "") for m in moments[:10]]
            moments_context = (
                "\n\nTEMAS PRINCIPALES DETECTADOS EN EL VIDEO:\n"
                + "\n".join(f"- {t}" for t in topics if t)
            )
        except Exception:
            pass

    on_progress("guion", f"Transcript: {len(transcript_text):,} chars", 10)

    # Decidir proveedor: Gemini o custom
    selected_model = model or cfg.get("default_model", "gemini-2.5-flash")
    custom_config = None

    custom_models = cfg.get("custom_models", [])
    custom_match = [cm for cm in custom_models if cm.get("name") == selected_model]

    if custom_match:
        custom_config = custom_match[0]
        key_env = custom_config.get("api_key_env", "")
        if key_env:
            custom_config["api_key"] = os.getenv(key_env, custom_config.get("api_key", ""))
        on_progress("guion", f"Usando modelo custom: {selected_model}", 12)
    else:
        api_keys = []
        for i in range(1, 4):
            key = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if key and key != "your_gemini_api_key_here":
                api_keys.append(key)
        if not api_keys:
            raise RuntimeError("Ninguna GEMINI_API_KEY configurada en .env")
        on_progress("guion", f"Modelo: {selected_model}", 12)

    # Fragmentar transcript si es muy largo (>100k chars)
    max_chars = cfg.get("max_tokens_per_chunk", 28000) * 4
    chunks = _chunk_for_guion(transcript_text, max_chars)
    on_progress("guion", f"Procesando {len(chunks)} fragmento(s)...", 15)

    guion_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        pct = 15 + (i / len(chunks)) * 70
        on_progress("guion", f"Generando guion parte {i}/{len(chunks)}...", pct)

        prompt = (
            f"{GUION_SYSTEM_PROMPT}{face_context}{moments_context}\n\n"
            f"---TRANSCRIPCIÓN (parte {i}/{len(chunks)})---\n"
            f"{chunk}\n"
            f"---FIN TRANSCRIPCIÓN---"
        )

        if custom_config:
            from .analyzer import _call_custom_model
            raw = _call_custom_model(custom_config, prompt)
        else:
            from .analyzer import _call_gemini_with_key_rotation
            raw = _call_gemini_with_key_rotation(
                api_keys, selected_model, prompt, api_keys[0] if api_keys else None, on_progress,
            )

        guion_parts.append(raw.strip())

    guion_text = "\n\n".join(guion_parts)

    # Guardar archivo de guion
    guion_path.write_text(guion_text, encoding="utf-8")
    on_progress("guion", f"Guion guardado ({len(guion_text):,} chars)", 90)

    # Generar guion.docx inmediatamente en la carpeta del episodio
    on_progress("guion", "Generando guion.docx...", 92)
    docx_path = guion_dir / "guion.docx"
    _generate_guion_docx(project, episode, guion_text, docx_path)
    on_progress("guion", "guion.docx generado", 98)

    on_progress("guion", "Guion completado", 100)
    return str(guion_path)


def _chunk_for_guion(text: str, max_chars: int) -> list[str]:
    """Fragmenta el texto en chunks para procesamiento por IA."""
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def _generate_guion_docx(project: str, episode: str, guion_text: str, out_path: Path) -> None:
    """Genera un DOCX formateado del guion (SPEAKER en negrita)."""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        logger.warning("python-docx no instalado, omitiendo guion.docx")
        return

    doc = Document()

    title = doc.add_heading(f"GUION — {project.upper()} / {episode}", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()  # separador

    for line in guion_text.splitlines():
        line = line.strip()
        if not line:
            doc.add_paragraph()
            continue

        # Detectar formato SPEAKER: texto
        if ":" in line:
            colon_idx = line.index(":")
            speaker_part = line[:colon_idx].strip()
            text_part = line[colon_idx + 1:].strip()

            # Verificar que parece un nombre de speaker (mayúsculas o corto)
            if speaker_part and (speaker_part.isupper() or len(speaker_part.split()) <= 3):
                p = doc.add_paragraph()
                run_name = p.add_run(f"{speaker_part}: ")
                run_name.bold = True
                run_name.font.size = Pt(11)
                if text_part:
                    run_text = p.add_run(text_part)
                    run_text.font.size = Pt(11)
                continue

        # Línea normal (acotaciones, etc.)
        p = doc.add_paragraph(line)
        for run in p.runs:
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(80, 80, 80)

    doc.save(str(out_path))
    logger.info("Guion DOCX generado: %s", out_path)
