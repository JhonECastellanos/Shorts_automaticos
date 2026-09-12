"""Análisis de momentos para shorts — Gemini + modelos custom (OpenAI-compatible)."""

import bisect
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir, save_json,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un editor experto en contenido viral para TikTok, Reels y YouTube Shorts.
Analiza la transcripción y extrae entre {min_moments} y {max_moments} segmentos óptimos para shorts.

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
- Si un momento interesante dura menos de {min_dur}s, extiéndelo incluyendo contexto antes/después.
- Si dura más de {max_dur}s, divídelo en partes lógicas.

REGLAS SEMÁNTICAS (obligatorias):
- NUNCA cortar a mitad de un argumento, explicación o historia.
- Cada segmento debe iniciar cuando un speaker comienza una idea completa.
- Respetar cambios de speaker: usar el inicio de una intervención como punto de corte natural.
- El segmento debe terminar tras una conclusión, remate o pausa natural del discurso.

CRITERIOS DE SELECCIÓN (busca TODO lo que aplique):
- Chiste, anécdota o historia con remate
- Tensión narrativa, conflicto o debate
- Pregunta → respuesta rápida
- Dato impactante, frase polémica o declaración fuerte
- Cambio emocional, risas, sorpresa
- Consejo práctico o enseñanza
- Momento de alta densidad informativa

REGLAS DE CANTIDAD:
- Genera entre {min_moments} y {max_moments} shorts. Prioriza calidad Y cantidad.
- start y end deben ser timestamps válidos presentes en la transcripción.
- Se permite solapamiento parcial entre shorts (hasta ~30% del segmento) si cubren perspectivas distintas.
- Extrae todos los segmentos con score >= {min_score}.
- Si se proporciona un GUION con speakers identificados, úsalo para detectar mejor los cambios de hablante y puntos de corte naturales.
- Responder ÚNICAMENTE con el array JSON, sin markdown ni texto adicional.
"""


# ── Utilidades de proveedor ─────────────────────────────────────


def _is_retryable_gemini(exc: BaseException) -> bool:
    """Devuelve True para errores de cuota/rate-limit que merecen reintento."""
    err = str(exc).lower()
    return "429" in err or "quota" in err or "rate" in err or "503" in err


def _call_gemini(client, model_name: str, prompt: str, max_retries: int = 3) -> str:
    """Llama a Gemini con reintentos exponenciales en errores de cuota / rate-limit.
    
    Usa google.genai (nuevo SDK).
    """

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=60, min=60, max=180),
        retry=retry_if_exception(_is_retryable_gemini),
        reraise=True,
    )
    def _attempt() -> str:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text
        except Exception as exc:
            err = str(exc)
            if _is_retryable_gemini(exc):
                logger.warning("[Gemini] Cuota/rate-limit para %s, reintentando…", model_name)
                raise  # tenacity reintentará
            if "404" in err:
                raise RuntimeError(
                    f"Gemini modelo '{model_name}' no encontrado (404). "
                    "Verifica el nombre del modelo."
                ) from exc
            if "403" in err:
                raise RuntimeError(
                    f"Gemini acceso denegado para '{model_name}' (403). "
                    "Verifica la API key y permisos."
                ) from exc
            if "400" in err:
                raise RuntimeError(
                    f"Gemini solicitud inválida para '{model_name}' (400): {err[:200]}"
                ) from exc
            logger.error("[Gemini] Error inesperado con %s: %s", model_name, err[:200])
            raise

    return _attempt()


def _call_gemini_with_key_rotation(
    api_keys: list[str],
    model_name: str,
    prompt: str,
    preferred_key: str | None = None,
    on_progress=None,
) -> str:
    """Llama a Gemini y rota API keys si la actual da 429/quota.

    Intenta primero con *preferred_key* (la key que pasó el test inicial).
    Si falla por cuota, prueba las demás keys antes de rendirse.
    """
    from google import genai

    ordered_keys = list(api_keys)
    if preferred_key and preferred_key in ordered_keys:
        ordered_keys.remove(preferred_key)
        ordered_keys.insert(0, preferred_key)

    last_exc: Exception | None = None
    for key in ordered_keys:
        try:
            client = genai.Client(api_key=key)
            return _call_gemini(client, model_name, prompt)
        except (RetryError, RuntimeError) as exc:
            err = str(exc).lower()
            if "429" in err or "quota" in err or "rate" in err:
                key_hint = key[:8] + "…"
                if on_progress:
                    on_progress("analyze", f"⚠️ Key {key_hint} agotada, rotando…", None)
                last_exc = exc
                continue
            raise
    raise RuntimeError(
        f"Todas las API keys agotadas (429) para {model_name}. "
        "Espera unos minutos o añade más keys."
    ) from last_exc


def _call_custom_model(config: dict, prompt: str, max_retries: int = 2) -> str:
    """Llama a un modelo custom vía API OpenAI-compatible (POST /chat/completions)."""
    import httpx

    base_url = config["base_url"].rstrip("/")
    api_key = config.get("api_key", "")
    model_id = config["model_id"]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 8192,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=300) as client:
                resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise RuntimeError(f"Custom model '{model_id}' error: {exc}") from exc
    raise RuntimeError(f"Se agotaron los reintentos para modelo custom '{model_id}'")


_TS_LINE_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->")


def _annotate_transcript_with_speakers(text: str, segments: list[dict]) -> str:
    """Prefija cada línea con timestamp con [SPEAKER_xx] según el segmento activo.

    Entrada: transcript estilo Whisper (líneas `[HH:MM:SS.mmm --> HH:MM:SS.mmm]  texto`).
    Salida: mismas líneas con prefijo `[SPEAKER_03] ` tras el timestamp.
    """
    if not segments:
        return text

    sorted_segs = sorted(segments, key=lambda s: s.get("start", 0.0))

    def _speaker_at(t: float) -> str:
        # búsqueda lineal corta (segmentos suelen ser < 2000)
        for seg in sorted_segs:
            if seg.get("start", 0.0) <= t <= seg.get("end", 0.0):
                return seg.get("speaker", "SPEAKER_?")
        return "SPEAKER_?"

    out_lines = []
    current_speaker: str | None = None
    for line in text.splitlines():
        m = _TS_LINE_RE.search(line)
        if not m:
            out_lines.append(line)
            continue
        h, mm, s, ms = m.groups()
        ms = ms.ljust(3, "0")
        t = int(h) * 3600 + int(mm) * 60 + int(s) + int(ms) / 1000.0
        speaker = _speaker_at(t)
        if speaker != current_speaker:
            out_lines.append(f"[{speaker}] {line}")
            current_speaker = speaker
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _preprocess_transcript(text: str) -> str:
    """Limpia la transcripción para mejorar la calidad del análisis.

    - Normaliza timestamps a formato uniforme.
    - Colapsa líneas vacías consecutivas.
    - Marca cambios de speaker con separadores claros.
    """
    lines = text.splitlines()
    cleaned = []
    prev_empty = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                cleaned.append("")
                prev_empty = True
            continue
        prev_empty = False

        # Normalizar timestamps: [HH:MM:SS,mmm --> HH:MM:SS,mmm] -> uniforme
        stripped = re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", stripped)

        cleaned.append(stripped)

    return "\n".join(cleaned)


def _setup_gemini(api_keys: list[str], models: list[str], on_progress=None):
    """Prueba combinaciones API key + modelo Gemini y devuelve (client, model, api_key).
    
    Usa google.genai (nuevo SDK).
    """
    from google import genai

    for api_key in api_keys:
        for model_name in models:
            try:
                if on_progress:
                    on_progress("analyze", f"Probando {model_name}...", None)
                client = genai.Client(api_key=api_key)
                test_response = client.models.generate_content(
                    model=model_name, contents="Test"
                )
                if test_response.text:
                    if on_progress:
                        on_progress("analyze", f"✅ Gemini OK: {model_name}", None)
                    return client, model_name, api_key
            except Exception as exc:
                err = str(exc)
                reason = "desconocido"
                if "429" in err or "quota" in err.lower():
                    reason = "cuota excedida"
                elif "404" in err:
                    reason = "modelo no encontrado"
                elif "403" in err:
                    reason = "acceso denegado"
                elif "connection" in err.lower():
                    reason = "sin conexión"
                if on_progress:
                    on_progress("analyze", f"⚠️ {model_name}: {reason}", None)
                continue
    return None, None, None


# ── Función principal de análisis ────────────────────────────────

def analyze(
    *,
    project: str,
    episode: str,
    min_score: float | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    force: bool = False,
    root: Path,
    model: str | None = None,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    load_dotenv(root / ".env")
    settings = load_settings(root)
    cfg = settings["analysis"]

    if min_duration is not None:
        cfg["min_duration_sec"] = min_duration
    if max_duration is not None:
        cfg["max_duration_sec"] = max_duration

    ep_dir = get_episode_dir(root, project, episode)
    transcripts_dir = ep_dir / "transcripts"
    analysis_dir = ep_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    moments_path = analysis_dir / "moments.json"
    if moments_path.exists() and not force:
        on_progress("analyze", "moments.json ya existe", 100)
        return json.loads(moments_path.read_text(encoding="utf-8"))

    # Al re-analizar, borrar archivos derivados para que no ensucien la vista y requieran regenerarse
    if force:
        for fname in ["moments_ranked.json", "moments_with_speaker.json"]:
            p = analysis_dir / fname
            if p.exists():
                p.unlink(missing_ok=True)

    transcript_files = list(transcripts_dir.glob("*_transcript.txt"))
    if not transcript_files:
        raise FileNotFoundError(f"No se encontró transcript en {transcripts_dir}")

    transcript_text = transcript_files[0].read_text(encoding="utf-8")
    transcript_text = _preprocess_transcript(transcript_text)
    effective_min_score = min_score if min_score is not None else cfg["min_score"]

    # ── Contexto de speakers desde diarización (pipeline v3) ──
    # Cargar speaker_segments.json de pyannote + speaker_face_map.json de CMIA.
    # El transcript se anota con [SPEAKER_xx] antes de pasarlo al LLM.
    guion_text = ""  # kept as empty for prompt template compat
    speaker_context = ""
    segments_path = ep_dir / "diarization" / "speaker_segments.json"
    face_map_path = ep_dir / "diarization" / "speaker_face_map.json"
    if segments_path.exists():
        try:
            segments = json.loads(segments_path.read_text(encoding="utf-8"))
            speakers = sorted({s.get("speaker", "UNKNOWN") for s in segments})
            speaker_context = (
                f"Speakers diarizados: {', '.join(speakers)} ({len(speakers)} personas).\n"
                "Usa los marcadores [SPEAKER_xx] de la transcripción para detectar "
                "cambios de hablante y elegir puntos de corte naturales."
            )
            if face_map_path.exists():
                face_map = json.loads(face_map_path.read_text(encoding="utf-8"))
                low_conf = [sp for sp, info in face_map.items() if info.get("confidence", 0) < 0.5]
                if low_conf:
                    speaker_context += f"\nAviso: binding voz↔cara con baja confianza para {', '.join(low_conf)}."
            transcript_text = _annotate_transcript_with_speakers(transcript_text, segments)
            on_progress("analyze", f"Transcript anotado con {len(speakers)} speakers", 7)
        except Exception as exc:
            logger.warning("No se pudo enriquecer transcript con speakers: %s", exc)

    on_progress("analyze", f"Transcript: {len(transcript_text):,} chars", 5)

    chunks = _chunk_transcript(
        transcript_text,
        max_tokens=cfg.get("max_tokens_per_chunk", 28000),
        overlap_sec=cfg.get("chunk_overlap_sec", 120),
    )
    on_progress("analyze", f"Fragmentado en {len(chunks)} chunks", 10)

    system = SYSTEM_PROMPT.format(
        min_dur=cfg["min_duration_sec"],
        max_dur=cfg["max_duration_sec"],
        min_score=effective_min_score,
        min_moments=cfg.get("min_moments", 30),
        max_moments=cfg.get("max_moments", 40),
    )

    # Decidir proveedor: Gemini o custom
    selected_model = model or cfg.get("default_model", "gemini-2.5-flash")
    custom_config = None
    api_keys: list[str] = []
    _active_key: str | None = None

    # Verificar si es un modelo custom
    custom_models = cfg.get("custom_models", [])
    custom_match = [cm for cm in custom_models if cm.get("name") == selected_model]
    if custom_match:
        custom_config = custom_match[0]
        # Resolver API key desde variable de entorno si es referencia
        key_env = custom_config.get("api_key_env", "")
        if key_env:
            custom_config["api_key"] = os.getenv(key_env, custom_config.get("api_key", ""))
        on_progress("analyze", f"Usando modelo custom: {selected_model}", 8)
    else:
        # Gemini: buscar API keys y probar
        api_keys = []
        for i in range(1, 4):
            key = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if key and key != "your_gemini_api_key_here":
                api_keys.append(key)

        if not api_keys:
            raise RuntimeError("Ninguna GEMINI_API_KEY configurada en .env")

        # Si el usuario eligió un modelo explícito, SOLO usar ese.
        # Si no (model=None), usar la lista de fallback de settings.yaml.
        if model:
            gemini_models = [selected_model]
        else:
            gemini_models = cfg.get("models", [cfg.get("model", "gemini-2.5-flash")])
            if isinstance(gemini_models, str):
                gemini_models = [gemini_models]
            if selected_model not in gemini_models:
                gemini_models.insert(0, selected_model)
        gemini_client, selected_model, _active_key = _setup_gemini(api_keys, gemini_models, on_progress)
        if not selected_model:
            raise RuntimeError("Todas las APIs Gemini fallaron. Verifica tus API keys.")

    on_progress("analyze", f"Modelo activo: {selected_model}", 10)

    # --- Caché parcial: moments_partial.json ---
    partial_path = analysis_dir / "moments_partial.json"
    cached_chunks: dict[int, list[dict]] = {}
    if partial_path.exists() and not force:
        try:
            cached_data = json.loads(partial_path.read_text(encoding="utf-8"))
            cached_chunks = {int(k): v for k, v in cached_data.items()}
            on_progress("analyze", f"Caché parcial: {len(cached_chunks)} chunks cacheados", 12)
        except Exception:
            cached_chunks = {}

    all_moments = []

    def _process_chunk(idx: int, chunk: str) -> tuple[int, list[dict]]:
        """Procesa un chunk individual. Thread-safe para Gemini y custom models.

        Si Gemini falla con 429 tras reintentos, rota a la siguiente API key
        disponible y recrea el client.
        """
        prompt = f"{system}\n\n---TRANSCRIPCIÓN---\n{chunk}\n---FIN TRANSCRIPCIÓN---"
        if guion_text:
            prompt += f"\n\n---GUION (speakers identificados)---\n{guion_text[:8000]}\n---FIN GUION---"
        if speaker_context:
            prompt += f"\n\n---CONTEXTO DE SPEAKERS---\n{speaker_context}\n---FIN CONTEXTO---"
        if custom_config:
            raw = _call_custom_model(custom_config, prompt)
        else:
            raw = _call_gemini_with_key_rotation(
                api_keys, selected_model, prompt, _active_key, on_progress,
            )
        return idx, _parse_moments_response(raw)

    use_parallel = len(chunks) > 1
    max_workers = min(3, len(chunks))

    if use_parallel:
        on_progress("analyze", f"Procesando {len(chunks)} chunks en paralelo (x{max_workers})...", 12)

    chunks_to_process = {i: chunk for i, chunk in enumerate(chunks, 1) if i not in cached_chunks}

    # Cargar resultados cacheados
    for idx, moments in cached_chunks.items():
        if idx <= len(chunks):
            all_moments.extend(moments)

    if chunks_to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_chunk, idx, chunk): idx
                for idx, chunk in chunks_to_process.items()
            }
            for future in as_completed(futures):
                idx = futures[future]
                pct = 10 + (idx / len(chunks)) * 75
                try:
                    _, moments = future.result()
                    all_moments.extend(moments)
                    cached_chunks[idx] = moments
                    # Guardar progreso parcial
                    save_json(partial_path, cached_chunks)
                    on_progress("analyze", f"Chunk {idx}/{len(chunks)} completado ({len(moments)} momentos)", pct)
                except (ValueError, json.JSONDecodeError):
                    on_progress("analyze", f"Chunk {idx}: no se pudo parsear respuesta", pct)
                except Exception as e:
                    on_progress("analyze", f"Chunk {idx}: error - {str(e)[:100]}", pct)
    else:
        on_progress("analyze", "Todos los chunks en caché, saltando análisis", 85)

    final = _merge_and_deduplicate(all_moments, effective_min_score)
    final = _enforce_shorts_constraints(
        final,
        min_count=cfg.get("min_moments", 30),
        max_count=cfg.get("max_moments", 40),
        min_dur=cfg["min_duration_sec"],
        max_dur=cfg["max_duration_sec"],
        min_score=effective_min_score,
        all_moments=all_moments,
    )

    # Snap a word boundaries usando el SRT con timestamps por palabra
    srt_files = list(transcripts_dir.glob("*.srt"))
    if srt_files:
        final = _snap_to_word_boundaries(final, srt_files[0])
        final = _snap_to_silences(final, srt_files[0])

    if not final:
        on_progress("analyze", f"❌ Cero momentos guardados. El modelo ({selected_model}) falló o dio JSON inválido.", 100)
        raise RuntimeError(f"Análisis fallido: 0 momentos generados por {selected_model}.")

    save_json(moments_path, final)

    # Limpiar caché parcial tras completar exitosamente
    if partial_path.exists():
        partial_path.unlink(missing_ok=True)

    on_progress("analyze", f"{len(final)} momentos guardados", 100)
    return final


# ── Utilidades internas ──────────────────────────────────────────

def _chunk_transcript(text: str, max_tokens: int, overlap_sec: int) -> list[str]:
    """Fragmenta la transcripción en chunks respetando boundaries de speaker/timestamp.

    Mejoras sobre el corte crudo por caracteres:
    - Corta preferentemente en líneas con gaps de timestamp > 2s (pausas naturales).
    - Incluye resumen del chunk anterior como contexto para el siguiente.
    - Overlap por timestamp retroactivo para no perder momentos en el borde.
    """
    lines = text.splitlines()
    chunks = []
    current: list[str] = []
    current_len = 0
    max_chars = max_tokens * 4
    prev_summary = ""
    ts_pattern = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})[,\.](\d+)")

    def _extract_ts_seconds(line: str) -> float | None:
        m = ts_pattern.search(line)
        if not m:
            return None
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000

    def _find_best_split(lines_block: list[str], start_idx: int) -> int:
        """Busca el mejor punto de corte: una línea con gap > 2s respecto a la anterior."""
        best = len(lines_block)
        # Buscar hacia atrás desde el 90% del bloque hasta el 60%
        search_start = max(start_idx, int(len(lines_block) * 0.6))
        search_end = int(len(lines_block) * 0.95)
        prev_ts = None
        for i in range(search_end, search_start, -1):
            if i >= len(lines_block):
                continue
            ts = _extract_ts_seconds(lines_block[i])
            if ts is None:
                continue
            if prev_ts is None:
                prev_ts = ts
                continue
            # Si hay un gap significativo (> 2s), es un buen punto de corte
            if prev_ts - ts > 2.0:
                return i + 1
            prev_ts = ts
        return best

    for line in lines:
        current.append(line)
        current_len += len(line)
        if current_len >= max_chars:
            split_idx = _find_best_split(current, int(len(current) * 0.5))
            chunk_lines = current[:split_idx]

            # Prepend resumen del chunk anterior para contexto continuo
            if prev_summary:
                chunk_text = f"[Contexto previo: {prev_summary}]\n\n" + "\n".join(chunk_lines)
            else:
                chunk_text = "\n".join(chunk_lines)

            chunks.append(chunk_text)

            # Generar resumen simple del chunk actual (últimas 3 líneas con timestamp)
            summary_lines = [l for l in chunk_lines[-10:] if ts_pattern.search(l)]
            prev_summary = " ".join(summary_lines[-3:])[:200] if summary_lines else ""

            # Overlap: mantener las últimas líneas que tienen timestamps
            overlap_lines = current[split_idx:]
            # Añadir algo del final del chunk actual como overlap
            tail_overlap = [l for l in chunk_lines[-30:] if ts_pattern.search(l)]
            current = tail_overlap + overlap_lines
            current_len = sum(len(l) for l in current)

    if current:
        if prev_summary:
            chunk_text = f"[Contexto previo: {prev_summary}]\n\n" + "\n".join(current)
        else:
            chunk_text = "\n".join(current)
        chunks.append(chunk_text)
    return chunks


def _parse_moments_response(text: str) -> list[dict]:
    text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No se encontró JSON array en la respuesta")
    return json.loads(match.group())


def _merge_and_deduplicate(all_moments: list[dict], min_score: float) -> list[dict]:
    filtered = [m for m in all_moments if m.get("score", 0) >= min_score]
    filtered.sort(key=lambda m: m["score"], reverse=True)
    kept = []
    for candidate in filtered:
        overlap = False
        for existing in kept:
            start_overlap = max(candidate["start"], existing["start"])
            end_overlap = min(candidate["end"], existing["end"])
            if end_overlap > start_overlap:
                duration_candidate = candidate["end"] - candidate["start"]
                # Solo descartamos si es el MISMO short (más del 70% idéntico)
                if duration_candidate > 0 and (end_overlap - start_overlap) / duration_candidate > 0.70:
                    overlap = True
                    break
        if not overlap:
            kept.append(candidate)
    return kept


def _enforce_shorts_constraints(
    moments: list[dict],
    *,
    min_count: int,
    max_count: int,
    min_dur: int,
    max_dur: int,
    min_score: float,
    all_moments: list[dict],
) -> list[dict]:
    """Aplica restricciones duras de cantidad y duración sobre los momentos filtrados.

    1. Rechaza segmentos fuera del rango de duración.
    2. Si hay más de max_count, se queda con los top por score.
    3. Si hay menos de min_count, relaja min_score y re-incorpora descartados.
    """
    # — Paso 1: filtrar por duración válida —
    valid = []
    for m in moments:
        dur = m["end"] - m["start"]
        if m["start"] >= m["end"]:
            continue
        if min_dur <= dur <= max_dur:
            valid.append(m)

    # — Paso 2: si sobran, recortar al top por score —
    valid.sort(key=lambda m: m.get("score", 0), reverse=True)
    if len(valid) > max_count:
        valid = valid[:max_count]

    # — Paso 3: si faltan, intentar recuperar segmentos del pool completo —
    if len(valid) < min_count:
        existing_keys = {(round(m["start"], 1), round(m["end"], 1)) for m in valid}
        relaxed_score = max(0, min_score - 0.5)
        extras = []
        for m in all_moments:
            key = (round(m["start"], 1), round(m["end"], 1))
            if key in existing_keys:
                continue
            dur = m["end"] - m["start"]
            if m["start"] >= m["end"]:
                continue
            if m.get("score", 0) < relaxed_score:
                continue
            if min_dur <= dur <= max_dur:
                extras.append(m)
                existing_keys.add(key)
        extras.sort(key=lambda m: m.get("score", 0), reverse=True)
        valid.extend(extras[:max_count - len(valid)])

    # — Paso 4: si todavía faltan, intentar merge de segmentos cortos adyacentes —
    if len(valid) < min_count:
        short_segments = [
            m for m in all_moments
            if 0 < (m["end"] - m["start"]) < min_dur
            and m.get("score", 0) >= relaxed_score
        ]
        short_segments.sort(key=lambda m: m["start"])
        i = 0
        while i < len(short_segments) - 1 and len(valid) < min_count:
            a = short_segments[i]
            b = short_segments[i + 1]
            gap = b["start"] - a["end"]
            merged_dur = b["end"] - a["start"]
            if gap <= 5.0 and min_dur <= merged_dur <= max_dur:
                merged = {
                    "start": a["start"],
                    "end": b["end"],
                    "score": max(a.get("score", 0), b.get("score", 0)),
                    "topic": a.get("topic", b.get("topic", "")),
                    "hook": a.get("hook", b.get("hook", "")),
                    "summary": a.get("summary", "") or b.get("summary", ""),
                    "reason": f"Merge: {a.get('topic', '')} + {b.get('topic', '')}",
                }
                valid.append(merged)
                i += 2
            else:
                i += 1

    valid.sort(key=lambda m: m.get("score", 0), reverse=True)
    if len(valid) > max_count:
        valid = valid[:max_count]

    # — Paso 5: si todavía faltan, generar shorts sintéticos desde huecos temporales —
    if len(valid) < min_count:
        # Calcular duración total del contenido
        all_ends = [m["end"] for m in all_moments if m.get("end", 0) > 0]
        total_duration = max(all_ends) if all_ends else 0

        if total_duration > 0:
            # Ordenar existentes por inicio para detectar huecos
            valid.sort(key=lambda m: m["start"])
            gaps = []
            # Hueco antes del primer short
            if valid and valid[0]["start"] > min_dur:
                gaps.append((0, valid[0]["start"]))
            # Huecos entre shorts
            for i in range(len(valid) - 1):
                gap_start = valid[i]["end"]
                gap_end = valid[i + 1]["start"]
                if gap_end - gap_start >= min_dur:
                    gaps.append((gap_start, gap_end))
            # Hueco después del último short
            if valid and total_duration - valid[-1]["end"] > min_dur:
                gaps.append((valid[-1]["end"], total_duration))
            # Si no hay shorts todavía, todo el contenido es un hueco
            if not valid:
                gaps = [(0, total_duration)]

            # Generar shorts sintéticos desde los huecos
            for gap_start, gap_end in gaps:
                if len(valid) >= min_count:
                    break
                gap_len = gap_end - gap_start
                pos = gap_start
                while pos + min_dur <= gap_end and len(valid) < min_count:
                    dur = random.randint(min_dur, min(max_dur, int(gap_end - pos)))
                    valid.append({
                        "start": round(pos, 2),
                        "end": round(pos + dur, 2),
                        "score": 3.0,
                        "topic": "Segmento complementario",
                        "hook": "",
                        "summary": "Segmento generado automáticamente para completar cuota mínima",
                        "reason": "auto-fill",
                    })
                    pos += dur + 5  # 5s de separación entre sintéticos

        # Si aún faltan (video muy corto), permitir solapamiento parcial con existentes
        if len(valid) < min_count and total_duration > 0:
            step = total_duration / (min_count - len(valid) + 1)
            pos = 0.0
            while len(valid) < min_count and pos + min_dur <= total_duration:
                dur = random.randint(min_dur, min(max_dur, int(total_duration - pos)))
                valid.append({
                    "start": round(pos, 2),
                    "end": round(pos + dur, 2),
                    "score": 2.5,
                    "topic": "Segmento complementario",
                    "hook": "",
                    "summary": "Segmento generado con solapamiento para completar cuota mínima",
                    "reason": "auto-fill-overlap",
                })
                pos += step

    valid.sort(key=lambda m: m.get("score", 0), reverse=True)
    if len(valid) > max_count:
        valid = valid[:max_count]

    if len(valid) < min_count:
        logger.warning(
            "Solo se obtuvieron %d shorts (mínimo deseado: %d). "
            "El contenido puede no tener suficientes momentos destacables.",
            len(valid), min_count,
        )

    # Ordenar cronológicamente para la salida final
    valid.sort(key=lambda m: m["start"])
    return valid


def _snap_to_word_boundaries(moments: list[dict], srt_path: Path) -> list[dict]:
    """Ajusta start/end de cada momento al timestamp exacto de la palabra más cercana en el SRT.

    Esto evita cortes a mitad de palabra en el audio/video exportado.
    """
    word_times = _parse_srt_word_times(srt_path)
    if not word_times:
        return moments

    starts = [w[0] for w in word_times]
    ends = [w[1] for w in word_times]

    for m in moments:
        # Snap start: buscar la palabra cuyo inicio sea >= m["start"] más cercano
        idx = bisect.bisect_left(starts, m["start"])
        if idx < len(starts):
            m["start"] = starts[idx]
        elif starts:
            m["start"] = starts[-1]

        # Snap end: buscar la palabra cuyo fin sea <= m["end"] más cercano
        idx = bisect.bisect_right(ends, m["end"]) - 1
        if idx >= 0:
            m["end"] = ends[idx]
        elif ends:
            m["end"] = ends[0]

        # Sanity: start < end
        if m["start"] >= m["end"] and starts and ends:
            m["end"] = m["start"] + 60.0  # fallback mínimo

    return moments


def _parse_srt_word_times(srt_path: Path) -> list[tuple[float, float]]:
    """Extrae lista de (start_seconds, end_seconds) de cada entrada SRT."""
    from .utils import srt_time_to_seconds

    content = srt_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
    )
    times = []
    for match in pattern.finditer(content):
        start = srt_time_to_seconds(match.group(1))
        end = srt_time_to_seconds(match.group(2))
        if end > start:
            times.append((start, end))
    return times


def _snap_to_silences(moments: list[dict], srt_path: Path, window: float = 1.5) -> list[dict]:
    """Ajusta start/end de cada momento al gap de silencio más cercano (dentro de ±window segundos).

    Un gap de silencio es un intervalo entre entradas SRT consecutivas > 0.3s.
    Esto produce cortes más limpios que coinciden con pausas naturales del habla.
    """
    word_times = _parse_srt_word_times(srt_path)
    if len(word_times) < 2:
        return moments

    # Detectar gaps de silencio (>0.3s entre entradas consecutivas)
    silence_points: list[float] = []
    for i in range(1, len(word_times)):
        gap = word_times[i][0] - word_times[i - 1][1]
        if gap >= 0.3:
            # El punto de silencio es el centro del gap
            silence_points.append((word_times[i - 1][1] + word_times[i][0]) / 2)

    if not silence_points:
        return moments

    for m in moments:
        # Snap start al silencio más cercano dentro de la ventana
        best_start = None
        best_start_dist = window
        for sp in silence_points:
            dist = abs(sp - m["start"])
            if dist < best_start_dist:
                best_start_dist = dist
                best_start = sp

        # Snap end al silencio más cercano dentro de la ventana
        best_end = None
        best_end_dist = window
        for sp in silence_points:
            dist = abs(sp - m["end"])
            if dist < best_end_dist:
                best_end_dist = dist
                best_end = sp

        if best_start is not None:
            m["start"] = best_start
        if best_end is not None:
            m["end"] = best_end

        # Sanity: start < end
        if m["start"] >= m["end"]:
            m["end"] = m["start"] + 60.0

    return moments


# ── Ranking heurístico integrado ─────────────────────────────────

_VIRAL_KEYWORDS = {
    "controversy": ["polémico", "controversial", "debate", "discrepo", "no estoy de acuerdo",
                     "mentira", "falso", "verdad que nadie", "te mintieron"],
    "emotional": ["increíble", "impactante", "lloré", "no lo puedo creer", "me cambió la vida",
                   "epifanía", "revelación", "confesión", "nunca antes"],
    "actionable": ["consejo", "tip", "truco", "secreto", "cómo", "paso a paso",
                    "en 5 minutos", "fácil", "rápido", "gratis"],
    "storytelling": ["historia", "anécdota", "cuando yo", "me pasó", "les cuento",
                     "no van a creer", "primera vez"],
    "data": ["estudio", "estadística", "dato", "porcentaje", "investigación",
             "científico", "según", "demuestra"],
}

_HOOK_PATTERNS = [
    (r"^\?|¿", 3.0),
    (r"sabías que|lo que nadie te dice", 2.5),
    (r"error|no hagas|deja de", 2.0),
    (r"la verdad sobre|te explico", 1.5),
]


# ── Enriquecimiento desde validate ───────────────────────────────

_ENRICH_PROMPT = """Eres un editor experto en contenido viral.
Tienes una lista de {n} shorts ya seleccionados. Tu tarea es ENRIQUECER cada short
con análisis adicional, sin crear nuevos ni eliminar existentes.

Para cada short, añade o mejora estos campos:
- "score": refina el score (0-10) considerando el contexto completo
- "categories": array de categorías aplicables (controversy, emotional, actionable, storytelling, data, humor)
- "reason": por qué este short tiene potencial viral (1-2 oraciones)

Si el short ya tiene "reason" o "score", puedes mantenerlos o mejorarlos.
NO cambies start, end, topic, hook ni summary.

Responde ÚNICAMENTE con el array JSON completo, sin markdown.

SHORTS ACTUALES:
{shorts_json}

CONTEXTO DEL GUION:
{guion_context}
"""


def _enrich_from_validate(
    *,
    validate_moments: list[dict],
    ep_dir: Path,
    cfg: dict,
    model: str | None,
    root: Path,
    on_progress: ProgressCallback,
) -> list[dict]:
    """Enriquece los shorts creados por validate con contexto IA adicional."""
    guion_dir = ep_dir / "guion"
    guion_path = guion_dir / "guion.txt"
    guion_text = guion_path.read_text(encoding="utf-8").strip() if guion_path.exists() else ""

    # Preparar contexto compacto de los shorts
    shorts_compact = []
    for m in validate_moments:
        shorts_compact.append({
            "start": m["start"],
            "end": m["end"],
            "score": m.get("score", 0),
            "topic": m.get("topic", ""),
            "hook": m.get("hook", ""),
            "summary": m.get("summary", ""),
            "reason": m.get("reason", ""),
        })

    prompt = _ENRICH_PROMPT.format(
        n=len(shorts_compact),
        shorts_json=json.dumps(shorts_compact, ensure_ascii=False, indent=1),
        guion_context=guion_text[:6000] if guion_text else "(sin guion)",
    )

    on_progress("analyze", f"Enriqueciendo {len(validate_moments)} shorts con IA...", 20)

    # Setup modelo (same pattern as analyze)
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
    else:
        for i in range(1, 4):
            key = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if key and key != "your_gemini_api_key_here":
                api_keys.append(key)
        if not api_keys:
            logger.warning("Sin API keys para enriquecer — devolviendo shorts sin cambios")
            return validate_moments

        if model:
            gemini_models = [selected_model]
        else:
            gemini_models = cfg.get("models", [cfg.get("model", "gemini-2.5-flash")])
            if isinstance(gemini_models, str):
                gemini_models = [gemini_models]
            if selected_model not in gemini_models:
                gemini_models.insert(0, selected_model)
        _, selected_model_resolved, _active_key = _setup_gemini(api_keys, gemini_models, on_progress)
        if not selected_model_resolved:
            logger.warning("APIs Gemini fallaron para enriquecer — devolviendo shorts sin cambios")
            return validate_moments
        selected_model = selected_model_resolved

    on_progress("analyze", f"Modelo para enriquecer: {selected_model}", 30)

    try:
        if custom_config:
            raw = _call_custom_model(custom_config, prompt)
        else:
            raw = _call_gemini_with_key_rotation(api_keys, selected_model, prompt, _active_key, on_progress)

        enriched = _parse_moments_response(raw)
        on_progress("analyze", f"IA devolvió {len(enriched)} shorts enriquecidos", 80)

        # Merge: mantener start/end/topic/hook/summary del original, tomar score/categories/reason del enriquecido
        if len(enriched) == len(validate_moments):
            for orig, enr in zip(validate_moments, enriched):
                if enr.get("score") is not None:
                    orig["score"] = enr["score"]
                if enr.get("categories"):
                    orig["categories"] = enr["categories"]
                if enr.get("reason"):
                    orig["reason"] = enr["reason"]
        else:
            # Si la IA devolvió cantidad diferente, indexar por start time
            enr_map = {round(e["start"], 1): e for e in enriched}
            for orig in validate_moments:
                key = round(orig["start"], 1)
                if key in enr_map:
                    enr = enr_map[key]
                    if enr.get("score") is not None:
                        orig["score"] = enr["score"]
                    if enr.get("categories"):
                        orig["categories"] = enr["categories"]
                    if enr.get("reason"):
                        orig["reason"] = enr["reason"]

        on_progress("analyze", f"Enriquecimiento completado — {len(validate_moments)} shorts", 90)
    except Exception as exc:
        logger.warning("Error al enriquecer shorts con IA: %s — devolviendo originales", exc)
        on_progress("analyze", f"⚠️ Error enriquecimiento IA: {str(exc)[:80]} — usando originales", 90)

    return validate_moments


def _rank_moments(moments: list[dict]) -> list[dict]:
    """Asigna trend_score y rank a cada momento usando heurísticas de viralidad.

    Calibrado para shorts de 60-180 segundos (óptimo 90-120s).
    """
    scored = []
    for m in moments:
        text_content = f"{m.get('hook', '')} {m.get('topic', '')} {m.get('reason', '')}".lower()

        # Score por keywords virales
        keyword_score = 0.0
        categories_hit = []
        for category, keywords in _VIRAL_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_content:
                    keyword_score += 1.5
                    categories_hit.append(category)
                    break

        # Score por hook patterns
        hook_score = 0.0
        hook_text = m.get("hook", "").lower()
        for pattern, pts in _HOOK_PATTERNS:
            if re.search(pattern, hook_text):
                hook_score += pts

        # Score por duración óptima (calibrado para 60-180s, óptimo 90-120s)
        duration = m.get("end", 0) - m.get("start", 0)
        if 90 <= duration <= 120:
            dur_score = 2.0
        elif 60 <= duration < 90 or 120 < duration <= 150:
            dur_score = 1.4
        elif 150 < duration <= 180:
            dur_score = 0.8
        else:
            dur_score = 0.0

        base_score = m.get("score", 5)
        trend_score = round(base_score * 0.4 + keyword_score * 0.25 + hook_score * 0.2 + dur_score * 0.15, 2)

        scored.append({
            **m,
            "trend_score": trend_score,
            "keyword_score": round(keyword_score, 2),
            "hook_score": round(hook_score, 2),
            "duration_score": round(dur_score, 2),
            "base_score": round(base_score, 2),
            "categories": list(set(categories_hit)),
        })

    scored.sort(key=lambda x: x["trend_score"], reverse=True)
    for rank, item in enumerate(scored, 1):
        item["rank"] = rank

    # Devolver en orden cronológico
    scored.sort(key=lambda x: x["start"])
    return scored
