"""Diarización de speakers con pyannote.audio / MFCC local (v3)."""

import os as _os_top
# Setear antes de import sklearn/joblib: evita warning de wmic en Windows 11.
_os_top.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os_top.cpu_count() or 4))

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from .utils import (
    ProgressCallback, noop_progress, load_settings, get_episode_dir, save_json, load_json,
    find_video,
)

logger = logging.getLogger(__name__)


def diarize(
    *,
    project: str,
    episode: str,
    force: bool = False,
    root: Path,
    provider: str | None = None,
    on_progress: ProgressCallback = noop_progress,
) -> dict:
    load_dotenv(root / ".env")
    settings = load_settings(root)
    cfg = settings["diarization"]
    
    # v3: solo pyannote. Los providers legacy (text/openroute) fueron eliminados
    # porque dependían de LLM o del guion textual y violan la regla
    # "LLM solo en analyze".
    provider = "pyannote"

    ep_dir = get_episode_dir(root, project, episode)
    audio_dir = ep_dir / "audio"
    diarization_dir = ep_dir / "diarization"
    diarization_dir.mkdir(parents=True, exist_ok=True)

    segments_path = diarization_dir / "speaker_segments.json"

    if segments_path.exists() and not force:
        on_progress("diarize", "speaker_segments.json ya existe", 100)
        segments = load_json(segments_path)
    else:
        audio_files = list(audio_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No se encontró .wav en {audio_dir}.")

        hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
        if not hf_token:
            on_progress(
                "diarize",
                "Sin HUGGINGFACE_TOKEN — usando diarización MFCC local (sin red). "
                "Para máxima precisión con pyannote: crea token en huggingface.co/settings/tokens "
                "y acepta los términos de pyannote/speaker-diarization-3.1.",
                15,
            )
            segments = _run_diarization_mfcc(audio_files[0], cfg, on_progress)
        else:
            try:
                segments = _run_diarization_pyannote(audio_files[0], cfg, on_progress)
            except Exception as e:
                err_str = str(e).lower()
                is_auth = any(k in err_str for k in ("401", "403", "unauthorized", "forbidden", "gated"))
                hint = (
                    "Acepta los términos en:\n"
                    "  - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "  - https://huggingface.co/pyannote/segmentation-3.0"
                    if is_auth else
                    "Revisa conectividad/instalación de pyannote.audio."
                )
                on_progress(
                    "diarize",
                    f"⚠️ pyannote falló ({e}). Fallback a diarización MFCC local. {hint}",
                    20,
                )
                segments = _run_diarization_mfcc(audio_files[0], cfg, on_progress)

        save_json(segments_path, segments)
        on_progress("diarize", f"{len(segments)} segmentos de speaker guardados", 80)

    # Detectar si solo hay 1 speaker (camera switching no se activará)
    unique_speakers = set(s.get("speaker", "") for s in segments)
    if len(unique_speakers) <= 1:
        on_progress(
            "diarize",
            "⚠️ Solo 1 speaker detectado — camera switching no se activará. "
            "Considera re-ejecutar con pyannote si hay múltiples personas.",
            82,
        )

    # v3: speaker ↔ face mapping se hace en el paso 'cmia_bind' con ASD scores.
    # Diarize solo produce speaker_segments.json y termina aquí.
    on_progress("diarize", "Diarización completada", 100)
    return {"speaker_segments": segments}


# ── Diarización MFCC local (sin HuggingFace) ──────────────────────
# Diarización determinista local usando torchaudio (MFCC) + sklearn (k-means
# con auto-K por silhouette). No requiere token HF, no descarga modelos,
# corre en CPU. Precisión menor que pyannote pero suficiente para distinguir
# N voces distintas en un podcast con cámara fija.

def _run_diarization_mfcc(audio_path: Path, cfg: dict, cb: ProgressCallback) -> list[dict]:
    """Diarización basada en MFCC + k-means + smoothing.

    Pipeline:
    1. Cargar audio (torchaudio), downsample a 16kHz mono si hace falta.
    2. Detectar actividad de voz por energía RMS (VAD simple).
    3. Extraer MFCC de ventanas de 1s con hop 0.5s sobre tramos con voz.
    4. Auto-K por silhouette score en rango [min_speakers, max_speakers].
    5. K-means con los MFCC → asigna un cluster a cada ventana.
    6. Smoothing temporal (mediana) para evitar flips espurios.
    7. Agrupar ventanas consecutivas del mismo cluster en segmentos.
    """
    import numpy as _np
    try:
        import torch  # noqa: F401
        import torchaudio
    except ImportError:
        cb("diarize", "torchaudio no disponible, fallback a single-speaker", 35)
        return _run_diarization_single_speaker(audio_path, cb)
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        cb("diarize", "sklearn no disponible, fallback a single-speaker", 35)
        return _run_diarization_single_speaker(audio_path, cb)

    min_speakers = int(cfg.get("min_speakers", 2))
    max_speakers = int(cfg.get("max_speakers", 5))
    # Preferir un K sugerido por el usuario si lo hay (evita que silhouette
    # colapse a K pequeño cuando las voces son similares en MFCC).
    preferred_k = cfg.get("expected_speakers")  # opcional en settings.yaml
    win_sec = 1.5    # ventana más larga → MFCC más estable por speaker
    hop_sec = 0.75
    vad_rms_threshold = 0.008  # ajustable; bajo = más permisivo

    cb("diarize", "Cargando audio para MFCC...", 25)
    # El .wav fue generado por ffmpeg (pcm_s16le, mono, 16kHz). Leemos con scipy
    # para evitar la dependencia libtorchcodec que torchaudio.load necesita.
    from scipy.io import wavfile
    sr, raw = wavfile.read(str(audio_path))
    # Convertir a float32 [-1, 1]
    if raw.dtype == _np.int16:
        wav = raw.astype(_np.float32) / 32768.0
    elif raw.dtype == _np.int32:
        wav = raw.astype(_np.float32) / 2147483648.0
    elif raw.dtype == _np.uint8:
        wav = (raw.astype(_np.float32) - 128) / 128.0
    else:
        wav = raw.astype(_np.float32)
    # Estéreo → mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    import torch
    waveform = torch.from_numpy(wav).unsqueeze(0)  # (1, T)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    total_samples = waveform.shape[1]
    duration = total_samples / sr
    if duration < 2.0:
        cb("diarize", "Audio muy corto (< 2s), asignando 1 speaker", 90)
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": round(duration, 2), "duration": round(duration, 2)}]

    cb("diarize", f"Audio {duration:.0f}s @ {sr}Hz — extrayendo MFCC por ventanas", 35)

    # MFCC extractor enriquecido: 20 coef + deltas + delta-deltas → 60 features
    # Más n_mels (40) para mejor resolución espectral, ventanas 25ms con solape.
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=20,
        melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40, "center": False},
    )
    try:
        compute_deltas = torchaudio.transforms.ComputeDeltas()
    except Exception:
        compute_deltas = None

    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    n_windows = max(1, (total_samples - win_samples) // hop_samples + 1)

    features: list[_np.ndarray] = []
    window_starts: list[float] = []
    waveform_np = waveform.squeeze(0).numpy()

    for i in range(n_windows):
        s = i * hop_samples
        e = s + win_samples
        if e > total_samples:
            break
        chunk = waveform_np[s:e]
        rms = float(_np.sqrt(_np.mean(chunk ** 2)))
        if rms < vad_rms_threshold:
            continue  # silencio, no analizar
        seg_tensor = torch_from_np(chunk)
        mfcc = mfcc_transform(seg_tensor)  # (1, 20, T)
        # Deltas + delta-deltas — capturan dinámica espectral (mejor para speaker ID)
        if compute_deltas is not None:
            delta = compute_deltas(mfcc)
            delta2 = compute_deltas(delta)
            stacked = torch.cat([mfcc, delta, delta2], dim=1)  # (1, 60, T)
        else:
            stacked = mfcc
        mfcc_np = stacked.squeeze(0).numpy()  # (60, T) o (20, T)
        # Stats por ventana: media + std por coef → dimensión final = 2×features
        feat = _np.concatenate([mfcc_np.mean(axis=1), mfcc_np.std(axis=1)])
        features.append(feat)
        window_starts.append(i * hop_sec)

    if len(features) < 4:
        cb("diarize", f"Pocas ventanas con voz ({len(features)}) — asignando 1 speaker", 90)
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": round(duration, 2), "duration": round(duration, 2)}]

    X = _np.vstack(features)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    cb("diarize", f"{len(features)} ventanas de voz — buscando K óptimo [{min_speakers}..{max_speakers}]", 55)

    # Si el usuario especifica expected_speakers en settings, lo respetamos.
    if preferred_k and min_speakers <= int(preferred_k) <= max_speakers:
        k = int(preferred_k)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        best_labels = km.fit_predict(X_norm)
        best_k = k
        best_score = -1.0
        cb("diarize", f"Usando K={k} fijo (expected_speakers en settings)", 62)
    else:
        # Auto-K: priorizamos K más alto si score cercano al mejor.
        # Penalizamos K=2 vs K altos por default cuando scores son similares
        # (silhouette subestima la bondad de más speakers en podcasts).
        best_k = min_speakers
        best_score = -1.0
        best_labels: _np.ndarray | None = None
        scores_per_k: dict[int, float] = {}
        for k in range(max(2, min_speakers), max_speakers + 1):
            if k >= len(X_norm):
                break
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X_norm)
            try:
                sample = min(2000, len(X_norm))
                score = silhouette_score(X_norm[:sample], labels[:sample], metric="euclidean")
            except Exception:
                score = -1.0
            scores_per_k[k] = score
            logger.info("MFCC k=%d silhouette=%.3f", k, score)
            # Ajuste: si score >= 0.85 * mejor_actual y k > best_k, preferir K mayor
            if score > best_score * 0.85 and score > 0:
                if k > best_k or score > best_score:
                    best_score = max(best_score, score)
                    best_k = k
                    best_labels = labels

        if best_labels is None:
            cb("diarize", "Clustering falló, asignando 1 speaker", 90)
            return [{"speaker": "SPEAKER_00", "start": 0.0, "end": round(duration, 2), "duration": round(duration, 2)}]
        cb("diarize", f"Scores por K: {scores_per_k} → elegido K={best_k}", 65)

    cb("diarize", f"Óptimo K={best_k} speakers (silhouette={best_score:.3f}) — smoothing temporal", 70)

    # Smoothing temporal: mediana con ventana 5
    smoothed = _median_smooth(best_labels, window=5)

    # Agrupar ventanas consecutivas del mismo speaker en segmentos
    segments: list[dict] = []
    if len(smoothed) > 0:
        current_spk = int(smoothed[0])
        seg_start = window_starts[0]
        seg_end = seg_start + win_sec
        for i in range(1, len(smoothed)):
            label = int(smoothed[i])
            t = window_starts[i]
            if label == current_spk and (t - seg_end) < (hop_sec * 3):
                seg_end = t + win_sec
            else:
                segments.append({
                    "speaker": f"SPEAKER_{current_spk:02d}",
                    "start": round(seg_start, 2),
                    "end": round(min(seg_end, duration), 2),
                    "duration": round(min(seg_end, duration) - seg_start, 2),
                })
                current_spk = label
                seg_start = t
                seg_end = t + win_sec
        segments.append({
            "speaker": f"SPEAKER_{current_spk:02d}",
            "start": round(seg_start, 2),
            "end": round(min(seg_end, duration), 2),
            "duration": round(min(seg_end, duration) - seg_start, 2),
        })

    # Filtrar segmentos < 0.5s
    segments = [s for s in segments if s["duration"] >= 0.5]
    cb("diarize", f"{len(segments)} segmentos de {best_k} speakers (MFCC local)", 90)
    return segments


def torch_from_np(x):
    import torch
    return torch.from_numpy(x).float().unsqueeze(0)


def _median_smooth(labels, window: int = 5):
    import numpy as _np
    n = len(labels)
    out = _np.array(labels, dtype=_np.int64).copy()
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals, counts = _np.unique(labels[lo:hi], return_counts=True)
        out[i] = int(vals[_np.argmax(counts)])
    return out


# ── Diarización degradada (single-speaker) ──────────────────────────
# Fallback si torchaudio/sklearn no están disponibles.

def _run_diarization_single_speaker(audio_path: Path, cb: ProgressCallback) -> list[dict]:
    import wave as _wave
    try:
        with _wave.open(str(audio_path), "rb") as wf:
            n_frames = wf.getnframes()
            rate = wf.getframerate()
            duration = n_frames / rate if rate else 0.0
    except Exception:
        duration = 0.0

    if duration <= 0:
        cb("diarize", "Audio sin duración — segmento único", 70)
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": 60.0, "duration": 60.0}]

    # Un único segmento que cubre toda la duración
    cb("diarize", f"Diarización degradada: 1 speaker cubriendo {duration:.0f}s", 70)
    return [{
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": round(duration, 2),
        "duration": round(duration, 2),
    }]


# ── Text-based diarization (guion + transcript) ──────────────────────────────

_TS_RE = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\]\s*(.*)"
)

def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _normalize(text: str) -> str:
    """Minúscula, sin puntuación, espacios colapsados."""
    text = re.sub(r"[^\w\sáéíóúñü]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _run_diarization_text(ep_dir: Path, cfg: dict, cb: ProgressCallback) -> list[dict]:
    """Diarización por texto: cruza guion (SPEAKER: texto) con transcript (timestamps).

    Algoritmo: matching por n-gramas de palabras con progresión estrictamente
    hacia adelante en la transcripción.  Después de asignar turnos, divide
    segmentos excesivamente largos usando los silencios (gaps) del transcript.
    """
    guion_dir = ep_dir / "guion"
    transcript_dir = ep_dir / "transcripts"

    # ── 1. Cargar guion ──
    guion_path = guion_dir / "guion.txt"
    if not guion_path.exists():
        raise FileNotFoundError(
            f"No se encontró guion/guion.txt en {ep_dir}. "
            "Ejecuta el paso 'guion' primero o usa provider=pyannote."
        )

    cb("diarize", "Cargando guion y transcripción...", 5)

    guion_blocks: list[dict] = []
    current_speaker = ""
    current_text = ""
    for line in guion_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ \t]+?):\s*(.+)", line)
        if match:
            if current_speaker and current_text:
                guion_blocks.append({"speaker": current_speaker, "text": current_text.strip()})
            current_speaker = match.group(1).strip()
            current_text = re.sub(r"\[.*?\]", "", match.group(2))
        elif current_speaker:
            current_text += " " + re.sub(r"\[.*?\]", "", line)
    if current_speaker and current_text:
        guion_blocks.append({"speaker": current_speaker, "text": current_text.strip()})

    if not guion_blocks:
        raise RuntimeError("No se encontraron speakers en el guion. Formato esperado: NOMBRE: texto")

    cb("diarize", f"{len(guion_blocks)} bloques de speaker en guion", 15)

    # ── 2. Cargar transcripción con timestamps ──
    transcript_files = list(transcript_dir.glob("*_transcript.txt"))
    if not transcript_files:
        raise FileNotFoundError(f"No se encontró *_transcript.txt en {transcript_dir}.")

    ts_segments: list[dict] = []
    for line in transcript_files[0].read_text(encoding="utf-8").splitlines():
        m = _TS_RE.match(line.strip())
        if m:
            start = _ts_to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
            end = _ts_to_sec(m.group(5), m.group(6), m.group(7), m.group(8))
            text = m.group(9).strip()
            if text:
                ts_segments.append({"start": start, "end": end, "text": text})

    if not ts_segments:
        raise RuntimeError("No se encontraron segmentos con timestamps en la transcripción.")

    cb("diarize", f"{len(ts_segments)} segmentos en transcripción", 25)

    # ── 3. Construir secuencia plana de palabras ──
    word_seq: list[tuple[str, int]] = []  # (palabra_normalizada, idx_segmento)
    for seg_i, seg in enumerate(ts_segments):
        for w in _normalize(seg["text"]).split():
            word_seq.append((w, seg_i))

    cb("diarize", "Buscando speakers en transcripción...", 35)

    # ── 4. Match cada bloque del guion con word-level n-gram scoring ──
    matches: list[dict] = []
    last_word_pos = 0

    for block_idx, block in enumerate(guion_blocks):
        guion_words = _normalize(block["text"]).split()
        if len(guion_words) < 2:
            continue

        needle_len = min(10, len(guion_words))
        needle = guion_words[:needle_len]
        needle_bigrams = set(zip(needle, needle[1:]))

        best_score = 0.0
        best_word_pos = -1

        search_start = max(0, last_word_pos - 50)
        remaining_blocks = len(guion_blocks) - block_idx
        words_remaining = len(word_seq) - search_start
        max_search_window = max(
            500,
            int(words_remaining / max(1, remaining_blocks) * 3),
        )
        search_end = min(len(word_seq) - needle_len, search_start + max_search_window)

        for pos in range(search_start, search_end):
            window = [w for w, _ in word_seq[pos:pos + needle_len + 5]]
            if not window:
                continue

            window_set = set(window)
            unigram_hits = sum(1 for w in needle if w in window_set)
            unigram_score = unigram_hits / needle_len

            window_bigrams = set(zip(window, window[1:]))
            bigram_hits = len(needle_bigrams & window_bigrams)
            bigram_score = bigram_hits / max(1, len(needle_bigrams))

            first_word_bonus = 0.15 if window[0] == needle[0] else 0.0

            score = 0.35 * unigram_score + 0.50 * bigram_score + first_word_bonus

            if score > best_score:
                best_score = score
                best_word_pos = pos

            if score >= 0.75:
                break

        if best_word_pos >= 0 and best_score >= 0.25:
            seg_idx = word_seq[best_word_pos][1]
            matches.append({"speaker": block["speaker"], "seg_idx": seg_idx})
            block_word_count = len(guion_words)
            last_word_pos = best_word_pos + max(block_word_count // 2, 5)
            logger.debug(
                "Block %d [%s] matched seg %d (score=%.2f, word_pos=%d)",
                block_idx, block["speaker"], seg_idx, best_score, best_word_pos,
            )
        else:
            logger.debug(
                "Block %d [%s] NO match (best_score=%.2f)",
                block_idx, block["speaker"], best_score,
            )

        pct = 35 + int(40 * (block_idx + 1) / len(guion_blocks))
        if block_idx % 10 == 0:
            cb("diarize", f"Procesando bloque {block_idx+1}/{len(guion_blocks)}...", pct)

    cb("diarize", f"{len(matches)} turnos de speaker encontrados", 75)

    if not matches:
        raise RuntimeError(
            "No se pudo encontrar ningún match entre guion y transcripción. "
            "Verifica que guion.txt y la transcripción correspondan al mismo audio."
        )

    # ── 5. Corregir monotonía: seg_idx debe ser estrictamente creciente ──
    clean_matches: list[dict] = [matches[0]]
    for m in matches[1:]:
        if m["seg_idx"] > clean_matches[-1]["seg_idx"]:
            clean_matches.append(m)
        else:
            logger.debug("Descartado match no-monótono: %s seg=%d (prev=%d)",
                         m["speaker"], m["seg_idx"], clean_matches[-1]["seg_idx"])
    matches = clean_matches

    # ── 6. Construir segmentos con timestamps ──
    segments: list[dict] = []
    min_dur = cfg.get("min_segment_duration", 0.5)

    for i, m in enumerate(matches):
        start = ts_segments[m["seg_idx"]]["start"]
        if i + 1 < len(matches):
            end = ts_segments[matches[i + 1]["seg_idx"]]["start"]
        else:
            end = ts_segments[-1]["end"]

        duration = round(end - start, 3)
        if duration < min_dur:
            continue

        segments.append({
            "speaker": m["speaker"],
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": duration,
        })

    # ── 7. Merge segmentos consecutivos del mismo speaker ──
    merged: list[dict] = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["duration"] = round(merged[-1]["end"] - merged[-1]["start"], 3)
        else:
            merged.append(dict(seg))

    # ── 8. Subdividir segmentos muy largos usando gaps del transcript ──
    max_seg_duration = cfg.get("max_segment_duration", 120)
    min_gap_for_split = 1.5
    final: list[dict] = []

    for seg in merged:
        if seg["duration"] <= max_seg_duration:
            final.append(seg)
            continue
        sub_segments = _split_long_segment(seg, ts_segments, max_seg_duration, min_gap_for_split)
        final.extend(sub_segments)

    cb("diarize", f"{len(final)} segmentos de speaker (text-based)", 80)
    logger.info("Diarización texto: %d bloques guion → %d matches → %d segmentos finales",
                len(guion_blocks), len(matches), len(final))
    return final


def _split_long_segment(
    seg: dict,
    ts_segments: list[dict],
    max_duration: float,
    min_gap: float,
) -> list[dict]:
    """Subdivide un segmento largo usando gaps (silencios) en la transcripción."""
    speaker = seg["speaker"]
    seg_start = seg["start"]
    seg_end = seg["end"]

    gaps: list[tuple[float, float]] = []
    for i in range(len(ts_segments) - 1):
        t_end = ts_segments[i]["end"]
        t_start_next = ts_segments[i + 1]["start"]
        gap = t_start_next - t_end
        gap_center = (t_end + t_start_next) / 2
        if gap >= min_gap and seg_start < gap_center < seg_end:
            gaps.append((gap_center, gap))

    if not gaps:
        return [seg]

    gaps.sort(key=lambda g: -g[1])

    needed_cuts = max(1, int(seg["duration"] / max_duration))
    cut_points = sorted([g[0] for g in gaps[:needed_cuts * 2]])

    final_cuts: list[float] = []
    last_start = seg_start
    for cp in cut_points:
        if cp - last_start >= max_duration * 0.4:
            final_cuts.append(cp)
            last_start = cp

    if not final_cuts:
        final_cuts = [gaps[0][0]]

    boundaries = [seg_start] + final_cuts + [seg_end]
    result = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        dur = round(e - s, 3)
        if dur > 0.5:
            result.append({
                "speaker": speaker,
                "start": round(s, 3),
                "end": round(e, 3),
                "duration": dur,
            })

    return result if result else [seg]


# ── Audio-based providers ─────────────────────────────────────────────────────

def _run_diarization_openroute(audio_path: Path, cfg: dict, cb: ProgressCallback) -> list[dict]:
    """Diarización usando OpenRoute API con modelo de Whisper."""
    import requests

    # Preferir el key del cfg (ya inyectado desde env por load_settings),
    # con fallback explícito a la variable de entorno por compatibilidad.
    api_key = cfg.get("openroute_api_key") or os.getenv("OPENROUTE_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError(
            "OPENROUTE_API_KEY no configurada. "
            "Añádela al archivo .env o como variable de entorno."
        )
    
    cb("diarize", "Preparando audio para OpenRoute...", 10)
    
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    cb("diarize", "Enviando a OpenRoute API...", 30)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    files = {
        "file": ("audio.wav", audio_data, "audio/wav"),
    }
    
    import time
    
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(
            "https://openrouter.ai/api/v1/audio/transcriptions",
            headers=headers,
            files=files if attempt == 0 else {"file": ("audio.wav", open(audio_path, "rb").read(), "audio/wav")},
            data={
                "model": "openai/whisper-1",
                "language": "es",
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            },
            timeout=300,
        )
        
        if response.status_code == 200:
            break
            
        err_msg = f"API error {response.status_code}"
        cb("diarize", f"⚠️ OpenRoute {err_msg} (intento {attempt+1}/{max_retries})", 40)
        
        # Si es 413 Payload Too Large, cancelar y forzar fallback de inmediato
        if response.status_code == 413:
            raise RuntimeError("El archivo de audio es muy grande para OpenRoute (413).")
            
        if attempt < max_retries - 1:
            time.sleep(10)
        else:
            raise RuntimeError("Fallo definitivo OpenRoute (502/413).")
            
    try:
        result = response.json()
    except Exception as e:
        raise RuntimeError(f"JSON Parsing Error: {e}")
    
    segments = []
    if "segments" in result:
        current_speaker = "SPEAKER_01"
        prev_end = 0.0
        for i, seg in enumerate(result["segments"]):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            duration = end - start
            
            if duration < cfg.get("min_segment_duration", 0.5):
                continue
            
            if seg.get("speaker", ""):
                current_speaker = seg["speaker"]
            else:
                # Heurística: si hay una pausa larga (>2s) entre segmentos,
                # asumir cambio de speaker; sino mantener el mismo
                gap = start - prev_end
                if gap > 2.0 and segments:
                    prev_sp = segments[-1]["speaker"]
                    # Rotar entre hasta max_speakers speakers
                    max_sp = cfg.get("max_speakers", 5)
                    prev_num = int(prev_sp.replace("SPEAKER_", "")) if prev_sp.startswith("SPEAKER_") else 1
                    next_num = (prev_num % max_sp) + 1
                    current_speaker = f"SPEAKER_{next_num:02d}"
                elif segments:
                    current_speaker = segments[-1]["speaker"]
            
            prev_end = end
            segments.append({
                "speaker": current_speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
            })
    
    if not segments:
        cb("diarize", "No se detectaron segmentos, creando segmento por defecto...", 60)
        segments = [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 60.0, "duration": 60.0},
        ]
    
    cb("diarize", f"{len(segments)} segmentos procesados", 70)
    return segments


def _run_diarization_pyannote(audio_path: Path, cfg: dict, cb: ProgressCallback) -> list[dict]:
    """Diarización usando pyannote con HuggingFace."""
    from pyannote.audio import Pipeline
    import torch

    hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
    if not hf_token or hf_token == "your_huggingface_token_here":
        raise RuntimeError(
            "HUGGINGFACE_TOKEN no configurado en .env. "
            "Acepta los términos en https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    cb("diarize", "Cargando modelo pyannote...", 10)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    cb("diarize", f"Ejecutando diarización en {device}...", 25)

    diarization = pipeline(
        str(audio_path),
        min_speakers=cfg.get("min_speakers", 1),
        max_speakers=cfg.get("max_speakers", 5),
    )

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = round(turn.end - turn.start, 3)
        if duration < cfg.get("min_segment_duration", 0.5):
            continue
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": duration,
        })

    cb("diarize", f"{len(segments)} segmentos procesados", 70)
    return segments


def _dominant_speaker_in_range(segments: list[dict], start: float, end: float) -> str:
    """Calcula el speaker dominante en un rango temporal."""
    time_per_speaker = defaultdict(float)
    for seg in segments:
        overlap_start = max(seg["start"], start)
        overlap_end = min(seg["end"], end)
        if overlap_end > overlap_start:
            time_per_speaker[seg["speaker"]] += overlap_end - overlap_start
    if not time_per_speaker:
        return "UNKNOWN"
    return max(time_per_speaker, key=time_per_speaker.get)


def _speaker_turns_in_range(segments: list[dict], start: float, end: float) -> list[dict]:
    """Calcula los turnos de speaker en un rango temporal."""
    turns = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        turns.append({
            "speaker": seg["speaker"],
            "start": max(seg["start"], start),
            "end": min(seg["end"], end),
        })
    # Merge consecutive same-speaker turns (from segment splitting)
    merged: list[dict] = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"]:
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    return merged


# ── Speaker → Face Slot assignment (absorbido de calibrator) ─────────────

def _assign_speakers_to_face_slots(
    *,
    ep_dir: Path,
    segments: list[dict],
    settings: dict,
    on_progress: ProgressCallback,
) -> dict:
    """Asigna speakers a face slots por movimiento labial y genera speaker_zones.json."""
    calibration_dir = ep_dir / "calibration"
    face_slots = load_json(calibration_dir / "face_slots.json")

    speakers = sorted({seg["speaker"] for seg in segments})
    n_speakers = len(speakers)
    n_slots = len(face_slots)

    if n_speakers == 0 or n_slots == 0:
        logger.warning("Sin speakers (%d) o slots (%d) para asignar", n_speakers, n_slots)
        return {}

    video_path = find_video(ep_dir / "input")
    video_info = _get_video_info_diarizer(video_path)
    fps = video_info["fps"]

    # Seleccionar timestamps representativos por speaker
    speaker_timestamps = _select_speaker_timestamps(segments, frames_per_speaker=15)

    cap = cv2.VideoCapture(str(video_path))

    # Construir matriz de actividad labial
    lip_activity = _compute_lip_activity_matrix(
        cap=cap,
        fps=fps,
        face_slots=face_slots,
        speaker_timestamps=speaker_timestamps,
        speakers=speakers,
        on_progress=on_progress,
    )

    # Asignar speakers a slots
    slot_assignments = _assign_speakers_to_slots(
        lip_activity=lip_activity,
        speakers=speakers,
        face_slots=face_slots,
        segments=segments,
    )

    on_progress("diarize", "Speakers asignados a posiciones por movimiento labial", 92)

    # Construir speaker_zones
    crop_cfg = settings.get("crop", {})
    padding = crop_cfg.get("face_padding", 0.3)

    speaker_zones = {}
    for slot_idx, speaker in slot_assignments.items():
        slot = face_slots[slot_idx]
        speaker_zones[speaker] = {
            "center_x": round(slot["cx"], 4),
            "center_y": round(slot["cy"], 4),
            "face_width": round(slot["w"], 4),
            "face_height": round(slot["h"], 4),
            "crop_x": round(max(0, slot["cx"] - 0.5 * (slot["w"] * (1 + padding))), 4),
            "crop_y": round(max(0, slot["cy"] - 0.5 * (slot["h"] * (1 + padding))), 4),
            "detections": len(speaker_timestamps.get(speaker, [])),
        }

    # Speakers no asignados → default center
    for sp in speakers:
        if sp not in speaker_zones:
            speaker_zones[sp] = {
                "center_x": 0.5, "center_y": 0.4,
                "face_width": 0, "face_height": 0,
                "crop_x": 0.25, "crop_y": 0.1, "detections": 0,
            }

    cap.release()

    # Generar thumbnails por speaker
    thumbs_dir = calibration_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)
    for speaker, zone in speaker_zones.items():
        ts_list = speaker_timestamps.get(speaker, [])
        if ts_list:
            from .face_detector import _extract_thumbnail
            _extract_thumbnail(video_path, ts_list[0], thumbs_dir / f"{speaker}.jpg")

    zones_path = calibration_dir / "speaker_zones.json"
    save_json(zones_path, speaker_zones)
    on_progress("diarize", f"{len(speaker_zones)} speakers calibrados", 95)
    return speaker_zones


def _get_video_info_diarizer(video_path: Path) -> dict:
    """Obtiene info del video usando ffprobe."""
    from .face_detector import _get_video_info
    return _get_video_info(video_path)


def _select_speaker_timestamps(segments: list[dict], frames_per_speaker: int) -> dict[str, list[float]]:
    """Selecciona timestamps diversos para cada speaker."""
    speaker_segs = defaultdict(list)
    for seg in segments:
        speaker_segs[seg["speaker"]].append(seg)

    result = {}
    for speaker, segs in speaker_segs.items():
        segs.sort(key=lambda s: s["duration"], reverse=True)
        timestamps = []
        for seg in segs:
            dur = seg["end"] - seg["start"]
            inner_start = seg["start"] + dur * 0.15
            inner_end = seg["end"] - dur * 0.15
            if inner_end <= inner_start:
                inner_start = inner_end = (seg["start"] + seg["end"]) / 2

            if dur >= 60:
                for k in range(4):
                    t = inner_start + (inner_end - inner_start) * k / 3
                    timestamps.append(t)
            elif dur >= 20:
                timestamps.append(inner_start + (inner_end - inner_start) * 0.33)
                timestamps.append(inner_start + (inner_end - inner_start) * 0.66)
            else:
                timestamps.append((seg["start"] + seg["end"]) / 2)

            if len(timestamps) >= frames_per_speaker:
                break
        result[speaker] = timestamps[:frames_per_speaker]
    return result


def _compute_lip_activity_matrix(
    *,
    cap,
    fps: float,
    face_slots: list[dict],
    speaker_timestamps: dict[str, list[float]],
    speakers: list[str],
    on_progress: ProgressCallback,
) -> np.ndarray:
    """Construye matriz [n_speakers × n_slots] de actividad labial (MAD)."""
    n_speakers = len(speakers)
    n_slots = len(face_slots)
    activity = np.zeros((n_speakers, n_slots), dtype=np.float64)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delta_frames = max(1, int(0.2 * fps))

    for sp_idx, speaker in enumerate(speakers):
        timestamps = speaker_timestamps.get(speaker, [])
        if not timestamps:
            continue

        pair_mads = [[] for _ in range(n_slots)]

        for ts in timestamps:
            frame_num = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret1, frame1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + delta_frames)
            ret2, frame2 = cap.read()
            if not ret1 or not ret2:
                continue

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            for slot_idx, slot in enumerate(face_slots):
                cx_px = int(slot["cx"] * frame_w)
                half_w = int(slot["w"] * frame_w * 0.6)
                face_top = int((slot["cy"] - slot["h"] / 2) * frame_h)
                face_bot = int((slot["cy"] + slot["h"] / 2) * frame_h)
                mouth_top = face_top + int((face_bot - face_top) * 0.6)

                x1 = max(0, cx_px - half_w)
                x2 = min(frame_w, cx_px + half_w)
                y1 = max(0, mouth_top)
                y2 = min(frame_h, face_bot)

                if x2 <= x1 or y2 <= y1:
                    continue

                mouth1 = gray1[y1:y2, x1:x2]
                mouth2 = gray2[y1:y2, x1:x2]

                if mouth1.size == 0 or mouth2.size == 0:
                    continue

                mad = np.mean(np.abs(mouth1.astype(np.float32) - mouth2.astype(np.float32)))
                pair_mads[slot_idx].append(mad)

        for slot_idx in range(n_slots):
            if pair_mads[slot_idx]:
                activity[sp_idx, slot_idx] = np.median(pair_mads[slot_idx])

        pct = 87 + int(5 * (sp_idx + 1) / n_speakers)
        on_progress("diarize", f"Analizando labios de {speaker}...", pct)

    logger.info("Matriz de actividad labial:\n%s", activity)
    return activity


def _assign_speakers_to_slots(
    *,
    lip_activity: np.ndarray,
    speakers: list[str],
    face_slots: list[dict],
    segments: list[dict],
) -> dict[int, str]:
    """Asigna speakers a slots usando la matriz de actividad labial + Hungarian."""
    n_speakers = len(speakers)
    n_slots = len(face_slots)
    slot_assignments: dict[int, str] = {}

    row_max = lip_activity.max(axis=1)
    has_signal = np.any(row_max > 1.0)

    if has_signal and n_speakers <= n_slots:
        col_mean = lip_activity.mean(axis=0)
        col_std = lip_activity.std(axis=0)
        col_std[col_std < 0.01] = 1.0
        normalized = (lip_activity - col_mean) / col_std

        logger.info("Matriz normalizada:\n%s", normalized)

        try:
            from scipy.optimize import linear_sum_assignment
            cost = -normalized[:n_speakers, :n_slots]
            if n_speakers < n_slots:
                padding = np.zeros((n_slots - n_speakers, n_slots))
                cost = np.vstack([cost, padding])
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if r < n_speakers:
                    slot_assignments[c] = speakers[r]
        except ImportError:
            logger.info("scipy no disponible, usando greedy con normalización")
            assigned_slots = set()
            assigned_speakers = set()
            pairs = []
            for sp_idx in range(n_speakers):
                for sl_idx in range(n_slots):
                    pairs.append((normalized[sp_idx, sl_idx], sp_idx, sl_idx))
            pairs.sort(key=lambda x: -x[0])
            for _, sp_idx, sl_idx in pairs:
                if sp_idx in assigned_speakers or sl_idx in assigned_slots:
                    continue
                slot_assignments[sl_idx] = speakers[sp_idx]
                assigned_slots.add(sl_idx)
                assigned_speakers.add(sp_idx)
                if len(assigned_speakers) == n_speakers:
                    break

        if len(slot_assignments) < n_speakers:
            _fill_remaining_temporal(slot_assignments, speakers, face_slots, segments)
    else:
        logger.warning("Sin señal labial suficiente (max MAD=%.2f). Usando fallback temporal.",
                       row_max.max() if row_max.size else 0)
        _fill_remaining_temporal(slot_assignments, speakers, face_slots, segments)

    logger.info("Asignación final: %s", {v: face_slots[k]["cx"] for k, v in slot_assignments.items()})
    return slot_assignments


def _fill_remaining_temporal(
    slot_assignments: dict[int, str],
    speakers: list[str],
    face_slots: list[dict],
    segments: list[dict],
) -> None:
    """Asigna speakers no asignados a slots restantes por orden temporal."""
    assigned_speakers = set(slot_assignments.values())
    assigned_slots = set(slot_assignments.keys())
    remaining_speakers = [sp for sp in speakers if sp not in assigned_speakers]

    speaker_first_ts: dict[str, float] = {}
    for seg in segments:
        sp = seg["speaker"]
        if sp not in speaker_first_ts or seg["start"] < speaker_first_ts[sp]:
            speaker_first_ts[sp] = seg["start"]

    remaining_speakers.sort(key=lambda s: speaker_first_ts.get(s, float("inf")))
    remaining_slots = sorted(set(range(len(face_slots))) - assigned_slots)

    for sp, sl in zip(remaining_speakers, remaining_slots):
        slot_assignments[sl] = sp
