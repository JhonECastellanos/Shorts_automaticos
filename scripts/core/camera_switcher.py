"""Genera segmentos de cámara por speaker para cambio dinámico con zoom."""

from __future__ import annotations

import random


def compute_camera_segments(
    speaker_turns: list[dict],
    speaker_zones: dict,
    start: float,
    end: float,
    camera_cfg: dict,
) -> list[dict]:
    """Genera segmentos de cámara a partir de speaker_turns dentro de [start, end].

    Cada segmento resultante contiene:
        speaker, start, end, center_x, center_y
    """
    min_seg = camera_cfg.get("min_segment_sec", 1.5)
    max_speakers = camera_cfg.get("max_speakers", 5)

    # 1. Filtrar turns que caen dentro del rango del short
    clipped: list[dict] = []
    for turn in speaker_turns:
        t_start = turn.get("start", 0)
        t_end = turn.get("end", 0)
        # Saltar turns fuera de rango
        if t_end <= start or t_start >= end:
            continue
        clipped.append({
            "speaker": turn["speaker"],
            "start": max(t_start, start),
            "end": min(t_end, end),
        })

    if not clipped:
        # Sin datos → un solo segmento fallback
        return [_fallback_segment(start, end)]

    # 2. Merge turnos consecutivos del mismo speaker
    merged: list[dict] = [clipped[0].copy()]
    for turn in clipped[1:]:
        prev = merged[-1]
        if turn["speaker"] == prev["speaker"]:
            prev["end"] = turn["end"]
        else:
            merged.append(turn.copy())

    # 3. Manejar interrupciones rápidas antes de absorber cortos
    merged = _handle_interruptions(merged, min_seg, camera_cfg, start)

    # 4. Absorber segmentos muy cortos en el vecino más cercano
    filtered: list[dict] = []
    for seg in merged:
        dur = seg["end"] - seg["start"]
        if dur < min_seg and filtered:
            # Extender el segmento anterior
            filtered[-1]["end"] = seg["end"]
        else:
            filtered.append(seg)

    if not filtered:
        return [_fallback_segment(start, end)]

    # 4b. Limitar a max_speakers distintos (mantener los más frecuentes)
    unique_speakers = list({seg["speaker"] for seg in filtered})
    if len(unique_speakers) > max_speakers:
        from collections import Counter
        speaker_dur = Counter()
        for seg in filtered:
            speaker_dur[seg["speaker"]] += seg["end"] - seg["start"]
        top_speakers = {sp for sp, _ in speaker_dur.most_common(max_speakers)}
        # Re-merge: reemplazar speakers minoritarios por el anterior
        limited: list[dict] = []
        for seg in filtered:
            if seg["speaker"] in top_speakers:
                if limited and limited[-1]["speaker"] == seg["speaker"]:
                    limited[-1]["end"] = seg["end"]
                else:
                    limited.append(seg.copy())
            elif limited:
                limited[-1]["end"] = seg["end"]
            else:
                seg_copy = seg.copy()
                # Asignar al speaker top con más duración
                seg_copy["speaker"] = speaker_dur.most_common(1)[0][0]
                limited.append(seg_copy)
        filtered = limited if limited else filtered

    # 5. Asignar center_x / center_y desde speaker_zones
    result: list[dict] = []
    for seg in filtered:
        zone = speaker_zones.get(seg["speaker"], {})
        result.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "center_x": zone.get("center_x", 0.5),
            "center_y": zone.get("center_y", 0.4),
        })

    return result


def _handle_interruptions(
    segments: list[dict],
    min_seg: float,
    camera_cfg: dict,
    seed_base: float,
) -> list[dict]:
    """Detecta interrupciones rápidas (3+ speakers en ventana corta) y las redistribuye.

    En vez de absorber todos los segmentos cortos en el anterior, divide los
    grupos de interrupción en sub-segmentos de ~2s asignando speakers
    aleatoriamente entre los participantes.
    """
    interrupt_window = camera_cfg.get("interrupt_window_sec", 5.0)
    min_interrupt_speakers = camera_cfg.get("min_interrupt_speakers", 3)
    sub_seg_dur = camera_cfg.get("interrupt_sub_segment_sec", 2.0)

    if len(segments) < min_interrupt_speakers:
        return segments

    result: list[dict] = []
    i = 0
    while i < len(segments):
        # Buscar ventana de interrupciones desde i
        j = i + 1
        while j < len(segments) and (segments[j]["end"] - segments[i]["start"]) <= interrupt_window:
            j += 1

        group = segments[i:j]
        group_speakers = {seg["speaker"] for seg in group}

        if len(group_speakers) >= min_interrupt_speakers and len(group) >= min_interrupt_speakers:
            # Grupo de interrupciones detectado: redistribuir
            group_start = group[0]["start"]
            group_end = group[-1]["end"]
            speakers_list = sorted(group_speakers)
            rng = random.Random(seed_base + group_start)

            t = group_start
            while t < group_end:
                seg_end = min(t + sub_seg_dur, group_end)
                chosen = rng.choice(speakers_list)
                if result and result[-1]["speaker"] == chosen:
                    result[-1]["end"] = seg_end
                else:
                    result.append({"speaker": chosen, "start": t, "end": seg_end})
                t = seg_end

            i = j
        else:
            result.append(segments[i])
            i += 1

    return result


def _fallback_segment(start: float, end: float) -> dict:
    return {
        "speaker": None,
        "start": start,
        "end": end,
        "center_x": 0.5,
        "center_y": 0.4,
    }
