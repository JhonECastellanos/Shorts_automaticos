"""Clasificador de tendencia/viralidad heurístico para shorts.

DEPRECADO: La lógica de ranking ahora está integrada directamente en analyzer.py
(función _rank_moments). Este módulo se conserva como referencia standalone.
"""

import re
from pathlib import Path

from .utils import (
    ProgressCallback, noop_progress, get_episode_dir, load_json, save_json,
)

# Palabras clave que aumentan el potencial viral por categoría
VIRAL_KEYWORDS = {
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

HOOK_PATTERNS = [
    (r"^\?|¿", 3.0),                          # Empieza con pregunta
    (r"sabías que|lo que nadie te dice", 2.5), # Hook curioso
    (r"error|no hagas|deja de", 2.0),          # Advertencia
    (r"la verdad sobre|te explico", 1.5),      # Revelador
]


def classify_trends(
    *,
    project: str,
    episode: str,
    root: Path,
    on_progress: ProgressCallback = noop_progress,
) -> list[dict]:
    ep_dir = get_episode_dir(root, project, episode, create=False)
    analysis_dir = ep_dir / "analysis"

    moments_path = analysis_dir / "moments.json"
    if not moments_path.exists():
        raise FileNotFoundError("No se encontró moments.json. Ejecuta analyze primero.")

    moments = load_json(moments_path)
    on_progress("trend", f"Clasificando {len(moments)} momentos", 10)

    ranked = []
    for i, m in enumerate(moments):
        text_content = f"{m.get('hook', '')} {m.get('topic', '')} {m.get('reason', '')}".lower()

        # Score por keywords
        keyword_score = 0
        categories_hit = []
        for category, keywords in VIRAL_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_content:
                    keyword_score += 1.5
                    categories_hit.append(category)
                    break

        # Score por hook patterns
        hook_score = 0
        hook_text = m.get("hook", "").lower()
        for pattern, pts in HOOK_PATTERNS:
            if re.search(pattern, hook_text):
                hook_score += pts

        # Score por duración óptima (40-60s es ideal para shorts)
        duration = m.get("end", 0) - m.get("start", 0)
        dur_score = 0
        if 35 <= duration <= 65:
            dur_score = 2.0
        elif 25 <= duration <= 75:
            dur_score = 1.0

        # Score base de Gemini
        base_score = m.get("score", 5)

        trend_score = round(base_score * 0.4 + keyword_score * 0.25 + hook_score * 0.2 + dur_score * 0.15, 2)

        ranked.append({
            **m,
            "trend_score": trend_score,
            "categories": list(set(categories_hit)),
            "keyword_score": round(keyword_score, 2),
            "hook_score": round(hook_score, 2),
            "duration_score": round(dur_score, 2),
            "optimal_duration": 35 <= duration <= 65,
        })

        pct = 10 + ((i + 1) / len(moments)) * 80
        on_progress("trend", f"Momento {i + 1}: trend_score={trend_score}", pct)

    ranked.sort(key=lambda x: x["trend_score"], reverse=True)

    # Asignar ranking
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank

    out_path = analysis_dir / "moments_ranked.json"
    save_json(out_path, ranked)
    on_progress("trend", f"Ranking guardado: {len(ranked)} momentos", 100)
    return ranked
