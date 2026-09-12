"""Ruta para validar modelos de IA disponibles y gestionar modelos custom."""

import asyncio
import os
import sys
import time as _time
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

router = APIRouter()

# ── Cache de resultados (TTL 5 minutos) ──────────────────────────────────────
_cache: dict | None = None
_cache_ts: float = 0
_CACHE_TTL = 300  # segundos

SETTINGS_PATH = ROOT / "config" / "settings.yaml"


# ── Modelos Pydantic ─────────────────────────────────────────────────────────

class CustomModelCreate(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    model_id: str
    provider_type: str = "openai_compatible"


# ── Test helpers ─────────────────────────────────────────────────────────────

_TRANSIENT_NET_MARKERS = (
    "handshake", "timeout", "timed out", "connection reset", "ecconnreset",
    "connection aborted", "temporarily unavailable", "503", "504", "502",
    "ssl", "_ssl.c", "getaddrinfo",
)


def _is_transient_net_error(err_str: str) -> bool:
    lo = err_str.lower()
    return any(m in lo for m in _TRANSIENT_NET_MARKERS)


def _test_gemini_model(model_name: str, api_keys: list[str]) -> dict:
    """Prueba un modelo Gemini con el nuevo SDK google.genai.

    Reintenta hasta 2 veces en errores transitorios de red (SSL handshake,
    connection reset, 502/503/504). Si falla definitivamente por cuota
    (429), no reintenta.
    """
    status: dict = {"model": model_name, "provider": "gemini", "available": False, "error": None, "response_time_ms": None}
    if not api_keys:
        status["error"] = "Sin API keys configuradas"
        return status
    try:
        from google import genai
    except ImportError:
        status["error"] = "google-genai no instalado"
        return status

    last_err = ""
    last_elapsed = 0
    for api_key in api_keys:
        for attempt in range(2):
            try:
                t0 = _time.monotonic()
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model_name, contents="Test: responde OK"
                )
                elapsed_ms = round((_time.monotonic() - t0) * 1000)
                if response.text and len(response.text.strip()) > 0:
                    status["available"] = True
                    status["response_time_ms"] = elapsed_ms
                    status["error"] = None
                    return status
                last_err = "Respuesta vacía"
                last_elapsed = elapsed_ms
                break  # no sirve reintentar con la misma key
            except Exception as e:
                err_str = str(e)
                last_elapsed = round((_time.monotonic() - t0) * 1000)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    last_err = "Cuota agotada (429 RESOURCE_EXHAUSTED)"
                    break  # con esta key ya no, probar la siguiente
                if _is_transient_net_error(err_str) and attempt == 0:
                    _time.sleep(1.5)
                    continue  # reintentar
                last_err = err_str[:200]
                break
    status["error"] = last_err or "Sin respuesta"
    status["response_time_ms"] = last_elapsed or None
    return status


def _test_custom_model(config: dict) -> dict:
    """Prueba un modelo custom vía API OpenAI-compatible."""
    status: dict = {
        "model": config["name"],
        "provider": "custom",
        "available": False,
        "error": None,
        "response_time_ms": None,
    }
    try:
        import httpx
    except ImportError:
        status["error"] = "httpx no instalado"
        return status

    base_url = config["base_url"].rstrip("/")
    api_key = config.get("api_key", "")
    model_id = config["model_id"]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Test: responde OK"}],
        "max_tokens": 10,
    }

    t0 = _time.monotonic()
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            elapsed_ms = round((_time.monotonic() - t0) * 1000)
            if text.strip():
                status["available"] = True
                status["response_time_ms"] = elapsed_ms
            else:
                status["error"] = "Respuesta vacía"
                status["response_time_ms"] = elapsed_ms
    except Exception as e:
        status["error"] = str(e)[:200]
        status["response_time_ms"] = round((_time.monotonic() - t0) * 1000)
    return status


# ── Endpoints de status ──────────────────────────────────────────────────────

@router.get("/status")
async def get_models_status():
    """Prueba cada modelo configurado y devuelve cuáles están disponibles."""
    global _cache, _cache_ts

    if _cache is not None and (_time.monotonic() - _cache_ts) < _CACHE_TTL:
        return _cache

    from dotenv import load_dotenv
    from scripts.core.utils import load_settings

    load_dotenv(ROOT / ".env")
    settings = load_settings(ROOT)
    cfg = settings.get("analysis", {})

    gemini_models = cfg.get("models", [])
    custom_models = cfg.get("custom_models", []) or []

    api_keys: list[str] = []
    for i in range(1, 4):
        key = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if key and key != "your_gemini_api_key_here":
            api_keys.append(key)

    loop = asyncio.get_running_loop()

    # Serializamos los tests con un semáforo pequeño: si el transcribe (u otro
    # paso pesado) satura la red, N llamadas TLS concurrentes provocan SSL
    # handshake timeouts falsos. Con concurrency=2 evitamos ese ruido.
    sem = asyncio.Semaphore(2)

    async def _guarded(fn, *args):
        async with sem:
            return await asyncio.wait_for(
                loop.run_in_executor(None, fn, *args),
                timeout=45,  # margen amplio para SSL + generate_content + retry
            )

    tasks = [_guarded(_test_gemini_model, m, api_keys) for m in gemini_models]
    for cm in custom_models:
        resolved = dict(cm)
        key_env = resolved.get("api_key_env", "")
        if key_env:
            resolved["api_key"] = os.getenv(key_env, resolved.get("api_key", ""))
        tasks.append(_guarded(_test_custom_model, resolved))

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    all_model_names = list(gemini_models) + [cm.get("name", "custom") for cm in custom_models]
    all_providers = ["gemini"] * len(gemini_models) + ["custom"] * len(custom_models)

    for i, res in enumerate(raw_results):
        if isinstance(res, dict):
            results.append(res)
        elif isinstance(res, asyncio.TimeoutError):
            results.append({
                "model": all_model_names[i],
                "provider": all_providers[i],
                "available": False,
                "error": "Timeout: el modelo no respondió en 45s",
                "response_time_ms": 45000,
            })
        else:
            results.append({
                "model": all_model_names[i],
                "provider": all_providers[i],
                "available": False,
                "error": str(res)[:200],
                "response_time_ms": None,
            })

    response = {
        "models": results,
        "gemini_keys_count": len(api_keys),
        "default_model": cfg.get("default_model", "gemini-2.5-flash"),
    }

    _cache = response
    _cache_ts = _time.monotonic()
    return response


@router.post("/refresh")
async def refresh_models_cache():
    """Fuerza re-validación de modelos limpiando la cache."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0
    return await get_models_status()


# ── Endpoints CRUD para modelos custom ───────────────────────────────────────

def _load_settings_yaml() -> dict:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_settings_yaml(data: dict):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


@router.get("/custom")
async def list_custom_models():
    """Lista modelos custom registrados."""
    settings = _load_settings_yaml()
    custom = settings.get("analysis", {}).get("custom_models", []) or []
    return {"custom_models": custom}


@router.post("/custom")
async def add_custom_model(model: CustomModelCreate):
    """Registra un nuevo modelo custom."""
    global _cache, _cache_ts

    settings = _load_settings_yaml()
    analysis = settings.setdefault("analysis", {})
    custom_models = analysis.get("custom_models") or []

    existing_names = {cm.get("name") for cm in custom_models}
    if model.name in existing_names:
        raise HTTPException(status_code=409, detail=f"Modelo '{model.name}' ya existe")

    new_model = {
        "name": model.name,
        "base_url": model.base_url,
        "model_id": model.model_id,
        "provider_type": model.provider_type,
    }
    if model.api_key:
        env_var = f"CUSTOM_API_KEY_{model.name.upper().replace(' ', '_').replace('-', '_')}"
        new_model["api_key_env"] = env_var
        _update_env_var(env_var, model.api_key)

    custom_models.append(new_model)
    analysis["custom_models"] = custom_models
    _save_settings_yaml(settings)

    _cache = None
    _cache_ts = 0
    return {"status": "ok", "model": new_model}


@router.delete("/custom/{name}")
async def delete_custom_model(name: str):
    """Elimina un modelo custom registrado."""
    global _cache, _cache_ts

    settings = _load_settings_yaml()
    analysis = settings.setdefault("analysis", {})
    custom_models = analysis.get("custom_models") or []

    original_len = len(custom_models)
    custom_models = [cm for cm in custom_models if cm.get("name") != name]

    if len(custom_models) == original_len:
        raise HTTPException(status_code=404, detail=f"Modelo '{name}' no encontrado")

    analysis["custom_models"] = custom_models
    _save_settings_yaml(settings)

    _cache = None
    _cache_ts = 0
    return {"status": "deleted", "name": name}


def _update_env_var(key: str, value: str):
    """Actualiza o añade una variable en .env."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
