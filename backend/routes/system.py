"""Rutas de sistema: detección de hardware y configuración de procesamiento."""

import subprocess
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import yaml

from ..config import get_settings

router = APIRouter()

ROOT = Path(__file__).parent.parent.parent


def _detect_gpu() -> dict | None:
    """Detecta GPU NVIDIA vía nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None
        return {
            "name": parts[0],
            "vram_mb": int(float(parts[1])),
            "driver": parts[2],
            "compute_cap": parts[3],
        }
    except Exception:
        return None


def _detect_cpu() -> dict:
    """Detecta CPU."""
    import platform
    cpu_name = platform.processor() or "Unknown"
    try:
        import os
        cores = os.cpu_count() or 0
    except Exception:
        cores = 0
    return {"name": cpu_name, "cores": cores}


def _detect_nvenc() -> bool:
    """Verifica si ffmpeg soporta h264_nvenc."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def _detect_cuda_compute() -> bool:
    """Verifica si CUDA sirve para inferencia AI (compute cap >= 7.0)."""
    gpu = _detect_gpu()
    if not gpu:
        return False
    try:
        major = int(gpu["compute_cap"].split(".")[0])
        return major >= 7
    except Exception:
        return False


@router.get("/hardware")
async def detect_hardware():
    """Detecta hardware disponible y capacidades."""
    gpu = _detect_gpu()
    cpu = _detect_cpu()
    nvenc = _detect_nvenc()
    cuda_ai = _detect_cuda_compute()

    recommendations = []
    if nvenc:
        recommendations.append({
            "component": "ffmpeg_encoding",
            "recommendation": "gpu",
            "reason": "NVENC ~2.5x más rápido que libx264",
        })
    if cuda_ai:
        recommendations.append({
            "component": "whisper",
            "recommendation": "gpu",
            "reason": "CUDA disponible para inferencia Whisper",
        })
    else:
        recommendations.append({
            "component": "whisper",
            "recommendation": "cpu",
            "reason": "GPU no compatible con CUDA AI (requiere Compute >= 7.0)" if gpu else "Sin GPU detectada",
        })

    return {
        "cpu": cpu,
        "gpu": gpu,
        "capabilities": {
            "nvenc": nvenc,
            "cuda_ai": cuda_ai,
        },
        "recommendations": recommendations,
    }


class ProcessingConfig(BaseModel):
    device: str = "auto"           # auto | cpu | gpu
    ffmpeg_encoder: str = "auto"   # auto | libx264 | h264_nvenc
    ffmpeg_preset: str = "auto"    # auto | slow | medium | p4 | p7 etc.


@router.get("/processing-config")
async def get_processing_config():
    """Lee la configuración actual de procesamiento."""
    settings = get_settings()
    processing = settings._yaml.get("processing", {})
    return {
        "device": processing.get("device", "auto"),
        "ffmpeg_encoder": processing.get("ffmpeg_encoder", "auto"),
        "ffmpeg_preset": processing.get("ffmpeg_preset", "auto"),
    }


@router.put("/processing-config")
async def update_processing_config(config: ProcessingConfig):
    """Actualiza la configuración de procesamiento en settings.yaml."""
    yaml_path = ROOT / "config" / "settings.yaml"
    if not yaml_path.exists():
        raise HTTPException(404, "settings.yaml no encontrado")

    content = yaml_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content) or {}

    data["processing"] = {
        "device": config.device,
        "ffmpeg_encoder": config.ffmpeg_encoder,
        "ffmpeg_preset": config.ffmpeg_preset,
    }

    yaml_path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # Invalidar cache de settings para que se recargue
    from ..config import get_settings as _gs
    _gs.cache_clear()

    return {"status": "updated", "config": data["processing"]}
