"""FastAPI backend — pipeline de video a shorts."""

import logging
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .db import check_connection, create_tables, init_engine
from .routes import calibration, jobs, logs, models, shorts, system, transcripts, videos

logger = logging.getLogger(__name__)

# ── Aplicación ────────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="Script Editor API",
    version="0.1.0",
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
)

# ── Middleware: CORS ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-Correlation-ID"],
)

# ── Middleware: Correlation ID ────────────────────────────────────────────────

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Propaga o genera un X-Correlation-ID para trazabilidad en cada request."""
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:8]
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response

# ── Handler global de excepciones ────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", "—")
    logger.exception(
        "Error no manejado [%s] %s %s",
        correlation_id,
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor. Revisa los logs para más detalles.",
            "correlation_id": correlation_id,
        },
    )

# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    """Inicializa BD y valida configuración al arrancar."""
    # Suprimir ConnectionResetError ruidoso de Windows ProactorEventLoop
    import asyncio
    loop = asyncio.get_event_loop()
    _default_handler = loop.get_exception_handler()

    def _quiet_handler(lp: asyncio.AbstractEventLoop, ctx: dict):
        exc = ctx.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            return  # Normal en Windows cuando el browser cancela range requests
        if _default_handler:
            _default_handler(lp, ctx)
        else:
            lp.default_exception_handler(ctx)

    loop.set_exception_handler(_quiet_handler)

    # Validar configuración y emitir advertencias
    warnings = settings.validate_for_startup()
    for warn in warnings:
        logger.warning("CONFIG: %s", warn)

    # Inicializar motor de BD y crear tablas si no existen
    try:
        init_engine(settings.database_url)
        create_tables()
        # Recuperar jobs persistidos del arranque anterior
        from .services.job_manager import recover_jobs_from_db
        recovered = recover_jobs_from_db()
        logger.info("Base de datos lista — %d jobs recuperados", recovered)
    except Exception as exc:
        logger.error("No se pudo inicializar la BD: %s", exc)
        logger.warning(
            "La API continuará sin persistencia en BD. "
            "Verifica DATABASE_URL en el archivo .env"
        )

    logger.info(
        "Script Editor API iniciada — entorno=%s cors=%s",
        settings.app_env,
        settings.cors_origins_list,
    )

# ── Rutas ─────────────────────────────────────────────────────────────────────

app.include_router(jobs.router,        prefix="/api/jobs",        tags=["jobs"])
app.include_router(shorts.router,      prefix="/api/shorts",      tags=["shorts"])
app.include_router(transcripts.router, prefix="/api/transcripts", tags=["transcripts"])
app.include_router(videos.router,      prefix="/api/videos",      tags=["videos"])
app.include_router(logs.router,        prefix="/api/logs",        tags=["logs"])
app.include_router(models.router,      prefix="/api/models",      tags=["models"])
app.include_router(system.router,      prefix="/api/system",      tags=["system"])
app.include_router(calibration.router, prefix="/api/calibration", tags=["calibration"])

# ── Archivos estáticos ────────────────────────────────────────────────────────

projects_dir = ROOT / "projects"
projects_dir.mkdir(exist_ok=True)
app.mount("/projects", StaticFiles(directory=str(projects_dir)), name="projects")

_public_root = settings.output_cfg().get("root")
if _public_root:
    _ocamo_dir = Path(_public_root)
    _ocamo_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/ocamo", StaticFiles(directory=str(_ocamo_dir)), name="ocamo")

# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["health"])
async def health():
    """Health check extendido: BD y configuración básica."""
    db_ok = check_connection()
    gemini_keys = len(settings.get_gemini_api_keys())
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "unavailable",
        "environment": settings.app_env,
        "gemini_keys_configured": gemini_keys,
    }

# ── SPA fallback ──────────────────────────────────────────────────────────────

_dist = ROOT / "frontend" / "dist"
_index = _dist / "index.html"


@app.get("/{full_path:path}", include_in_schema=False)
async def spa(full_path: str):
    if not _index.exists():
        return JSONResponse(status_code=503, content={"detail": "Frontend no compilado."})
    candidate = (_dist / full_path).resolve()
    try:
        candidate.relative_to(_dist.resolve())
    except ValueError:
        return FileResponse(_index)
    return FileResponse(candidate) if candidate.is_file() else FileResponse(_index)
