# Shorts Automáticos — Pipeline de edición de video vertical

Pipeline automatizado que convierte episodios largos (podcasts, entrevistas, grabaciones con cámara fija) en clips verticales listos para redes sociales.

## Cómo funciona

Flujo por etapas (cada una ejecutable por separado o en cadena):

1. **Descarga** — descarga el video fuente desde una URL.
2. **Transcripción** — genera subtítulos con sincronización por palabra (Whisper, español).
3. **Análisis** — detecta los momentos más relevantes del contenido usando un modelo de lenguaje (Gemini) para puntuar cada segmento.
4. **Diarización** — identifica qué locutor habla en cada tramo (pyannote + seguimiento facial).
5. **Detección de voz activa y reencuadre** — decide qué cámara/enfoque usar en cada momento y recorta el video a formato vertical (1080×1920).
6. **Exportación** — genera el video final (FFmpeg) y, opcionalmente, un guion FCPXML para edición profesional.

## Stack

- **Backend:** Python 3.12 · FastAPI · SQLAlchemy · Alembic · Pydantic Settings
- **Procesamiento:** Whisper (transcripción), pyannote (diarización), MediaPipe (seguimiento facial), FFmpeg (recorte y exportación)
- **Análisis de contenido:** Google Gemini (los modelos y parámetros se configuran en `config/settings.yaml`)
- **Frontend:** React + Vite + TypeScript

## Requisitos

- Python 3.12+
- PostgreSQL 14+ (o superior)
- Node.js 20+ (para el frontend)
- FFmpeg en el PATH del sistema

## Configuración rápida

```powershell
# 1. Clonar y preparar el entorno Python
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Variables de entorno (API keys, base de datos, JWT)
Copy-Item .env.example .env
```

Las API keys y secretos se leen **exclusivamente** de variables de entorno o del archivo `.env` (nunca del código ni de `config/settings.yaml`). Consulta `.env.example` para las variables soportadas.

## Uso

### Ejecutar el pipeline completo

```powershell
python scripts/pipeline.py run --project MI_PROYECTO --episode ep01 --url https://youtu.be/...
```

### Ejecutar una etapa suelta

Cada script en `scripts/` es autónomo: `01_download.py`, `02_transcribe.py`, `03_analyze.py`, `04_diarize.py`, `06_export.py`, `07_fcpxml.py`.

### Frontend (opcional)

```powershell
cd frontend
npm install
npm run dev
```

### Backend (API)

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

Endpoints disponibles en `backend/routes/`: `shorts`, `videos`, `jobs`, `transcripts`, `logs`, `models`, `system` y `calibration`.

## Configuración

- `config/settings.yaml` — parámetros no sensibles (modelos de análisis, diarización, recorte, exportación, cambio de cámara).
- `.env` — secretos y credenciales (API keys, `DATABASE_URL`, `JWT_SECRET_KEY`, `APP_ENV`, `CORS_ORIGINS`).
- `alembic/` — migraciones de la base de datos.

## Despliegue en producción

Guía completa en [`PRODUCTION.md`](PRODUCTION.md): creación de la base de datos, migraciones, compilación del frontend y arranque del backend (Windows o uvicorn directo).

## Pruebas del frontend

```powershell
cd frontend
npm test
```

## Organización del código

- `scripts/core/` — módulos del pipeline (`downloader`, `transcriber`, `analyzer`, `diarizer`, `asd`, `face_tracker`, `camera_switcher`, `editor`, `exporter`, `fcpxml_gen`).
- `backend/` — API REST + gestión de jobs.
- `frontend/src/` — panel de control (estado de procesamiento, transcripción, configuración, lista de shorts).