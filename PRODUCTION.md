# Guía de Despliegue en Producción

## Requisitos previos

- Python 3.12+
- PostgreSQL 14+ (o 16)
- Node.js 20+ (para compilar el frontend)
- FFmpeg en el PATH del sistema

---

## 1. Clonar y configurar el entorno Python

```powershell
# Desde la raíz del proyecto
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

---

## 2. Crear la base de datos PostgreSQL

```sql
-- Conectar como superusuario y ejecutar:
CREATE DATABASE script_editor;
CREATE USER script_editor_user WITH PASSWORD 'elige_contraseña_segura';
GRANT ALL PRIVILEGES ON DATABASE script_editor TO script_editor_user;
-- PostgreSQL 15+: también necesario:
\c script_editor
GRANT ALL ON SCHEMA public TO script_editor_user;
```

---

## 3. Configurar variables de entorno

```powershell
# Copiar la plantilla
Copy-Item .env.example .env
```

Editar `.env` con los valores reales:

```env
# ── Google Gemini ──────────────────────────────────────────────
GEMINI_API_KEY_1=tu_api_key_real_aqui
# GEMINI_API_KEY_2=segunda_key_opcional   # para rotación automática

# ── HuggingFace (pyannote diarización) ─────────────────────────
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx

# ── OpenRoute (diarización alternativa) ────────────────────────
OPENROUTE_API_KEY=sk-or-v1-xxxxxxxxxxxxx

# ── Base de datos ──────────────────────────────────────────────
DATABASE_URL=postgresql://script_editor_user:elige_contraseña_segura@localhost:5432/script_editor

# ── Seguridad JWT ──────────────────────────────────────────────
# Generar con: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=reemplaza_con_token_aleatorio_de_64_hex

# ── Entorno ────────────────────────────────────────────────────
APP_ENV=production

# ── CORS: solo dominios propios en producción ──────────────────
CORS_ORIGINS=https://tu-dominio.com
```

---

## 4. Ejecutar migraciones de base de datos

```powershell
# Verificar que DATABASE_URL está cargada (puede requerir activar .env manualmente)
$env:DATABASE_URL = "postgresql://script_editor_user:contraseña@localhost:5432/script_editor"

# Aplicar todas las migraciones hasta la versión más reciente
.venv\Scripts\alembic upgrade head

# Verificar estado de migraciones
.venv\Scripts\alembic current
.venv\Scripts\alembic history --verbose
```

> Para rollback a la versión anterior:
> ```powershell
> .venv\Scripts\alembic downgrade -1
> ```

---

## 5. Compilar el frontend

```powershell
cd frontend
npm install
npm run build          # genera frontend/dist/
cd ..
```

---

## 6. Arrancar el backend en producción

### Opción A — Script incluido (Windows)

```powershell
.venv\Scripts\python.exe run_prod.py
```

### Opción B — Uvicorn directo con workers

```powershell
# Un worker = threading interno del job_manager (no multiprocess)
.venv\Scripts\python.exe -m uvicorn backend.main:app `
    --host 0.0.0.0 `
    --port 8000 `
    --workers 1 `
    --log-level warning `
    --access-log
```

> **Importante:** mantener `--workers 1`. El job_manager usa un dict en memoria
> compartido por threads. Con múltiples workers (procesos separados) cada proceso
> tendría su propio `_jobs` dict y los estados de jobs quedarían inconsistentes.

---

## 7. Verificar que el servidor arrancó correctamente

```powershell
# Health check completo
Invoke-RestMethod http://localhost:8000/api/health | ConvertTo-Json

# La respuesta esperada en producción con BD conectada:
# {
#   "status": "ok",
#   "database": "ok",
#   "environment": "production",
#   "gemini_keys_configured": 1,
#   "openroute_configured": true
# }
```

Si `database` devuelve `"unavailable"`:
1. Verificar que PostgreSQL está corriendo: `Get-Service postgresql*`
2. Verificar `DATABASE_URL` en el `.env`
3. Verificar que la migración `alembic upgrade head` corrió sin errores

---

## 8. Rotación de migraciones (cuando se añaden modelos)

```powershell
# Generar nueva migración automáticamente desde cambios en los modelos ORM
.venv\Scripts\alembic revision --autogenerate -m "descripcion_del_cambio"

# Revisar el archivo generado en alembic/versions/ antes de aplicar
# Luego aplicar:
.venv\Scripts\alembic upgrade head
```

---

## 9. Monitoreo de logs

```powershell
# Logs del proceso uvicorn en tiempo real (si se arrancó en background)
# Redirigir la salida al arrancar:
.venv\Scripts\python.exe run_prod.py *>> logs\backend.log

# Tail de los últimos 100 registros
Get-Content logs\backend.log -Tail 100 -Wait
```

---

## 10. Variables de entorno — referencia completa

| Variable | Requerida | Descripción |
|---|---|---|
| `GEMINI_API_KEY_1` | Sí | API key de Google Gemini (análisis IA) |
| `GEMINI_API_KEY_2` | No | Segunda key para rotación automática |
| `GEMINI_API_KEY_3` | No | Tercera key para rotación automática |
| `HUGGINGFACE_TOKEN` | Si usa pyannote | Token HuggingFace para modelos de diarización |
| `OPENROUTE_API_KEY` | Si usa openroute | API key de OpenRouter (diarización alternativa) |
| `DATABASE_URL` | Sí | Cadena de conexión PostgreSQL |
| `JWT_SECRET_KEY` | Sí | Secreto para firmar tokens JWT (min 32 chars) |
| `APP_ENV` | Sí | `development` o `production` |
| `CORS_ORIGINS` | No | Orígenes CORS separados por coma (default: localhost:5173) |

---

## Solución de problemas frecuentes

### `ModuleNotFoundError: No module named 'psycopg2'`
```powershell
.venv\Scripts\pip install psycopg2-binary
```

### `alembic: command not found`
```powershell
# Usar la ruta completa dentro del venv
.venv\Scripts\alembic upgrade head
```

### Jobs quedan en estado "failed" al reiniciar
Comportamiento esperado: al arrancar, `recover_jobs_from_db()` detecta jobs que estaban
PENDING/RUNNING/PAUSED y los marca como FAILED porque sus threads de ejecución ya
no existen. Los jobs completados/cancelled se mantienen en historial.

### `status: degraded` en /api/health con `database: unavailable`
El backend sigue funcionando pero sin persistencia en BD. Los jobs se guardarán
solo en memoria (se pierden al reiniciar) y en los archivos `pipeline_status.json`.
Revisar la conexión a PostgreSQL.
