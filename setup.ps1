# setup.ps1 — Configura el entorno completo del proyecto
# Ejecutar UNA VEZ después de instalar Python 3.11 y ffmpeg
#
# USO: .\setup.ps1
# Desde PowerShell en la carpeta del proyecto:

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot

Write-Host ""
Write-Host "=== Script Editor — Setup ===" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar Python
Write-Host "[1/5] Verificando Python..." -ForegroundColor Yellow
$python = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
if (-not $python) {
    Write-Host "ERROR: Python no encontrado." -ForegroundColor Red
    Write-Host "Descarga Python 3.11 desde https://www.python.org/downloads/release/python-3119/" -ForegroundColor White
    Write-Host "IMPORTANTE: marca 'Add Python to PATH' durante la instalacion." -ForegroundColor Yellow
    exit 1
}
$version = & $python.Source --version
Write-Host "  OK: $version" -ForegroundColor Green

# 2. Verificar ffmpeg
Write-Host "[2/5] Verificando ffmpeg..." -ForegroundColor Yellow
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "  AVISO: ffmpeg no encontrado en PATH." -ForegroundColor Yellow
    Write-Host "  Descarga desde: https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-full.7z)" -ForegroundColor White
    Write-Host "  Extrae y agrega la carpeta bin al PATH del sistema." -ForegroundColor White
    Write-Host "  El pipeline no funcionara sin ffmpeg." -ForegroundColor Yellow
} else {
    $ffver = & ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "  OK: $ffver" -ForegroundColor Green
}

# 3. Crear entorno virtual
Write-Host "[3/5] Creando entorno virtual (.venv)..." -ForegroundColor Yellow
if (Test-Path "$ROOT\.venv") {
    Write-Host "  Ya existe, omitiendo." -ForegroundColor DarkGray
} else {
    & $python.Source -m venv "$ROOT\.venv"
    Write-Host "  OK: .venv creado" -ForegroundColor Green
}

# 4. Instalar dependencias
Write-Host "[4/5] Instalando dependencias (esto puede tardar varios minutos)..." -ForegroundColor Yellow
$pip = "$ROOT\.venv\Scripts\pip.exe"
& $pip install --upgrade pip --quiet
& $pip install -r "$ROOT\requirements.txt"
Write-Host "  OK: dependencias instaladas" -ForegroundColor Green

# 5. Copiar .env si no existe
Write-Host "[5/5] Configurando .env..." -ForegroundColor Yellow
if (Test-Path "$ROOT\.env") {
    Write-Host "  Ya existe .env, no se sobreescribe." -ForegroundColor DarkGray
} else {
    Copy-Item "$ROOT\.env.example" "$ROOT\.env"
    Write-Host "  OK: .env creado desde .env.example" -ForegroundColor Green
    Write-Host "  ACCION REQUERIDA: edita .env y agrega tus API keys" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup completado ===" -ForegroundColor Green
Write-Host ""
Write-Host "Proximos pasos:" -ForegroundColor Cyan
Write-Host "  1. Edita .env con tus API keys (GEMINI_API_KEY y HUGGINGFACE_TOKEN)"
Write-Host "  2. Activa el entorno: .\.venv\Scripts\Activate.ps1"
Write-Host "  3. Ejecuta el pipeline:"
Write-Host "     python scripts\pipeline.py run --project mi_podcast --episode ep01 --url https://youtu.be/..."
Write-Host ""
Write-Host "Referencia rapida de pasos:  download | transcribe | analyze | diarize | calibrate | export | fcpxml"
