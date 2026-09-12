"""Script de arranque producción: uvicorn para el backend.

Uso:
    .venv\\Scripts\\python.exe run_prod.py [--port 8000] [--host 0.0.0.0]

Variables de entorno requeridas (ver .env.example o PRODUCTION.md):
    DATABASE_URL, JWT_SECRET_KEY, APP_ENV=production, GEMINI_API_KEY_1

Guía completa de despliegue: PRODUCTION.md
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

# Forzar UTF-8 en stdout/stderr para que los mensajes con acentos o símbolos
# no revienten en consolas Windows con codepage cp1252.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Silenciar el warning de loky/joblib: en Windows 11 `wmic` fue removido y joblib
# intenta usarlo para contar cores físicos. Seteamos la env var antes de cualquier
# import de sklearn/joblib para evitar el subprocess fallido + UserWarning ruidoso.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

ROOT = Path(__file__).parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

# En Linux/macOS el ejecutable está en bin/, no en Scripts/
if not PYTHON.exists():
    PYTHON = ROOT / ".venv" / "bin" / "python"


def _free_port(port: int) -> None:
    """Si el puerto está ocupado, mata el proceso que lo usa."""
    # Test rápido
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return  # libre
        except OSError:
            pass  # ocupado

    print(f"⚠️  Puerto {port} ocupado — liberando...")

    if sys.platform == "win32":
        # netstat para encontrar PID
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "TCP"], text=True, stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                    pid = int(parts[4])
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True)
                    print(f"   Proceso PID {pid} terminado.")
                    break
        except Exception as e:
            print(f"   No se pudo liberar el puerto automáticamente: {e}")
    else:
        # Linux/macOS: lsof
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL,
            )
            for pid_str in out.strip().split():
                os.kill(int(pid_str), signal.SIGTERM)
                print(f"   Proceso PID {pid_str} terminado.")
        except Exception:
            pass

    # Esperar a que se libere
    import time
    for _ in range(10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                print(f"   Puerto {port} liberado.\n")
                return
            except OSError:
                time.sleep(0.5)
    print(f"   ⚠️ No se pudo liberar el puerto {port}. Puede que falle al iniciar.\n")


def main():
    parser = argparse.ArgumentParser(description="Arrancar Script Editor en producción")
    parser.add_argument("--host", default="0.0.0.0", help="Host de escucha (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto (default: 8000)")
    parser.add_argument("--log-level", default="info", help="Nivel de log uvicorn (default: info)")
    args = parser.parse_args()

    if not PYTHON.exists():
        print(f"ERROR: No se encontró el intérprete Python en {PYTHON}")
        print("Crea el entorno virtual primero: python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt")
        sys.exit(1)

    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    # Advertir si APP_ENV no está en producción
    if env.get("APP_ENV", "development") != "production":
        print("ADVERTENCIA: APP_ENV no está configurado como 'production'.")
        print("  Configura APP_ENV=production en el archivo .env o como variable de entorno.")

    cmd = [
        str(PYTHON), "-m", "uvicorn", "backend.main:app",
        "--host", args.host,
        "--port", str(args.port),
        "--workers", "1",          # 1 worker: job_manager usa estado en memoria compartida por threads
        "--log-level", args.log_level,
        "--access-log",
    ]

    print(f"Iniciando Script Editor API en http://{args.host}:{args.port}")
    print(f"Health check: http://localhost:{args.port}/api/health")
    print("Detener con Ctrl+C\n")

    _free_port(args.port)

    try:
        # Redirigir stderr → stdout. Uvicorn loguea a stderr por defecto, y
        # PowerShell lo marca como "NativeCommandError" en rojo aunque no sea
        # un error real. Mezclándolo con stdout evitamos ese ruido cosmético.
        subprocess.run(cmd, cwd=str(ROOT), env=env, stderr=subprocess.STDOUT)
    except KeyboardInterrupt:
        # Ctrl+C es la forma normal de detener el servidor.
        # uvicorn ya recibió la señal y se apagó limpiamente.
        pass


if __name__ == "__main__":
    main()
