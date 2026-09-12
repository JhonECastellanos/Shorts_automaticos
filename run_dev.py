"""Script de arranque rápido: dev servers."""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")


def run():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    # Backend
    backend = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "backend.main:app", "--reload", "--port", "8000"],
        cwd=str(ROOT), env=env,
    )

    # Frontend
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(ROOT / "frontend"), shell=True,
    )

    try:
        backend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()


if __name__ == "__main__":
    run()
