"""Script de arranque rápido: uvicorn para el backend."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main():
    env = {"PYTHONPATH": str(ROOT)}
    import os
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(
        [
            str(PYTHON), "-m", "uvicorn", "backend.main:app",
            "--reload",
            "--reload-dir", str(ROOT / "backend"),
            "--port", "8000",
        ],
        cwd=str(ROOT),
        env=env,
    )


if __name__ == "__main__":
    main()
