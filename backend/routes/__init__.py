"""Helpers compartidos entre rutas del backend."""

import re

from fastapi import HTTPException

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def validate_path_params(project: str, episode: str) -> None:
    """Valida que project y episode sean seguros y no contengan path traversal.

    Lanza HTTPException 400 si algún valor es inválido.
    Debe llamarse al inicio de cualquier endpoint que reciba estos parámetros de ruta.
    """
    for field, value in (("project", project), ("episode", episode)):
        if ".." in value or "/" in value or "\\" in value:
            raise HTTPException(400, f"Nombre de {field} inválido: contiene caracteres no permitidos")
        if not _SAFE_NAME_RE.match(value):
            raise HTTPException(
                400,
                f"Nombre de {field} inválido: solo se permiten letras, "
                "números, guiones, puntos y guiones bajos",
            )
