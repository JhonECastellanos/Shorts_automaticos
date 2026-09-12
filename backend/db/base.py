"""Motor de base de datos y fábrica de sesiones (SQLAlchemy 2.x).

init_engine() debe llamarse en el startup de la aplicación.
get_db() es un context manager que proporciona sesiones thread-safe.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal: sessionmaker | None = None


class Base(DeclarativeBase):
    pass


def init_engine(database_url: str) -> None:
    """Inicializa el motor SQLAlchemy.

    Llamar una sola vez durante el startup. Thread-safe: usa pool_pre_ping para
    detectar conexiones rotas y las reconecta automáticamente.
    """
    global _engine, _SessionLocal

    connect_args: dict = {}
    if database_url.startswith("sqlite"):
        # SQLite no soporta acceso concurrente por defecto; necesitamos esta opción
        connect_args = {"check_same_thread": False}
        pool_kwargs: dict = {}
    else:
        pool_kwargs = {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 1800,
        }

    _engine = create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args=connect_args,
        **pool_kwargs,
    )
    _SessionLocal = sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    logger.info("Motor de BD inicializado: %s", _mask_url(database_url))


def _mask_url(url: str) -> str:
    """Oculta passwords en URLs de conexión para logging seguro."""
    import re
    return re.sub(r"(?<=://)[^:@]+:[^@]+@", "***:***@", url)


def get_engine():
    if _engine is None:
        raise RuntimeError(
            "Base de datos no inicializada. Asegúrate de llamar init_engine() en el startup."
        )
    return _engine


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager para obtener una sesión de BD con commit/rollback automático."""
    if _SessionLocal is None:
        raise RuntimeError(
            "Base de datos no inicializada. Asegúrate de llamar init_engine() en el startup."
        )
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_tables() -> None:
    """Crea todas las tablas si no existen (útil en desarrollo y tests).

    En producción usar `alembic upgrade head` para controlar migraciones.
    """
    from . import models as _  # noqa: F401 — registra modelos con Base.metadata
    Base.metadata.create_all(bind=get_engine())
    logger.info("Tablas de BD verificadas/creadas")


def check_connection() -> bool:
    """Verifica que la BD responde. Usado por /api/health."""
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Fallo de conexión a BD: %s", exc)
        return False
