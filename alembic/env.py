"""Alembic environment configuration.

Lee la DATABASE_URL desde Settings para que las migraciones funcionen
con el mismo valor que la aplicación sin duplicar configuración.
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Asegurar que el paquete backend sea importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Importar Base y los modelos para que Alembic los detecte
from backend.db.base import Base  # noqa: E402
import backend.db.models  # noqa: F401, E402 — registra los modelos con Base.metadata

# -- Leer settings de la app para obtener la DATABASE_URL --
from backend.config import get_settings  # noqa: E402

settings = get_settings()

# -- Configuración de Alembic --
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Inyectar la URL desde Settings (sobreescribe la de alembic.ini si existe)
config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    """Genera SQL sin conectar a la BD (útil para review de cambios)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Ejecuta las migraciones contra la BD conectada."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
