"""Configuración validada del backend con Pydantic Settings.

Los secretos se leen EXCLUSIVAMENTE de variables de entorno o del archivo .env.
El archivo config/settings.yaml solo contiene parámetros no sensibles.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, PrivateAttr, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Configuración global. Secretos solo desde entorno, nunca desde YAML."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Entorno ──────────────────────────────────────────────────────
    app_env: str = Field(default="development", description="development | production")

    # ── Secrets (solo variables de entorno) ─────────────────────────
    gemini_api_key_1: Optional[SecretStr] = None
    gemini_api_key_2: Optional[SecretStr] = None
    gemini_api_key_3: Optional[SecretStr] = None
    huggingface_token: Optional[SecretStr] = None

    # ── Base de datos ────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/script_editor",
        description=(
            "SQLAlchemy connection string. "
            "Ejemplos:\n"
            "  PostgreSQL: postgresql://user:pass@localhost:5432/script_editor\n"
            "  SQLite dev:  sqlite:///./script_editor.db"
        ),
    )

    # ── Seguridad / JWT ──────────────────────────────────────────────
    jwt_secret_key: SecretStr = Field(
        default="CHANGE_THIS_SECRET_IN_PRODUCTION_MIN_32_CHARS",
        description="Clave HMAC para firmar tokens JWT. Mínimo 32 chars aleatorios en prod.",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = Field(default=480, description="Duración del token JWT (minutos)")

    # ── CORS ─────────────────────────────────────────────────────────
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Lista de orígenes CORS separados por coma",
    )

    # ── Config YAML cargado en validación (sin secretos) ─────────────
    _yaml: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _load_yaml_config(self) -> "Settings":
        """Carga settings.yaml para secciones no sensibles."""
        yaml_path = ROOT / "config" / "settings.yaml"
        if yaml_path.exists():
            self._yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        return self

    # ── Accessors de secciones YAML ──────────────────────────────────

    def whisper_cfg(self) -> dict:
        return dict(self._yaml.get("whisper", {}))

    def analysis_cfg(self) -> dict:
        return dict(self._yaml.get("analysis", {}))

    def diarization_cfg(self) -> dict:
        """Sección diarization (pyannote es el único provider soportado en v3)."""
        cfg = dict(self._yaml.get("diarization", {}))
        cfg.pop("openroute_api_key", None)  # legacy cleanup
        return cfg

    def export_cfg(self) -> dict:
        return dict(self._yaml.get("export", {}))

    def crop_cfg(self) -> dict:
        return dict(self._yaml.get("crop", {}))

    def output_cfg(self) -> dict:
        return dict(self._yaml.get("output", {}))

    # ── Helpers ──────────────────────────────────────────────────────

    def get_gemini_api_keys(self) -> list[str]:
        """Devuelve las Gemini API keys configuradas, excluyendo placeholders."""
        placeholder = "your_gemini_api_key_here"
        keys: list[str] = []
        for attr in ("gemini_api_key_1", "gemini_api_key_2", "gemini_api_key_3"):
            val: Optional[SecretStr] = getattr(self, attr)
            if val:
                secret = val.get_secret_value()
                if secret and secret != placeholder:
                    keys.append(secret)
        return keys

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    def validate_for_startup(self) -> list[str]:
        """Valida la configuración al arrancar. Devuelve lista de advertencias."""
        warnings: list[str] = []
        placeholder_jwt = "CHANGE_THIS_SECRET_IN_PRODUCTION_MIN_32_CHARS"
        if self.is_production and self.jwt_secret_key.get_secret_value() == placeholder_jwt:
            warnings.append("JWT_SECRET_KEY usa el valor por defecto — cambiarlo antes de producción")
        if not self.get_gemini_api_keys():
            warnings.append(
                "Sin GEMINI_API_KEY_1 en entorno — el paso de análisis requiere "
                "al menos una API key de Gemini o un modelo custom configurado"
            )
        if "localhost" in self.database_url and self.is_production:
            warnings.append("DATABASE_URL apunta a localhost en entorno production")
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de configuración. Thread-safe gracias al lru_cache."""
    return Settings()
