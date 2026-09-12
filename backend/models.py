"""Modelos Pydantic para la API."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

# Patrón seguro para nombres de proyecto/episodio: alfanumérico, guiones, puntos y guiones bajos.
# Excluye .. y separadores de ruta para prevenir path traversal.
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _validate_safe_name(value: str, field_name: str) -> str:
    """Previene path traversal y caracteres peligrosos en nombres de proyecto/episodio."""
    if ".." in value:
        raise ValueError(f"'{field_name}' no puede contener '..'")
    if "/" in value or "\\" in value:
        raise ValueError(f"'{field_name}' no puede contener separadores de ruta")
    if not _SAFE_NAME_RE.match(value):
        raise ValueError(
            f"'{field_name}' solo puede contener letras, números, guiones, "
            "puntos y guiones bajos"
        )
    return value


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_EXPORT = "awaiting_export"


class PipelineStep(str, Enum):
    # Pipeline v3.5 — orden canónico
    DOWNLOAD = "download"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    FACE_POSITIONS = "face_positions"
    SPEAKER_BIND = "speaker_bind"
    ANALYZE = "analyze"
    EDIT = "edit"
    RANK = "rank"
    EXPORT = "export"
    # Aliases legacy (backward compat con jobs viejos en BD).
    FACE_TRACK = "face_track"
    ASD = "asd"
    CMIA_BIND = "cmia_bind"
    DETECT = "detect"
    GUION = "guion"
    VALIDATE = "validate"
    RANKING = "ranking"
    CALIBRATE = "calibrate"
    FCPXML = "fcpxml"
    DOC_EXPORT = "doc_export"


class JobCreate(BaseModel):
    project: str = Field(..., min_length=1, max_length=100)
    episode: str = Field(..., min_length=1, max_length=100)
    url: str | None = None
    local_file: str | None = None
    ai_model: str | None = None
    diarization_provider: str | None = Field(default=None, description="Provider de diarización (solo 'pyannote' en v3)")
    min_duration: int | None = Field(default=None, ge=10, le=600, description="Duración mínima de short en segundos")
    max_duration: int | None = Field(default=None, ge=10, le=3600, description="Duración máxima de short en segundos")
    steps: list[PipelineStep] = Field(default=[
        PipelineStep.DOWNLOAD,
        PipelineStep.TRANSCRIBE,
        PipelineStep.DIARIZE,
        PipelineStep.FACE_POSITIONS,
        PipelineStep.SPEAKER_BIND,
        PipelineStep.ANALYZE,
        PipelineStep.EDIT,
        PipelineStep.RANK,
        PipelineStep.EXPORT,
    ])

    @field_validator("project")
    @classmethod
    def validate_project(cls, v: str) -> str:
        return _validate_safe_name(v, "project")

    @field_validator("episode")
    @classmethod
    def validate_episode(cls, v: str) -> str:
        return _validate_safe_name(v, "episode")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        # Aceptar URLs http/https y rutas locales (para local_file se usa el otro campo)
        if v and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("La URL debe comenzar con http:// o https://")
        return v

    @field_validator("diarization_provider")
    @classmethod
    def validate_diarization_provider(cls, v: str | None) -> str | None:
        # v3: solo pyannote es soportado. Los providers legacy (text/openroute)
        # dependían de LLM o del guion — viola la regla "LLM solo en analyze".
        if v is not None and v != "pyannote":
            raise ValueError("diarization_provider solo acepta 'pyannote' en v3")
        return v

    @model_validator(mode="after")
    def validate_duration_order(self) -> "JobCreate":
        if self.min_duration and self.max_duration:
            if self.min_duration > self.max_duration:
                raise ValueError("min_duration no puede ser mayor que max_duration")
        return self


class ProgressEvent(BaseModel):
    job_id: str
    step: str
    message: str
    percent: float | None = None


class JobResponse(BaseModel):
    id: str
    project: str
    episode: str
    steps: list[str]
    status: JobStatus
    current_step: str | None = None
    progress: float | None = None
    steps_completed: list[str] = []
    error: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    step_durations: dict[str, float] = {}


class EpisodePipelineState(BaseModel):
    job_id: str | None = None
    project: str
    episode: str
    steps: list[str] = []
    status: str = "ready"
    current_step: str | None = None
    progress: float | None = None
    steps_completed: list[str] = []
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    step_durations: dict[str, float] = {}
    can_pause: bool = False
    can_resume: bool = False
    can_stop: bool = False
    can_cancel: bool = False


class ShortResponse(BaseModel):
    index: int
    start: float
    end: float
    duration: float
    score: float | None = None
    trend_score: float | None = None
    keyword_score: float | None = None
    hook_score: float | None = None
    duration_score: float | None = None
    base_score: float | None = None
    topic: str | None = None
    hook: str | None = None
    reason: str | None = None
    dominant_speaker: str | None = None
    rank: int | None = None
    file: str | None = None
    has_video: bool = False
    is_draft: bool = False  # true = borrador low-res (del paso 'edit'); false = final HD
    categories: list[str] = []


class VideoInfo(BaseModel):
    filename: str
    source: str | None = None
    title: str | None = None
    duration: float | None = None
    url: str | None = None


class TranscriptResponse(BaseModel):
    srt_file: str | None = None
    transcript_file: str | None = None
    language: str | None = None
    segments_count: int | None = None
    duration_seconds: float | None = None


class EpisodeItem(BaseModel):
    name: str
    shorts_count: int = 0
    has_transcript: bool = False
    has_public_video: bool = False
    status: str = "ready"
    steps: list[str] = []
    steps_completed: list[str] = []
    last_elapsed_seconds: float | None = None
    current_step: str | None = None
    progress: float | None = None
    job_id: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    step_durations: dict[str, float] = {}
    can_pause: bool = False
    can_resume: bool = False
    can_stop: bool = False
    can_cancel: bool = False


class ProjectListItem(BaseModel):
    project: str
    episodes: list[EpisodeItem]


class JobUpdateConfig(BaseModel):
    ai_model: str | None = None
    diarization_provider: str | None = None


class ShortUpdateTime(BaseModel):
    start: float
    end: float
