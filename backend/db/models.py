"""Modelos ORM para la base de datos (SQLAlchemy 2.x declarative style)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Job(Base):
    """Registro de un job de procesamiento de pipeline."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(8), primary_key=True)
    project: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    episode: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    url: Mapped[Optional[str]] = mapped_column(Text)
    local_file: Mapped[Optional[str]] = mapped_column(Text)
    ai_model: Mapped[Optional[str]] = mapped_column(String(100))
    diarization_provider: Mapped[Optional[str]] = mapped_column(String(50))
    min_duration: Mapped[Optional[int]] = mapped_column(Integer)
    max_duration: Mapped[Optional[int]] = mapped_column(Integer)

    # JSON columns — compatible con PostgreSQL (JSONB bajo el capó) y SQLite
    steps: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    steps_completed: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    step_durations: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )
    current_step: Mapped[Optional[str]] = mapped_column(String(50))
    progress: Mapped[Optional[float]] = mapped_column(Float)
    error: Mapped[Optional[str]] = mapped_column(Text)
    elapsed_seconds: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    events: Mapped[list[JobEvent]] = relationship(
        "JobEvent",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobEvent.id",
        lazy="dynamic",
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id!r} project={self.project!r} status={self.status!r}>"


class JobEvent(Base):
    """Evento de progreso / entrada de log persistida por job."""

    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(8),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    percent: Mapped[Optional[float]] = mapped_column(Float)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    job: Mapped[Job] = relationship("Job", back_populates="events")

    def __repr__(self) -> str:
        return f"<JobEvent job_id={self.job_id!r} step={self.step!r}>"


class ShortSnapshot(Base):
    """Snapshot de un short antes de reprocesar, para aprendizaje y comparación."""

    __tablename__ = "short_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    episode: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    short_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text)
    start: Mapped[float] = mapped_column(Float, nullable=False)
    end: Mapped[float] = mapped_column(Float, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Float)
    speakers: Mapped[Optional[list]] = mapped_column(JSON)
    speaker_zones: Mapped[Optional[dict]] = mapped_column(JSON)
    camera_segments: Mapped[Optional[list]] = mapped_column(JSON)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)
    snapshot_reason: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def __repr__(self) -> str:
        return f"<ShortSnapshot project={self.project!r} ep={self.episode!r} #{self.short_index}>"
