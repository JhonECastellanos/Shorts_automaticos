"""Rutas de transcripciones."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from ..models import TranscriptResponse
from . import validate_path_params

router = APIRouter()

ROOT = Path(__file__).parent.parent.parent


@router.get("/{project}/{episode}", response_model=TranscriptResponse)
async def get_transcript_info(project: str, episode: str):
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    transcripts_dir = ep_dir / "transcripts"
    meta_path = transcripts_dir / "transcription_meta.json"

    if not meta_path.exists():
        return TranscriptResponse()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return TranscriptResponse(
        srt_file=meta.get("srt_file"),
        transcript_file=meta.get("transcript_file"),
        language=meta.get("language"),
        segments_count=meta.get("segments_count"),
        duration_seconds=meta.get("duration_seconds"),
    )


@router.get("/{project}/{episode}/srt")
async def get_srt(project: str, episode: str):
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    transcripts_dir = ep_dir / "transcripts"
    srt_files = list(transcripts_dir.glob("*.srt"))
    if not srt_files:
        return PlainTextResponse("", media_type="text/plain; charset=utf-8")
    content = srt_files[0].read_text(encoding="utf-8")
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


@router.get("/{project}/{episode}/text")
async def get_text(project: str, episode: str):
    validate_path_params(project, episode)
    ep_dir = ROOT / "projects" / project / episode
    transcripts_dir = ep_dir / "transcripts"
    txt_files = list(transcripts_dir.glob("*_transcript.txt"))
    if not txt_files:
        return PlainTextResponse("", media_type="text/plain; charset=utf-8")
    content = txt_files[0].read_text(encoding="utf-8")
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")
