"""
02_transcribe.py — CLI wrapper para transcripción con faster-whisper.

Uso:
    python scripts/02_transcribe.py --project mi_podcast --episode ep42
    python scripts/02_transcribe.py --project mi_podcast --episode ep42 --lang en
"""

import sys
from pathlib import Path

import click
from rich.console import Console

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.core.transcriber import transcribe

console = Console()


@click.command()
@click.option("--project", required=True, help="Nombre del proyecto")
@click.option("--episode", required=True, help="Nombre del episodio")
@click.option("--lang", default=None, help="Código de idioma (ej: es, en). None = auto-detect")
@click.option("--force", is_flag=True, help="Re-transcribir aunque ya exista el .srt")
def main(project: str, episode: str, lang: str, force: bool):
    console.print(f"[bold cyan]Transcribiendo:[/bold cyan] {project}/{episode}")

    def on_progress(step, msg, pct):
        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
        console.print(f"  [dim]{msg}{pct_str}[/dim]")

    try:
        result = transcribe(
            project=project, episode=episode,
            lang=lang, force=force,
            root=ROOT, on_progress=on_progress,
        )
        console.print(f"[green]\u2713[/green] {result.get('segments_count', '?')} segmentos transcritos")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
