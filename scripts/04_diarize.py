"""
04_diarize.py — CLI wrapper para diarización de speakers.

Uso:
    python scripts/04_diarize.py --project mi_podcast --episode ep42
"""

import sys
from pathlib import Path

import click
from rich.console import Console

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.core.diarizer import diarize

console = Console()


@click.command()
@click.option("--project", required=True, help="Nombre del proyecto")
@click.option("--episode", required=True, help="Nombre del episodio")
@click.option("--force", is_flag=True, help="Re-diarizar aunque ya exista")
def main(project: str, episode: str, force: bool):
    console.print(f"[bold cyan]Diarizando:[/bold cyan] {project}/{episode}")

    def on_progress(step, msg, pct):
        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
        console.print(f"  [dim]{msg}{pct_str}[/dim]")

    try:
        result = diarize(
            project=project, episode=episode,
            force=force, root=ROOT, on_progress=on_progress,
        )
        segs = result.get("speaker_segments", result if isinstance(result, list) else [])
        console.print(f"[green]✓[/green] {len(segs)} segmentos de speaker procesados")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
