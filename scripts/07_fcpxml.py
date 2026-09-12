"""
07_fcpxml.py — CLI wrapper para generación de FCPXML.

Uso:
    python scripts/07_fcpxml.py --project mi_podcast --episode ep42
"""

import sys
from pathlib import Path

import click
from rich.console import Console

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.core.fcpxml_gen import generate_fcpxml

console = Console()


@click.command()
@click.option("--project", required=True, help="Nombre del proyecto")
@click.option("--episode", required=True, help="Nombre del episodio")
@click.option("--fps", default=None, type=float, help="FPS del video fuente")
def main(project: str, episode: str, fps: float):
    console.print(f"[bold cyan]Generando FCPXML:[/bold cyan] {project}/{episode}")

    def on_progress(step, msg, pct):
        console.print(f"  [dim]{msg}[/dim]")

    try:
        out_path = generate_fcpxml(
            project=project, episode=episode,
            fps=fps, root=ROOT, on_progress=on_progress,
        )
        console.print(f"[green]✓[/green] {out_path}")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
