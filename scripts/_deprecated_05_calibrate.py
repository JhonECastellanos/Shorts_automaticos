"""
05_calibrate.py — CLI wrapper para calibración de zonas de speaker.

Uso:
    python scripts/05_calibrate.py --project mi_podcast --episode ep42
"""

import sys
from pathlib import Path

import click
from rich.console import Console

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.core.calibrator import calibrate

console = Console()


@click.command()
@click.option("--project", required=True, help="Nombre del proyecto")
@click.option("--episode", required=True, help="Nombre del episodio")
@click.option("--frames-per-speaker", default=5, show_default=True, help="Frames a analizar por speaker")
@click.option("--force", is_flag=True, help="Re-calibrar aunque ya exista")
def main(project: str, episode: str, frames_per_speaker: int, force: bool):
    console.print(f"[bold cyan]Calibrando:[/bold cyan] {project}/{episode}")

    def on_progress(step, msg, pct):
        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
        console.print(f"  [dim]{msg}{pct_str}[/dim]")

    try:
        zones = calibrate(
            project=project, episode=episode,
            frames_per_speaker=frames_per_speaker,
            force=force, root=ROOT, on_progress=on_progress,
        )
        console.print(f"[green]✓[/green] {len(zones)} speakers calibrados")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
