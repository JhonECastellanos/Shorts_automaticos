"""
06_export.py — CLI wrapper para exportación de shorts.

Uso:
    python scripts/06_export.py --project mi_podcast --episode ep42
    python scripts/06_export.py --project mi_podcast --episode ep42 --index 1
"""

import sys
from pathlib import Path

import click
from rich.console import Console

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.core.exporter import export_shorts

console = Console()


@click.command()
@click.option("--project", required=True, help="Nombre del proyecto")
@click.option("--episode", required=True, help="Nombre del episodio")
@click.option("--index", default=None, type=int, help="Exportar solo el momento N (1-based)")
@click.option("--no-subtitles", is_flag=True, help="Exportar sin subtítulos")
@click.option("--force", is_flag=True, help="Re-exportar aunque exista")
def main(project: str, episode: str, index: int, no_subtitles: bool, force: bool):
    console.print(f"[bold cyan]Exportando shorts:[/bold cyan] {project}/{episode}")

    def on_progress(step, msg, pct):
        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
        console.print(f"  [dim]{msg}{pct_str}[/dim]")

    try:
        results = export_shorts(
            project=project, episode=episode,
            index=index, no_subtitles=no_subtitles,
            force=force, root=ROOT, on_progress=on_progress,
        )
        for r in results:
            status = "[dim]omitido[/dim]" if r.get("skipped") else "[green]✓[/green]"
            console.print(f"  {status} {r['file']} (score: {r.get('score', '?')})")
        console.print(f"[green]✓[/green] {len(results)} shorts procesados")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
