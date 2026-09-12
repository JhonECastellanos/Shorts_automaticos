"""
03_analyze.py — CLI wrapper para análisis de momentos con Gemini.

Uso:
    python scripts/03_analyze.py --project mi_podcast --episode ep42
    python scripts/03_analyze.py --project mi_podcast --episode ep42 --min-score 7
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.core.analyzer import analyze

console = Console()


@click.command()
@click.option("--project", required=True, help="Nombre del proyecto")
@click.option("--episode", required=True, help="Nombre del episodio")
@click.option("--min-score", default=None, type=float, help="Score mínimo")
@click.option("--force", is_flag=True, help="Re-analizar aunque ya exista moments.json")
def main(project: str, episode: str, min_score: float, force: bool):
    console.print(f"[bold cyan]Analizando:[/bold cyan] {project}/{episode}")

    def on_progress(step, msg, pct):
        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
        console.print(f"  [dim]{msg}{pct_str}[/dim]")

    try:
        moments = analyze(
            project=project, episode=episode,
            min_score=min_score, force=force,
            root=ROOT, on_progress=on_progress,
        )

        table = Table(title=f"Mejores momentos — {project}/{episode}", show_lines=True)
        table.add_column("#", style="bold", width=4)
        table.add_column("Start", width=8)
        table.add_column("End", width=8)
        table.add_column("Score", width=7)
        table.add_column("Topic", min_width=20)
        table.add_column("Hook", min_width=30)

        for n, m in enumerate(moments, 1):
            def fmt(s): return f"{int(s)//60:02d}:{int(s)%60:02d}"
            table.add_row(
                str(n), fmt(m["start"]), fmt(m["end"]),
                f"[bold green]{m['score']}[/bold green]",
                m.get("topic", ""), m.get("hook", "")
            )

        console.print(table)
        console.print(f"[green]✓[/green] {len(moments)} momentos detectados")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
