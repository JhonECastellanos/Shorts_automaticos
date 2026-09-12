"""
01_download.py — CLI wrapper para descarga de videos.

Uso:
    python scripts/01_download.py --project mi_podcast --episode ep42 --url https://youtu.be/...
    python scripts/01_download.py --project mi_podcast --episode ep42 --file "C:/Videos/ep42.mp4"
    python scripts/01_download.py --project mi_podcast --episode ep42 --urls-file urls.txt
"""

import sys
from pathlib import Path

import click
from rich.console import Console

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.core.downloader import download

console = Console()


@click.command()
@click.option("--project", required=True, help="Nombre del proyecto (carpeta raíz)")
@click.option("--episode", required=True, help="Nombre del episodio")
@click.option("--url", default=None, help="URL de YouTube u otro sitio compatible con yt-dlp")
@click.option("--file", "local_file", default=None, help="Ruta a un video local")
@click.option("--urls-file", default=None, help="Archivo .txt con una URL por línea")
def main(project: str, episode: str, url: str, local_file: str, urls_file: str):
    if not any([url, local_file, urls_file]):
        console.print("[red]Error:[/red] Debes especificar --url, --file o --urls-file")
        raise SystemExit(1)

    console.print(f"[bold cyan]Proyecto:[/bold cyan] {project} / [bold cyan]Episodio:[/bold cyan] {episode}")

    def on_progress(step, msg, pct):
        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
        console.print(f"  [dim]{msg}{pct_str}[/dim]")

    # Resolver URLs
    urls = []
    if url:
        urls = [url]
    elif urls_file:
        txt = Path(urls_file)
        if not txt.exists():
            console.print(f"[red]No se encontró el archivo de URLs:[/red] {txt}")
            raise SystemExit(1)
        urls = [line.strip() for line in txt.read_text().splitlines() if line.strip()]

    try:
        if local_file:
            result = download(
                project=project, episode=episode,
                local_file=local_file,
                root=ROOT, on_progress=on_progress,
            )
            console.print(f"[green]✓[/green] {result.get('filename', 'OK')}")
        else:
            for u in urls:
                result = download(
                    project=project, episode=episode,
                    url=u,
                    root=ROOT, on_progress=on_progress,
                )
                console.print(f"[green]✓[/green] {result.get('filename', 'OK')}")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
