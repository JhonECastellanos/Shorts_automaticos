"""
pipeline.py — Orquestador CLI del pipeline completo.

Ejecuta los scripts en orden, con checkpoints y selección de pasos.

Uso:
    # Pipeline completo
    python scripts/pipeline.py run --project mi_podcast --episode ep42

    # Solo pasos específicos
    python scripts/pipeline.py run --project mi_podcast --episode ep42 --steps transcribe,analyze

    # Desde un paso en adelante
    python scripts/pipeline.py run --project mi_podcast --episode ep42 --from-step diarize

    # Ver estado de un episodio
    python scripts/pipeline.py status --project mi_podcast --episode ep42

    # Listar proyectos y episodios
    python scripts/pipeline.py list

Pasos disponibles: download, transcribe, diarize, face_positions, speaker_bind, analyze, edit, rank, export
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

console = Console()
SETTINGS_PATH = ROOT / "config" / "settings.yaml"

STEPS_ORDER = [
    "download", "transcribe", "diarize", "face_positions",
    "speaker_bind", "analyze", "edit", "rank", "export",
]

STEP_SCRIPTS = {
    "download":       "scripts/01_download.py",
    "transcribe":     "scripts/02_transcribe.py",
    "diarize":        "scripts/04_diarize.py",
    "face_positions": None,   # scripts/core/face_positions.py
    "speaker_bind":   None,   # scripts/core/speaker_bind.py
    "analyze":        "scripts/03_analyze.py",
    "edit":           None,   # scripts/core/editor.py
    "rank":           None,   # scripts/core/ranker.py
    "export":         "scripts/06_export.py",
}

STEP_OUTPUTS = {
    "download":       lambda ep: list((ep / "input").glob("*.mp4")) + list((ep / "input").glob("*.mkv")),
    "transcribe":     lambda ep: list((ep / "transcripts").glob("*.srt")),
    "diarize":        lambda ep: [ep / "diarization" / "speaker_segments.json"] if (ep / "diarization" / "speaker_segments.json").exists() else [],
    "face_positions": lambda ep: [ep / "faces" / "face_positions.json"] if (ep / "faces" / "face_positions.json").exists() else [],
    "speaker_bind":   lambda ep: [ep / "diarization" / "speaker_face_map.json"] if (ep / "diarization" / "speaker_face_map.json").exists() else [],
    "analyze":        lambda ep: [ep / "analysis" / "moments.json"] if (ep / "analysis" / "moments.json").exists() else [],
    "edit":           lambda ep: list((ep / "output" / "drafts").glob("short_*.mp4")),
    "rank":           lambda ep: [ep / "analysis" / "moments_ranked.json"] if (ep / "analysis" / "moments_ranked.json").exists() else [],
    "export":         lambda ep: list((ep / "output").glob("short_*/short_*.mp4")) + list((ep / "output").glob("*.fcpxml")),
}

STEP_DESCRIPTIONS = {
    "download":       "Descarga o registra el video fuente",
    "transcribe":     "Extrae audio + transcribe con Whisper (genera .srt y .txt)",
    "diarize":        "Diarización acústica: identifica cuándo habla cada SPEAKER_xx",
    "face_positions": "Detecta las N posiciones fijas de los participantes",
    "speaker_bind":   "Mapea SPEAKER_xx ↔ posición facial (calibración simple)",
    "analyze":        "ÚNICO paso LLM: elige los mejores momentos del transcript",
    "edit":           "Borradores low-res con cambios de cámara por voz activa",
    "rank":           "Ranking heurístico + guion DOCX estructurado",
    "export":         "Render final HD + distribución a OCAMO + docs + FCPXML",
}


def get_checkpoint_path(ep_dir: Path) -> Path:
    return ep_dir / ".pipeline_checkpoint.json"


def load_checkpoint(ep_dir: Path) -> dict:
    p = get_checkpoint_path(ep_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(ep_dir: Path, step: str, status: str) -> None:
    p = get_checkpoint_path(ep_dir)
    data = load_checkpoint(ep_dir)
    data[step] = {"status": status, "timestamp": datetime.utcnow().isoformat()}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_step(step: str, project: str, episode: str, extra_args: list[str]) -> bool:
    script = ROOT / STEP_SCRIPTS[step]
    cmd = [sys.executable, str(script), "--project", project, "--episode", episode] + extra_args
    console.print(f"\n[bold cyan]▶ {step.upper()}[/bold cyan] — {STEP_DESCRIPTIONS[step]}")
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd)
    return result.returncode == 0


def get_step_status(step: str, ep_dir: Path, checkpoint: dict) -> str:
    outputs = STEP_OUTPUTS[step](ep_dir)
    if outputs:
        return "[green]✓ completado[/green]"
    cp = checkpoint.get(step, {})
    if cp.get("status") == "error":
        return "[red]✗ error[/red]"
    return "[dim]pendiente[/dim]"


@click.group()
def cli():
    pass


@cli.command()
@click.option("--project", required=True)
@click.option("--episode", required=True)
@click.option("--steps", default="all", help="Pasos separados por coma. 'all' ejecuta todos.")
@click.option("--from-step", default=None, help="Ejecutar desde este paso en adelante.")
@click.option("--url", default=None, help="URL para el paso 'download'")
@click.option("--file", "local_file", default=None, help="Video local para el paso 'download'")
@click.option("--min-score", default=None, type=float, help="Score mínimo para el paso 'analyze'")
@click.option("--no-subtitles", is_flag=True, help="Exportar sin subtítulos en el paso 'export'")
@click.option("--fps", default=30, show_default=True, help="FPS para el paso 'fcpxml'")
@click.option("--force", is_flag=True, help="Forzar re-ejecución de todos los pasos")
def run(project, episode, steps, from_step, url, local_file, min_score, no_subtitles, fps, force):
    """Ejecuta el pipeline completo o pasos específicos."""
    ep_dir = ROOT / "projects" / project / episode
    checkpoint = load_checkpoint(ep_dir)

    # Resolver qué pasos ejecutar
    if steps == "all":
        steps_to_run = list(STEPS_ORDER)
    else:
        steps_to_run = [s.strip() for s in steps.split(",")]

    if from_step:
        if from_step not in STEPS_ORDER:
            console.print(f"[red]Paso desconocido: {from_step}. Válidos: {', '.join(STEPS_ORDER)}[/red]")
            raise SystemExit(1)
        idx = STEPS_ORDER.index(from_step)
        steps_to_run = [s for s in steps_to_run if STEPS_ORDER.index(s) >= idx]

    # Validar pasos
    invalid = [s for s in steps_to_run if s not in STEPS_ORDER]
    if invalid:
        console.print(f"[red]Pasos inválidos: {invalid}. Válidos: {', '.join(STEPS_ORDER)}[/red]")
        raise SystemExit(1)

    console.print(Panel(
        f"[bold]Proyecto:[/bold] {project}  [bold]Episodio:[/bold] {episode}\n"
        f"[bold]Pasos:[/bold] {' → '.join(steps_to_run)}",
        title="[bold cyan]Pipeline de Shorts[/bold cyan]",
        border_style="cyan",
    ))

    failed = []
    for step in steps_to_run:
        # Saltar si ya completado y no --force
        outputs = STEP_OUTPUTS[step](ep_dir)
        if outputs and not force and step != "download":
            console.print(f"[dim]⏭ {step}: ya completado, omitiendo (usa --force para repetir)[/dim]")
            continue

        # Construir args extra según el paso
        extra = []
        if step == "download":
            if url:
                extra += ["--url", url]
            elif local_file:
                extra += ["--file", local_file]
            else:
                console.print("[yellow]⚠ Paso 'download' omitido: sin --url ni --file.[/yellow]")
                continue
        elif step == "analyze" and min_score is not None:
            extra += ["--min-score", str(min_score)]
        elif step == "export" and no_subtitles:
            extra += ["--no-subtitles"]
        elif step == "fcpxml":
            extra += ["--fps", str(fps)]

        if force:
            extra += ["--force"]

        success = run_step(step, project, episode, extra)
        save_checkpoint(ep_dir, step, "done" if success else "error")

        if not success:
            failed.append(step)
            console.print(f"\n[red]✗ El paso '{step}' falló. Pipeline interrumpido.[/red]")
            console.print(f"[dim]Revisa el error y vuelve a correr con --from-step {step}[/dim]")
            break

    if not failed:
        console.print(Panel(
            f"[bold green]✓ Pipeline completado[/bold green]\n"
            f"Outputs en: projects/{project}/{episode}/output/",
            border_style="green",
        ))


@cli.command()
@click.option("--project", required=True)
@click.option("--episode", required=True)
def status(project, episode):
    """Muestra el estado actual de un episodio."""
    ep_dir = ROOT / "projects" / project / episode
    checkpoint = load_checkpoint(ep_dir)

    table = Table(title=f"Estado: {project}/{episode}", show_lines=False)
    table.add_column("Paso", style="bold", width=12)
    table.add_column("Descripción", min_width=35)
    table.add_column("Estado", min_width=20)
    table.add_column("Output(s)", min_width=25)

    for step in STEPS_ORDER:
        outputs = STEP_OUTPUTS[step](ep_dir)
        status_str = get_step_status(step, ep_dir, checkpoint)
        out_names = ", ".join(o.name for o in outputs[:2]) if outputs else "—"
        if len(outputs) > 2:
            out_names += f" (+{len(outputs)-2})"
        table.add_row(step, STEP_DESCRIPTIONS[step], status_str, out_names)

    console.print(table)


@cli.command(name="list")
def list_projects():
    """Lista todos los proyectos y episodios."""
    projects_dir = ROOT / "projects"
    if not projects_dir.exists() or not list(projects_dir.iterdir()):
        console.print("[dim]No hay proyectos todavía. Usa 'pipeline.py run' para empezar.[/dim]")
        return

    table = Table(title="Proyectos", show_lines=False)
    table.add_column("Proyecto", style="bold cyan")
    table.add_column("Episodio")
    table.add_column("Videos")
    table.add_column("Shorts")

    for proj in sorted(projects_dir.iterdir()):
        if not proj.is_dir():
            continue
        episodes = sorted(proj.iterdir())
        for ep in episodes:
            if not ep.is_dir():
                continue
            videos = len(list((ep / "input").glob("*.mp4"))) + len(list((ep / "input").glob("*.mkv"))) if (ep / "input").exists() else 0
            shorts = len(list((ep / "output").glob("short_*.mp4"))) if (ep / "output").exists() else 0
            table.add_row(proj.name, ep.name, str(videos), str(shorts))

    console.print(table)


if __name__ == "__main__":
    cli()
