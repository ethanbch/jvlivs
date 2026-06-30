from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from jvl.core.config import load_config
from jvl.utils.errors import ConfigError

console = Console()

ALLOWED_BACKENDS = {"ollama", "openai", "anthropic", "azure", "gemini"}
SESSION_DIR = Path.home() / ".jvl"
SESSION_FILE = SESSION_DIR / "session"


def _get_tty_id() -> str:
    """Retourne un identifiant unique pour le TTY courant."""
    try:
        return str(os.ttyname(0))
    except Exception:
        return "notty"


def _session_file_for_tty() -> Path:
    tty_id = _get_tty_id().replace("/", "_")
    return SESSION_DIR / f"session_{tty_id}"


def get_session_backend() -> str | None:
    """Lit le backend de session pour le TTY courant. None si absent."""
    session_file = _session_file_for_tty()
    if session_file.exists():
        return session_file.read_text().strip() or None
    return None


def model(
    backend: str | None = typer.Argument(
        None,
        help="Backend à utiliser pour cette session (ollama, openai, anthropic, azure, gemini)",
    ),
    reset: bool = typer.Option(
        False, "--reset", "-r", help="Réinitialise le backend de session"
    ),
    show: bool = typer.Option(
        False, "--show", "-s", help="Affiche le backend actif pour cette session"
    ),
) -> None:
    """Définit le backend LLM pour la session courante (reset à la fermeture du terminal)."""

    if reset:
        session_file = _session_file_for_tty()
        if session_file.exists():
            session_file.unlink()
            console.print(
                "[green]✓ Session réinitialisée — retour au backend par défaut.[/green]"
            )
        else:
            console.print("[dim]Aucune session active à réinitialiser.[/dim]")
        return

    if show or backend is None:
        current = get_session_backend()
        try:
            config = load_config()
            default = config.default_backend
        except Exception:
            default = "?"
        if current:
            console.print(f"[bold]Backend session :[/bold] [cyan]{current}[/cyan]")
        else:
            console.print(f"[bold]Backend session :[/bold] [dim]aucun override[/dim]")
        console.print(f"[bold]Backend par défaut :[/bold] [cyan]{default}[/cyan]")
        console.print(
            f"[bold]Backend actif :[/bold] [green]{current or default}[/green]"
        )
        return

    if backend not in ALLOWED_BACKENDS:
        console.print(
            f"[red]❌ Backend '{backend}' inconnu. Valeurs possibles : {', '.join(sorted(ALLOWED_BACKENDS))}[/red]"
        )
        raise typer.Exit(1)

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    session_file = _session_file_for_tty()
    session_file.write_text(backend)

    console.print(
        f"[green]✓ Backend de session défini sur[/green] [bold cyan]{backend}[/bold cyan]"
    )
    console.print("[dim]Actif jusqu'à la fermeture de ce terminal.[/dim]")
