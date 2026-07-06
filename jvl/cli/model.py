from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
import questionary
from questionary import Style

from jvl.core.config import load_config
from jvl.core.session import get_session_backend, session_file_for_tty
from jvl.utils.errors import ConfigError

console = Console()

ALLOWED_BACKENDS = {"ollama", "openai", "anthropic", "azure", "gemini"}
SESSION_DIR = Path.home() / ".jvl"
SESSION_FILE = SESSION_DIR / "session"


_style = Style([
    ("qmark",        "fg:#4f98a3 bold"),
    ("question",     "bold"),
    ("answer",       "fg:#4f98a3 bold"),
    ("pointer",      "fg:#4f98a3 bold"),
    ("highlighted",  "fg:#4f98a3 bold"),
    ("selected",     "fg:#4f98a3"),
    ("separator",    "fg:#5a5957"),
    ("instruction",  "fg:#5a5957"),
])


def pick_backend_interactive(current: str | None = None) -> str | None:
    try:
        config = load_config()
    except Exception as e:
        console.print(f"[red]Erreur de config: {e}[/red]")
        return None

    if current is None:
        current = get_session_backend() or config.default_backend

    choices = []
    for name in ["ollama", "openai", "anthropic", "azure", "gemini"]:
        backend_cfg = getattr(config.backends, name)
        if backend_cfg is None:
            continue
        model_name = getattr(backend_cfg, "model", "—")
        label = f"{name:<12} {model_name}"
        if name == current:
            label += "  [actif]"
        choices.append(questionary.Choice(title=label, value=name))

    if not choices:
        console.print("[red]Aucun backend configuré dans votre config.yaml[/red]")
        return None

    default_choice = None
    for choice in choices:
        if choice.value == current:
            default_choice = choice
            break
    if not default_choice:
        default_choice = choices[0]

    return questionary.select(
        "Choisir un backend",
        choices=choices,
        default=default_choice,
        style=_style,
        use_shortcuts=False,
    ).ask()


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
        session_file = session_file_for_tty()
        if session_file.exists():
            session_file.unlink()
            console.print(
                "[green]Session réinitialisée — retour au backend par défaut.[/green]"
            )
        else:
            console.print("[dim]Aucune session active à réinitialiser.[/dim]")
        return

    if show:
        current = get_session_backend()
        try:
            config = load_config()
            default_val = config.default_backend
        except Exception:
            default_val = "?"
        if current:
            console.print(f"[bold]Backend session :[/bold] [cyan]{current}[/cyan]")
        else:
            console.print(f"[bold]Backend session :[/bold] [dim]aucun override[/dim]")
        console.print(f"[bold]Backend par défaut :[/bold] [cyan]{default_val}[/cyan]")
        console.print(
            f"[bold]Backend actif :[/bold] [green]{current or default_val}[/green]"
        )
        return

    if backend is None:
        backend = pick_backend_interactive()
        if backend is None:
            return

    if backend not in ALLOWED_BACKENDS:
        console.print(
            f"[red]Backend '{backend}' inconnu. Valeurs possibles : {', '.join(sorted(ALLOWED_BACKENDS))}[/red]"
        )
        raise typer.Exit(1)

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    session_file = session_file_for_tty()
    session_file.write_text(backend)

    console.print(
        f"[green]Backend de session défini sur[/green] [bold cyan]{backend}[/bold cyan]"
    )
    console.print("[dim]Actif jusqu'à la fermeture de ce terminal.[/dim]")

