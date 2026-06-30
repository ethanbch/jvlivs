from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console

from jvl.core.config import load_config

console = Console()

ALLOWED_BACKENDS = {"ollama", "openai", "anthropic", "azure", "gemini"}
LOCAL_CONFIG_PATH = Path("config/config.local.yaml")


def default(
    backend: str | None = typer.Argument(
        None, help="Backend à définir comme défaut permanent"
    ),
    show: bool = typer.Option(
        False, "--show", "-s", help="Affiche le backend par défaut actuel"
    ),
) -> None:
    """Définit le backend LLM par défaut de façon permanente (écrit dans config.local.yaml)."""

    if show or backend is None:
        try:
            config = load_config()
            console.print(
                f"[bold]Backend par défaut :[/bold] [cyan]{config.default_backend}[/cyan]"
            )
        except Exception as e:
            console.print(f"[red]Erreur lecture config : {e}[/red]")
            raise typer.Exit(1)
        return

    if backend not in ALLOWED_BACKENDS:
        console.print(
            f"[red]❌ Backend '{backend}' inconnu. Valeurs possibles : {', '.join(sorted(ALLOWED_BACKENDS))}[/red]"
        )
        raise typer.Exit(1)

    # Lit le local.yaml existant ou démarre avec un dict vide
    if LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH) as f:
            local_data: dict = yaml.safe_load(f) or {}
    else:
        local_data = {}
        LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    old_default = local_data.get("default_backend", "?")
    local_data["default_backend"] = backend

    with open(LOCAL_CONFIG_PATH, "w") as f:
        yaml.dump(local_data, f, default_flow_style=False, allow_unicode=True)

    console.print(
        f"[green]✓ Backend par défaut mis à jour :[/green] "
        f"[dim]{old_default}[/dim] → [bold cyan]{backend}[/bold cyan]"
    )
    console.print(f"[dim]Écrit dans {LOCAL_CONFIG_PATH}[/dim]")
