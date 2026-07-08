from __future__ import annotations

import os
from pathlib import Path

import httpx
import typer
from rich.console import Console
import questionary
from questionary import Style

from jvl.core.config import (
    UserConfig,
    load_config,
    load_user_config,
)
from jvl.core.router import PROVIDER_REGISTRY
from jvl.core.session import (
    get_session_backend,
    get_session_info,
    session_file_for_tty,
    set_session_info,
)
from jvl.utils.errors import ConfigError

console = Console()

ALLOWED_BACKENDS = set(PROVIDER_REGISTRY.keys())
SESSION_DIR = Path.home() / ".jvl"

# Modèles curated par provider (pour les providers cloud)
_CURATED_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o3", "o4-mini"],
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-35", "claude-opus-4"],
    "azure": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
}


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


def _list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Interroge l'API Ollama /api/tags pour lister les modèles installés."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _get_model_choices(provider: str, user_config: UserConfig) -> list[str]:
    """Retourne la liste des modèles disponibles pour un provider."""
    if provider == "ollama":
        pcfg = user_config.providers.get("ollama")
        base_url = pcfg.base_url if pcfg and pcfg.base_url else "http://localhost:11434"
        return _list_ollama_models(base_url)
    return _CURATED_MODELS.get(provider, [])


def _get_current_info(user_config: UserConfig) -> tuple[str, str | None]:
    """Récupère le provider/modèle actif (session > user config > repo)."""
    session_info = get_session_info()
    if session_info:
        return session_info.get("provider", "ollama"), session_info.get("model")

    try:
        repo_config = load_config()
        default_backend = repo_config.default_backend
        backend_cfg = getattr(repo_config.backends, default_backend, None)
        default_model = getattr(backend_cfg, "model", None) if backend_cfg else None
    except Exception:
        default_backend = user_config.active_provider
        default_model = user_config.active_model

    return default_backend, default_model


# ── Picker interactif (provider + modèle) ────────────────────────────────────


def pick_backend_interactive(current: str | None = None) -> str | None:
    """Picker interactif simple — retourne le nom du provider choisi.

    Conservé pour rétrocompatibilité avec `jvl default` et l'ancien `jvl model`.
    """
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


async def pick_backend_interactive_async(current: str | None = None) -> str | None:
    """Picker interactif async simple — retourne le nom du provider choisi.

    Conservé pour rétrocompatibilité avec le chat `/model` basique.
    """
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

    return await questionary.select(
        "Choisir un backend",
        choices=choices,
        default=default_choice,
        style=_style,
        use_shortcuts=False,
    ).ask_async()


def pick_provider_and_model() -> tuple[str, str | None] | None:
    """Picker interactif en 2 étapes : provider → modèle.

    Retourne ``(provider, model)`` ou ``None`` si annulé.
    Pour Ollama, interroge ``/api/tags`` pour lister les modèles installés.
    Pour les providers cloud, propose une liste curated + saisie libre.
    """
    user_config = load_user_config()
    current_provider, current_model = _get_current_info(user_config)

    # Étape 1 : choisir le provider
    provider_choices = []
    for name in sorted(PROVIDER_REGISTRY):
        pcfg = user_config.providers.get(name)
        default_model = pcfg.default_model if pcfg else None
        label = f"{name:<12}"
        if default_model:
            label += f" ({default_model})"
        if name == current_provider:
            label += "  [actif]"
        provider_choices.append(questionary.Choice(title=label, value=name))

    default_provider_choice = None
    for c in provider_choices:
        if c.value == current_provider:
            default_provider_choice = c
            break

    provider = questionary.select(
        "Choisir un provider",
        choices=provider_choices,
        default=default_provider_choice or provider_choices[0],
        style=_style,
    ).ask()

    if not provider:
        return None

    # Étape 2 : choisir le modèle
    model_list = _get_model_choices(provider, user_config)
    pcfg = user_config.providers.get(provider)
    provider_default = pcfg.default_model if pcfg else None

    if model_list:
        model_choices = []
        for m in model_list:
            label = m
            if m == current_model and provider == current_provider:
                label += "  [actif]"
            elif m == provider_default:
                label += "  [défaut]"
            model_choices.append(questionary.Choice(title=label, value=m))

        # Option de saisie libre
        model_choices.append(questionary.Choice(title="✏️  Saisir un autre modèle…", value="__custom__"))

        model = questionary.select(
            f"Choisir un modèle ({provider})",
            choices=model_choices,
            style=_style,
        ).ask()

        if model == "__custom__":
            model = questionary.text(
                "Nom du modèle",
                style=_style,
            ).ask()
    else:
        # Pas de liste disponible → saisie libre
        model = questionary.text(
            f"Nom du modèle ({provider})",
            default=provider_default or "",
            style=_style,
        ).ask()

    return (provider, model) if model else (provider, None)


async def pick_provider_and_model_async(
    current_provider: str | None = None,
    current_model: str | None = None,
) -> tuple[str, str | None] | None:
    """Version async du picker provider + modèle (pour le chat REPL).

    Retourne ``(provider, model)`` ou ``None`` si annulé.
    """
    user_config = load_user_config()
    if current_provider is None:
        current_provider, current_model = _get_current_info(user_config)

    # Étape 1 : choisir le provider
    provider_choices = []
    for name in sorted(PROVIDER_REGISTRY):
        pcfg = user_config.providers.get(name)
        default_model = pcfg.default_model if pcfg else None
        label = f"{name:<12}"
        if default_model:
            label += f" ({default_model})"
        if name == current_provider:
            label += "  [actif]"
        provider_choices.append(questionary.Choice(title=label, value=name))

    default_provider_choice = None
    for c in provider_choices:
        if c.value == current_provider:
            default_provider_choice = c
            break

    provider = await questionary.select(
        "Choisir un provider",
        choices=provider_choices,
        default=default_provider_choice or provider_choices[0],
        style=_style,
    ).ask_async()

    if not provider:
        return None

    # Étape 2 : choisir le modèle
    model_list = _get_model_choices(provider, user_config)
    pcfg = user_config.providers.get(provider)
    provider_default = pcfg.default_model if pcfg else None

    if model_list:
        model_choices = []
        for m in model_list:
            label = m
            if m == current_model and provider == current_provider:
                label += "  [actif]"
            elif m == provider_default:
                label += "  [défaut]"
            model_choices.append(questionary.Choice(title=label, value=m))

        model_choices.append(questionary.Choice(title="✏️  Saisir un autre modèle…", value="__custom__"))

        model = await questionary.select(
            f"Choisir un modèle ({provider})",
            choices=model_choices,
            style=_style,
        ).ask_async()

        if model == "__custom__":
            model = await questionary.text(
                "Nom du modèle",
                style=_style,
            ).ask_async()
    else:
        model = await questionary.text(
            f"Nom du modèle ({provider})",
            default=provider_default or "",
            style=_style,
        ).ask_async()

    return (provider, model) if model else (provider, None)


# ── Commande `jvl model` ─────────────────────────────────────────────────────


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
        session_info = get_session_info()
        user_config = load_user_config()
        try:
            repo_config = load_config()
            repo_default = repo_config.default_backend
        except Exception:
            repo_default = "?"
        
        default_val = user_config.active_provider
        
        if session_info:
            provider = session_info.get("provider", "?")
            model_name = session_info.get("model", "—")
            console.print(f"[bold]Backend session :[/bold] [cyan]{provider} / {model_name}[/cyan]")
        else:
            console.print(f"[bold]Backend session :[/bold] [dim]aucun override[/dim]")
        
        console.print(f"[bold]Backend par défaut (user) :[/bold] [cyan]{default_val}[/cyan]")
        console.print(f"[bold]Backend par défaut (repo) :[/bold] [dim]{repo_default}[/dim]")
        
        active_p = provider if session_info else default_val
        console.print(f"[bold]Backend actif :[/bold] [green]{active_p}[/green]")
        return

    if backend is None:
        # Picker interactif en 2 étapes
        result = pick_provider_and_model()
        if result is None:
            return
        provider, chosen_model = result
    else:
        if backend not in ALLOWED_BACKENDS:
            console.print(
                f"[red]Backend '{backend}' inconnu. Valeurs possibles : {', '.join(sorted(ALLOWED_BACKENDS))}[/red]"
            )
            raise typer.Exit(1)
        provider = backend
        chosen_model = None

    # Écrire la session en JSON
    set_session_info(provider, chosen_model)

    model_display = chosen_model or "(défaut)"
    console.print(
        f"[green]Backend de session défini sur[/green] "
        f"[bold cyan]{provider} / {model_display}[/bold cyan]"
    )
    console.print("[dim]Actif jusqu'à la fermeture de ce terminal.[/dim]")
