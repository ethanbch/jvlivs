"""Sous-commandes `jvl config` — gestion des providers, modèles et clés API."""
from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from jvl.core.config import (
    ProviderConfig,
    UserConfig,
    load_config,
    load_user_config,
    save_user_config,
)
from jvl.core.router import PROVIDER_REGISTRY

console = Console()

config_app = typer.Typer(help="Gérer la configuration utilisateur (providers, modèles, clés).")
provider_app = typer.Typer(help="Gérer les providers LLM.")
model_app = typer.Typer(help="Gérer les modèles LLM.")
key_app = typer.Typer(help="Gérer les clés API.")

config_app.add_typer(provider_app, name="provider")
config_app.add_typer(model_app, name="model")
config_app.add_typer(key_app, name="key")


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context) -> None:
    """Gérer la configuration utilisateur (providers, modèles, clés)."""
    if ctx.invoked_subcommand is None:
        console.print("\n[bold cyan]🔧 Guide de Configuration JVLIVS[/bold cyan]")
        console.print("Gérez votre configuration utilisateur persistée dans `~/.jvl/config.yaml`.\n")
        
        console.print("[bold]1. Migration initiale depuis la configuration de dev[/bold]")
        console.print("  [cyan]jvl config migrate[/cyan]                  - Copie la config globale du dépôt vers votre espace utilisateur\n")
        
        console.print("[bold]2. Gestion des Providers[/bold]")
        console.print("  [cyan]jvl config provider list[/cyan]              - Liste les providers configurés et l'état actif")
        console.print("  [cyan]jvl config provider add <name>[/cyan]        - Ajoute/met à jour un provider (ollama, openai, etc.)")
        console.print("  [cyan]jvl config provider remove <name>[/cyan]     - Supprime un provider de la configuration\n")
        
        console.print("[bold]3. Choix des Modèles[/bold]")
        console.print("  [cyan]jvl config model show[/cyan]                 - Affiche le provider et modèle actif")
        console.print("  [cyan]jvl config model use <name> [model][/cyan]  - Change le modèle et provider par défaut")
        console.print("  [cyan]jvl config model set <name> <model>[/cyan]   - Définit le modèle par défaut pour un provider spécifique\n")
        
        console.print("[bold]4. Clés API[/bold]")
        console.print("  [cyan]jvl config key show[/cyan]                   - Liste les clés API configurées (masquées)")
        console.print("  [cyan]jvl config key set <name> <valeur>[/cyan]    - Configure une clé API en clair")
        console.print("  [cyan]jvl config key set <name> --env <VAR>[/cyan] - Associe une clé API à une variable d'environnement")
        console.print("  [cyan]jvl config key remove <name>[/cyan]          - Supprime la clé API d'un provider\n")
        
        console.print("[dim]Pour afficher l'aide détaillée d'une commande : jvl config <command> --help[/dim]\n")



# ── jvl config provider ──────────────────────────────────────────────────────


@provider_app.command("add")
def provider_add(
    name: str = typer.Argument(..., help="Nom du provider (ollama, openai, anthropic, azure, gemini)"),
    api_key: str | None = typer.Option(None, "--api-key", help="Clé API"),
    base_url: str | None = typer.Option(None, "--base-url", help="URL de base (Ollama)"),
    api_base: str | None = typer.Option(None, "--api-base", help="Azure endpoint"),
    api_version: str | None = typer.Option(None, "--api-version", help="Azure API version"),
    default_model: str | None = typer.Option(None, "--default-model", "-m", help="Modèle par défaut"),
) -> None:
    """Ajoute ou met à jour un provider dans la configuration utilisateur."""
    if name not in PROVIDER_REGISTRY:
        console.print(
            f"[red]Provider '{name}' inconnu. Providers disponibles : "
            f"{', '.join(sorted(PROVIDER_REGISTRY))}[/red]"
        )
        raise typer.Exit(1)

    user_config = load_user_config()
    existing = user_config.providers.get(name, ProviderConfig())

    # Merge : ne remplacer que les valeurs fournies
    updated = ProviderConfig(
        api_key=api_key or existing.api_key,
        base_url=base_url or existing.base_url,
        api_base=api_base or existing.api_base,
        api_version=api_version or existing.api_version,
        default_model=default_model or existing.default_model,
    )
    user_config.providers[name] = updated
    save_user_config(user_config)
    console.print(f"[green]Provider '{name}' configuré.[/green]")


@provider_app.command("list")
def provider_list() -> None:
    """Affiche tous les providers configurés."""
    user_config = load_user_config()

    table = Table(title="Providers configurés")
    table.add_column("Provider", style="bold")
    table.add_column("Default Model")
    table.add_column("API Key")
    table.add_column("Status")

    for name in sorted(PROVIDER_REGISTRY):
        pcfg = user_config.providers.get(name)
        if pcfg is None:
            model = "—"
            key_status = "—"
            status = "[dim]non configuré[/dim]"
        else:
            model = pcfg.default_model or "—"
            if pcfg.api_key:
                # Masquer la clé
                key_status = pcfg.api_key[:8] + "…" if len(pcfg.api_key) > 8 else "***"
            else:
                key_status = "—"
            status = "[green]✓[/green]"

        if name == user_config.active_provider:
            status += " [cyan][actif][/cyan]"

        table.add_row(name, model, key_status, status)

    console.print(table)


@provider_app.command("remove")
def provider_remove(
    name: str = typer.Argument(..., help="Nom du provider à supprimer"),
) -> None:
    """Supprime un provider de la configuration utilisateur."""
    user_config = load_user_config()
    if name not in user_config.providers:
        console.print(f"[red]Provider '{name}' non trouvé dans la configuration.[/red]")
        raise typer.Exit(1)

    del user_config.providers[name]
    if user_config.active_provider == name:
        user_config.active_provider = "ollama"
        console.print(f"[yellow]Provider actif réinitialisé à 'ollama'.[/yellow]")

    save_user_config(user_config)
    console.print(f"[green]Provider '{name}' supprimé.[/green]")


# ── jvl config model ──────────────────────────────────────────────────────────


@model_app.command("set")
def model_set(
    provider: str = typer.Argument(..., help="Nom du provider"),
    model: str = typer.Argument(..., help="Nom du modèle par défaut"),
) -> None:
    """Définit le modèle par défaut d'un provider."""
    user_config = load_user_config()
    pcfg = user_config.providers.get(provider, ProviderConfig())
    pcfg.default_model = model
    user_config.providers[provider] = pcfg
    save_user_config(user_config)
    console.print(
        f"[green]Modèle par défaut de '{provider}' défini sur[/green] "
        f"[bold cyan]{model}[/bold cyan]"
    )


@model_app.command("use")
def model_use(
    provider: str = typer.Argument(..., help="Nom du provider"),
    model: str | None = typer.Argument(None, help="Nom du modèle (optionnel)"),
) -> None:
    """Change le provider et modèle actifs de façon permanente."""
    if provider not in PROVIDER_REGISTRY:
        console.print(
            f"[red]Provider '{provider}' inconnu. Providers disponibles : "
            f"{', '.join(sorted(PROVIDER_REGISTRY))}[/red]"
        )
        raise typer.Exit(1)

    user_config = load_user_config()
    user_config.active_provider = provider
    user_config.active_model = model
    save_user_config(user_config)

    model_display = model or "(défaut du provider)"
    console.print(
        f"[green]Configuration active :[/green] "
        f"[bold cyan]{provider} / {model_display}[/bold cyan]"
    )


@model_app.command("show")
def model_show() -> None:
    """Affiche le provider et modèle actifs."""
    user_config = load_user_config()

    try:
        repo_config = load_config()
    except Exception:
        repo_config = None

    provider = user_config.active_provider
    model = user_config.active_model

    if not model:
        pcfg = user_config.providers.get(provider)
        if pcfg:
            model = pcfg.default_model

    if not model and repo_config:
        repo_backend = getattr(repo_config.backends, provider, None)
        if repo_backend:
            model = getattr(repo_backend, "model", None)

    console.print(f"[bold]Provider :[/bold] [cyan]{provider}[/cyan]")
    console.print(f"[bold]Model    :[/bold] [cyan]{model or '—'}[/cyan]")


# ── jvl config key ────────────────────────────────────────────────────────────


@key_app.command("set")
def key_set(
    provider: str = typer.Argument(..., help="Nom du provider"),
    value: str | None = typer.Argument(None, help="Valeur de la clé API"),
    env: str | None = typer.Option(None, "--env", help="Nom de la variable d'environnement"),
) -> None:
    """Définit la clé API d'un provider."""
    if env:
        api_key = f"${{{env}}}"
        console.print(f"[green]Clé API de '{provider}' liée à[/green] [cyan]${{{env}}}[/cyan]")
    elif value:
        api_key = value
        console.print(f"[green]Clé API de '{provider}' définie.[/green]")
    else:
        console.print("[red]Spécifiez une valeur ou --env NOM_VARIABLE.[/red]")
        raise typer.Exit(1)

    user_config = load_user_config()
    pcfg = user_config.providers.get(provider, ProviderConfig())
    pcfg.api_key = api_key
    user_config.providers[provider] = pcfg
    save_user_config(user_config)


@key_app.command("show")
def key_show() -> None:
    """Affiche les clés API configurées (masquées)."""
    user_config = load_user_config()
    table = Table(title="Clés API")
    table.add_column("Provider", style="bold")
    table.add_column("Clé API")

    for name in sorted(PROVIDER_REGISTRY):
        pcfg = user_config.providers.get(name)
        if pcfg and pcfg.api_key:
            if pcfg.api_key.startswith("${"):
                display = pcfg.api_key  # Afficher la ref env var
            else:
                display = pcfg.api_key[:8] + "…" if len(pcfg.api_key) > 8 else "***"
        else:
            display = "[dim]non configurée[/dim]"
        table.add_row(name, display)

    console.print(table)


@key_app.command("remove")
def key_remove(
    provider: str = typer.Argument(..., help="Nom du provider"),
) -> None:
    """Supprime la clé API d'un provider."""
    user_config = load_user_config()
    pcfg = user_config.providers.get(provider)
    if pcfg is None or pcfg.api_key is None:
        console.print(f"[red]Aucune clé API configurée pour '{provider}'.[/red]")
        raise typer.Exit(1)

    pcfg.api_key = None
    save_user_config(user_config)
    console.print(f"[green]Clé API de '{provider}' supprimée.[/green]")


# ── jvl config migrate ────────────────────────────────────────────────────────


@config_app.command("migrate")
def config_migrate() -> None:
    """Migre la configuration repo vers ~/.jvl/config.yaml."""
    try:
        repo_config = load_config()
    except Exception as e:
        console.print(f"[red]Impossible de lire la config repo : {e}[/red]")
        raise typer.Exit(1)

    user_config = load_user_config()
    migrated = 0

    for name in ["ollama", "openai", "anthropic", "azure", "gemini"]:
        repo_backend = getattr(repo_config.backends, name, None)
        if repo_backend is None:
            continue
        if name in user_config.providers:
            continue  # Ne pas écraser la config user existante

        user_config.providers[name] = ProviderConfig(
            api_key=getattr(repo_backend, "api_key", None),
            base_url=getattr(repo_backend, "base_url", None),
            api_base=getattr(repo_backend, "api_base", None),
            api_version=getattr(repo_backend, "api_version", None),
            default_model=getattr(repo_backend, "model", None),
        )
        migrated += 1
        console.print(f"  [green]✓[/green] {name} → {getattr(repo_backend, 'model', '?')}")

    user_config.active_provider = repo_config.default_backend
    save_user_config(user_config)

    if migrated:
        console.print(f"\n[green]{migrated} provider(s) migré(s) vers ~/.jvl/config.yaml[/green]")
    else:
        console.print("[dim]Aucun nouveau provider à migrer.[/dim]")
