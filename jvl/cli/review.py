"""Commande `jvl review` — revue de code automatique via fichiers ou stdin."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from jvl.core.config import load_config, load_user_config, resolve_active_config
from jvl.core.router import BackendRouter
from jvl.db import get_db
from jvl.db.session import add_message, create_chat_session
from jvl.utils.errors import BackendNotAvailable, ConfigError

console = Console()
DEFAULT_REVIEW_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "review.md"
USER_REVIEW_PROMPT_PATH = Path.home() / ".jvl" / "prompts" / "review.md"


def _load_review_prompt() -> str:
    """Charge le system prompt de revue depuis la configuration utilisateur ou le package."""
    if USER_REVIEW_PROMPT_PATH.exists():
        try:
            content = USER_REVIEW_PROMPT_PATH.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception as e:
            console.print(f"[yellow]Avertissement : Erreur de lecture de '{USER_REVIEW_PROMPT_PATH}' : {e}[/yellow]")

    if not DEFAULT_REVIEW_PROMPT_PATH.exists():
        console.print("[red]Erreur : Le fichier de prompt de revue par défaut est introuvable dans le package.[/red]")
        raise typer.Exit(1)
    
    try:
        content = DEFAULT_REVIEW_PROMPT_PATH.read_text(encoding="utf-8").strip()
        if not content:
            console.print("[red]Erreur : Le fichier de prompt de revue par défaut est vide.[/red]")
            raise typer.Exit(1)
        return content
    except Exception as e:
        console.print(f"[red]Erreur de lecture du prompt par défaut : {e}[/red]")
        raise typer.Exit(1)


def review(
    filepath: Path | None = typer.Argument(
        None, help="Chemin du fichier de code à reviewer"
    ),
    backend: str | None = typer.Option(
        None, "--backend", "-b", help="Backend à utiliser"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Modèle à utiliser"),
    temperature: float = typer.Option(
        0.2, "--temperature", "-t", help="Température (0.2 recommandé pour l'analyse)"
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Nombre max de tokens"
    ),
    no_markdown: bool = typer.Option(
        False, "--no-markdown", help="Afficher en texte brut"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Afficher les métadonnées de débogage"
    ),
) -> None:
    """Effectue une revue de code automatique sur un fichier ou du code fourni via stdin."""
    code_content = ""
    source_label = ""

    # 1. Lecture du code depuis le fichier
    if filepath is not None:
        if not filepath.exists():
            console.print(f"[red]Erreur : Le fichier '{filepath}' n'existe pas.[/red]")
            raise typer.Exit(1)
        if not filepath.is_file():
            console.print(
                f"[red]Erreur : '{filepath}' n'est pas un fichier valide.[/red]"
            )
            raise typer.Exit(1)
        try:
            code_content = filepath.read_text(encoding="utf-8")
            source_label = f"Fichier: {filepath.name}"
        except Exception as e:
            console.print(f"[red]Erreur de lecture du fichier : {e}[/red]")
            raise typer.Exit(1)

    # 2. Lecture du code depuis stdin
    else:
        if not sys.stdin.isatty():
            code_content = sys.stdin.read().rstrip("\n")
            source_label = "Entrée standard (stdin)"
            if not code_content.strip():
                console.print(
                    "[red]Erreur : L'entrée standard (stdin) est vide ou absente. "
                    "Vous devez spécifier un fichier à reviewer ou passer du code via un pipe (stdin).[/red]"
                )
                raise typer.Exit(1)
        else:
            console.print(
                "[red]Erreur : Vous devez spécifier un fichier à reviewer ou passer du code via un pipe (stdin).[/red]"
            )
            console.print("[dim]Exemples :[/dim]")
            console.print("  jvl review mon_fichier.py")
            console.print("  cat mon_fichier.py | jvl review")
            raise typer.Exit(1)

    # 3. Préparer les messages pour le modèle
    system_prompt = _load_review_prompt()
    user_prompt = f"Voici le code provenant de {source_label} à reviewer :\n\n```\n{code_content}\n```"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    asyncio.run(
        _run_review(
            messages,
            user_prompt,
            system_prompt,
            backend,
            model,
            temperature,
            max_tokens,
            no_markdown,
            debug,
        )
    )


async def _run_review(
    messages: list[dict],
    user_prompt: str,
    system_prompt: str,
    backend: str | None,
    model: str | None,
    temperature: float,
    max_tokens: int | None,
    no_markdown: bool,
    debug: bool,
) -> None:
    try:
        user_config = load_user_config()
        repo_config = load_config()
        router = BackendRouter(user_config, repo_config=repo_config)

        if backend or model:
            router.switch(backend or router.active_backend, model=model)

        active_backend = router.active_backend

        try:
            _, resolved_model, _ = resolve_active_config(
                user_config, repo_config=repo_config
            )
            active_model = router.active_model or resolved_model
        except Exception:
            active_model = router.active_model or "?"

        # Init session SQLite
        db = get_db()
        session_id = None
        with db:
            try:
                chat_session = create_chat_session(
                    db,
                    backend=active_backend,
                    model=active_model,
                    system_prompt=system_prompt,
                )
                session_id = chat_session.id
                add_message(db, session_id, "user", user_prompt)
            except Exception as e:
                # Ne pas bloquer la revue si la DB échoue
                if debug:
                    console.print(f"[yellow]Avertissement DB : {e}[/yellow]")

        full_response = ""
        with console.status(
            f"[dim]Analyse du code avec {active_backend} ({active_model})...[/dim]"
        ):
            available = await router.validate(active_backend)
            if not available:
                raise BackendNotAvailable(f"Backend '{active_backend}' non joignable.")

        async for chunk in router.stream(
            messages,
            backend=active_backend,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            full_response += chunk
            if no_markdown:
                print(chunk, end="", flush=True)

        if no_markdown:
            print()
        else:
            console.print(Markdown(full_response))

        # Enregistrer la réponse de l'assistant dans la DB
        if session_id:
            with db:
                try:
                    usage = router.last_usage
                    tokens_in = usage.get("prompt_tokens", 0) if usage else 0
                    tokens_out = usage.get("completion_tokens", 0) if usage else 0
                    add_message(
                        db,
                        session_id,
                        "assistant",
                        full_response,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
                except Exception as e:
                    if debug:
                        console.print(f"[yellow]Avertissement DB : {e}[/yellow]")

        if debug:
            console.print()
            console.rule("[dim]debug[/dim]")
            console.print(f"[dim]session_id :[/dim] [cyan]{session_id or '—'}[/cyan]")
            console.print(f"[dim]backend    :[/dim] [cyan]{active_backend}[/cyan]")
            console.print(f"[dim]model      :[/dim] [cyan]{active_model}[/cyan]")

    except BackendNotAvailable as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ConfigError as e:
        console.print(f"[red]Config invalide : {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Erreur inattendue : {e}[/red]")
        raise typer.Exit(1)
