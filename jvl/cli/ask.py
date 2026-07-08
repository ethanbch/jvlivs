from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown

from jvl.core.config import load_config, load_user_config, resolve_active_config
from jvl.core.router import BackendRouter
from jvl.utils.errors import BackendNotAvailable, ConfigError

console = Console()


def ask(
    prompt: str = typer.Argument(..., help="Le prompt à envoyer au modèle"),
    backend: str | None = typer.Option(
        None, "--backend", "-b", help="Backend à utiliser"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Modèle à utiliser"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Température du modèle"
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Nombre max de tokens"
    ),
    no_markdown: bool = typer.Option(
        False, "--no-markdown", help="Afficher en texte brut"
    ),
    system: str | None = typer.Option(
        None, "--system", "-s", help="System prompt custom"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Affiche le backend et modèle utilisés"
    ),
) -> None:
    """Envoie un prompt au modèle et affiche la réponse en streaming."""

    messages: list[dict] = []

    if system:
        messages.append({"role": "system", "content": system})

    stdin_content = ""
    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read().rstrip("\n")
        if not stdin_content.strip():
            console.print(
                "[red]Attention : l'entrée standard (stdin) est vide (le fichier est peut-être vide ou inexistant).[/red]"
            )

    if stdin_content.strip():
        composite_prompt = f"""{prompt}

Contexte fourni via stdin :
{stdin_content}
"""
        messages.append({"role": "user", "content": composite_prompt})
    else:
        messages.append({"role": "user", "content": prompt})

    asyncio.run(
        _stream_response(
            messages,
            backend,
            model,
            temperature,
            max_tokens,
            no_markdown,
            debug,
            stdin_content=stdin_content if stdin_content.strip() else None,
        )
    )


async def _stream_response(
    messages: list[dict],
    backend: str | None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    no_markdown: bool = False,
    debug: bool = False,
    stdin_content: str | None = None,
) -> None:
    try:
        user_config = load_user_config()
        repo_config = load_config()
        router = BackendRouter(user_config, repo_config=repo_config)
        
        if backend or model:
            router.switch(backend or router.active_backend, model=model)

        active_backend = router.active_backend
        
        # Résolution du modèle actif
        try:
            _, resolved_model, _ = resolve_active_config(user_config, repo_config=repo_config)
            active_model = router.active_model or resolved_model
        except Exception:
            active_model = router.active_model or "?"

        full_response = ""

        with console.status(
            f"[dim]Connexion à {active_backend} ({active_model})...[/dim]"
        ):
            available = await router.validate(active_backend)
            if not available:
                raise BackendNotAvailable(
                    f"Backend '{active_backend}' non joignable."
                )

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

        if debug:
            usage = router.last_usage

            console.print()
            console.rule("[dim]debug[/dim]")
            console.print(f"[dim]backend  :[/dim] [cyan]{active_backend}[/cyan]")
            console.print(f"[dim]model    :[/dim] [cyan]{active_model}[/cyan]")
            if usage:
                console.print(
                    f"[dim]tokens   :[/dim] [cyan]{usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out[/cyan]"
                )
            if stdin_content:
                console.print("[dim]stdin    :[/dim]")
                console.print(stdin_content, highlight=False, markup=False)

    except BackendNotAvailable as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ConfigError as e:
        console.print(f"[red]Config invalide : {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Erreur inattendue : {e}[/red]")
        raise typer.Exit(1)
