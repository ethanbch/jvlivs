from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

from jvl.core.config import load_config
from jvl.core.router import BackendRouter
from jvl.db import get_db
from jvl.db.session import (
    add_message,
    create_chat_session,
    get_chat_session,
    get_session_messages,
    get_session_usage,
)
from jvl.cli.model import pick_backend_interactive_async
from jvl.utils.errors import BackendNotAvailable, ConfigError

console = Console()

SYSTEM_PROMPT_PATH = Path("prompts/system.md")
HISTORY_PATH = Path.home() / ".jvl" / "chat_history"

HELP_TEXT = """[bold]Commandes disponibles :[/bold]

  [cyan]/help[/cyan]      — Affiche cette aide
  [cyan]/exit[/cyan]      — Quitte le chat
  [cyan]/clear[/cyan]     — Efface l'historique de la conversation
  [cyan]/model[/cyan]     — Affiche ou change le backend/modèle actif
  [cyan]/context[/cyan]   — Affiche le system prompt actif
  [cyan]/usage[/cyan]     — Affiche les tokens consommés dans la session
"""


def _load_system_prompt() -> str | None:
    """Charge le system prompt depuis prompts/system.md."""
    if SYSTEM_PROMPT_PATH.exists():
        content = SYSTEM_PROMPT_PATH.read_text().strip()
        return content or None
    return None


def chat(
    backend: str | None = typer.Option(
        None, "--backend", "-b", help="Backend à utiliser"
    ),
    session_id: str | None = typer.Option(
        None, "--session", help="Reprendre une session existante"
    ),
    system: str | None = typer.Option(
        None, "--system", "-s", help="System prompt custom"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Température du modèle"
    ),
    no_markdown: bool = typer.Option(
        False, "--no-markdown", help="Afficher en texte brut"
    ),
) -> None:
    """Lance une conversation interactive multi-turn avec JVLIVS."""
    asyncio.run(_chat_repl(backend, session_id, system, temperature, no_markdown))


async def _chat_repl(
    backend_override: str | None,
    session_id: str | None,
    system_override: str | None,
    temperature: float,
    no_markdown: bool,
) -> None:
    # ── Config & Router ──
    try:
        config = load_config()
        router = BackendRouter(config)
    except ConfigError as e:
        console.print(f"[red]Config invalide : {e}[/red]")
        raise typer.Exit(1)

    if backend_override:
        router.switch(backend_override)

    active_backend = router.active_backend
    backend_cfg = getattr(config.backends, active_backend, None)
    model_name = getattr(backend_cfg, "model", "?") if backend_cfg else "?"

    # ── Validate connectivity ──
    with console.status(f"[dim]Connexion à {active_backend}...[/dim]"):
        try:
            available = await router.validate()
            if not available:
                raise BackendNotAvailable(
                    f"Backend '{active_backend}' non joignable."
                )
        except BackendNotAvailable as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    # ── DB & Session ──
    with get_db() as db:
        messages: list[dict] = []
        total_tokens_in = 0
        total_tokens_out = 0
        system_prompt = system_override or _load_system_prompt()

        if session_id:
            chat_session = get_chat_session(db, session_id)
            if not chat_session:
                console.print(f"[red]Session '{session_id}' introuvable.[/red]")
                raise typer.Exit(1)
            messages = get_session_messages(db, session_id)
            system_prompt = chat_session.system_prompt
            usage_data = get_session_usage(db, session_id)
            total_tokens_in = usage_data["total_in"]
            total_tokens_out = usage_data["total_out"]
            console.print(f"\n[dim]Session reprise : {session_id}[/dim]")
            console.print(f"[dim]{usage_data['message_count']} messages chargés.[/dim]")
        else:
            chat_session = create_chat_session(
                db,
                backend=active_backend,
                model=model_name,
                system_prompt=system_prompt,
            )
            session_id = chat_session.id
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

        # ── Banner ──
        console.print()
        console.rule("[bold cyan]JVLIVS Chat[/bold cyan]")
        console.print(f"[dim]Session  : {session_id}[/dim]")
        console.print(f"[dim]Backend  : {active_backend} / {model_name}[/dim]")
        if system_prompt:
            preview = system_prompt[:60] + ("…" if len(system_prompt) > 60 else "")
            console.print(f"[dim]System   : {preview}[/dim]")
        console.print("[dim]Tapez /help pour les commandes, /exit pour quitter.[/dim]")
        console.print()

        # ── REPL ──
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        prompt_session = PromptSession(history=FileHistory(str(HISTORY_PATH)))

        while True:
            # ── Input ──
            try:
                prompt_text = HTML(
                    f'<style fg="#4f98a3">[{active_backend}/{model_name}]</style>'
                    f' <b>›</b> '
                )
                user_input = await prompt_session.prompt_async(prompt_text)
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Au revoir ![/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # ── Slash commands ──
            if user_input.startswith("/"):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()
                args = cmd_parts[1:]

                if cmd in ("/exit", "/quit", "/q"):
                    console.print("[dim]Au revoir ![/dim]")
                    break

                elif cmd == "/help":
                    console.print(HELP_TEXT)

                elif cmd == "/clear":
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    console.print("[green]Historique effacé.[/green]")

                elif cmd == "/model":
                    new_backend = None
                    if args:
                        new_backend = args[0]
                    else:
                        new_backend = await pick_backend_interactive_async(current=active_backend)

                    if new_backend:
                        new_cfg = getattr(config.backends, new_backend, None)
                        if new_cfg is None:
                            console.print(
                                f"[red]Backend '{new_backend}' non configuré.[/red]"
                            )
                            continue
                        with console.status(
                            f"[dim]Connexion à {new_backend}...[/dim]"
                        ):
                            try:
                                if not await router.validate(new_backend):
                                    console.print(
                                        f"[red]Backend '{new_backend}' non joignable.[/red]"
                                    )
                                    continue
                            except Exception:
                                console.print(
                                    f"[red]Backend '{new_backend}' non joignable.[/red]"
                                )
                                continue
                        router.switch(new_backend)
                        active_backend = new_backend
                        model_name = getattr(new_cfg, "model", "?")
                        console.print(
                            f"[green]Backend changé →[/green] "
                            f"[bold cyan]{active_backend} / {model_name}[/bold cyan]"
                        )

                elif cmd == "/context":
                    if system_prompt:
                        console.print("[bold]System prompt actif :[/bold]")
                        console.print(Markdown(system_prompt))
                    else:
                        console.print("[dim]Aucun system prompt défini.[/dim]")

                elif cmd == "/usage":
                    msg_count = len(
                        [m for m in messages if m["role"] != "system"]
                    )
                    console.print("[bold]Tokens session :[/bold]")
                    console.print(
                        f"  [dim]prompt     :[/dim] [cyan]{total_tokens_in}[/cyan]"
                    )
                    console.print(
                        f"  [dim]completion :[/dim] [cyan]{total_tokens_out}[/cyan]"
                    )
                    console.print(
                        f"  [dim]total      :[/dim] "
                        f"[cyan]{total_tokens_in + total_tokens_out}[/cyan]"
                    )
                    console.print(
                        f"  [dim]messages   :[/dim] [cyan]{msg_count}[/cyan]"
                    )

                else:
                    console.print(f"[red]Commande inconnue : {cmd}[/red]")
                    console.print(
                        "[dim]Tapez /help pour la liste des commandes.[/dim]"
                    )

                continue

            # ── User message ──
            messages.append({"role": "user", "content": user_input})
            user_msg = add_message(db, session_id, "user", user_input)

            # ── Stream response ──
            full_response = ""
            try:
                if not no_markdown:
                    console.print()

                async for chunk in router.stream(
                    messages, temperature=temperature,
                ):
                    full_response += chunk
                    if no_markdown:
                        print(chunk, end="", flush=True)

                if no_markdown:
                    print()
                else:
                    console.print(Markdown(full_response))

            except (BackendNotAvailable, Exception) as e:
                console.print(f"[red]Erreur : {e}[/red]")
                error_msg = f"[Erreur : {e}]"
                messages.append({"role": "assistant", "content": error_msg})
                add_message(db, session_id, "assistant", error_msg)
                continue

            # ── Track usage ──
            usage = router.last_usage
            tokens_in = usage.get("prompt_tokens", 0) if usage else 0
            tokens_out = usage.get("completion_tokens", 0) if usage else 0
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out

            # ── Save assistant response ──
            messages.append({"role": "assistant", "content": full_response})
            add_message(
                db, session_id, "assistant", full_response,
                tokens_in=tokens_in, tokens_out=tokens_out,
            )
            console.print()
