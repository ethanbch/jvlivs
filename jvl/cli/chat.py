from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

from jvl.core.config import load_config, load_user_config, resolve_active_config
from jvl.core.router import BackendRouter
from jvl.core.session import get_session_info, set_session_info
from jvl.db import get_db
from jvl.db.session import (
    add_message,
    create_chat_session,
    get_chat_session,
    get_session_messages,
    get_session_usage,
)
from jvl.cli.model import pick_backend_interactive_async, pick_provider_and_model_async
from jvl.utils.errors import BackendNotAvailable, ConfigError

console = Console()

DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system.md"
USER_SYSTEM_PROMPT_PATH = Path.home() / ".jvl" / "prompts" / "system.md"
HISTORY_PATH = Path.home() / ".jvl" / "chat_history"

HELP_TEXT = """[bold]Commandes disponibles :[/bold]

  [cyan]/help[/cyan]      — Affiche cette aide
  [cyan]/exit[/cyan]      — Quitte le chat
  [cyan]/clear[/cyan]     — Efface l'historique de la conversation
  [cyan]/model[/cyan]     — Affiche/change le modèle actif (ex: `/model` pour le picker, `/model openai gpt-4o` pour changer directement)
  [cyan]/context[/cyan]   — Affiche le system prompt actif
  [cyan]/usage[/cyan]     — Affiche les tokens consommés dans la session
  [cyan]/think[/cyan]     — Active ou configure le mode thinking (ex: `/think true`, `/think low`, `/think clear` pour réinitialiser).
                 Prend en charge le streaming en temps réel de la pensée (thinking trace) sous forme de Panel grisé.
  [cyan]/nothink[/cyan]   — Désactive le mode thinking (équivaut à `/think false`)
"""


def _load_system_prompt() -> str | None:
    """Charge le system prompt depuis la configuration utilisateur ou le package."""
    if USER_SYSTEM_PROMPT_PATH.exists():
        try:
            content = USER_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass

    if DEFAULT_SYSTEM_PROMPT_PATH.exists():
        try:
            content = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
            return content or None
        except Exception:
            pass

    return None


def chat(
    backend: str | None = typer.Option(
        None, "--backend", "-b", help="Backend à utiliser"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Modèle à utiliser"
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
    """Lance une conversation interactive multi-turn avec JVLIVS.

    Prend en charge l'affichage dynamique et le streaming en temps réel du mode
    thinking pour les modèles avec raisonnement (ex: DeepSeek-R1, Qwen3.5:4b).
    """
    asyncio.run(_chat_repl(backend, model, session_id, system, temperature, no_markdown))


async def _chat_repl(
    backend_override: str | None,
    model_override: str | None,
    session_id: str | None,
    system_override: str | None,
    temperature: float,
    no_markdown: bool,
) -> None:
    # ── Config & Router ──
    try:
        user_config = load_user_config()
        repo_config = load_config()
        router = BackendRouter(user_config, repo_config=repo_config)
    except ConfigError as e:
        console.print(f"[red]Config invalide : {e}[/red]")
        raise typer.Exit(1)

    if backend_override or model_override:
        router.switch(backend_override or router.active_backend, model=model_override)

    active_backend = router.active_backend
    if not isinstance(active_backend, str):
        active_backend = "ollama"

    active_model = router.active_model
    if not isinstance(active_model, str):
        active_model = None
    
    # Résolution du modèle actif
    try:
        _, resolved_model, _ = resolve_active_config(user_config, repo_config=repo_config)
        model_name = active_model or resolved_model
    except Exception:
        model_name = active_model or "?"

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

        # Charger la mémoire de départ
        from jvl.core.memory import load_memory, enrich_system_prompt, strip_memory_tags
        memory_content = load_memory()
        if memory_content:
            if "[Contenu de la mémoire tronqué" in memory_content:
                console.print("[yellow]Avertissement : Mémoire tronquée (fichiers trop longs)[/yellow]")
            console.print("[dim]Avec mémoire active[/dim]")

        if session_id:
            chat_session = get_chat_session(db, session_id)
            if not chat_session:
                console.print(f"[red]Session '{session_id}' introuvable.[/red]")
                raise typer.Exit(1)
            messages = get_session_messages(db, session_id)
            system_prompt = chat_session.system_prompt
            
            # Injection dynamique linéaire de la mémoire
            clean_base = strip_memory_tags(system_prompt)
            enriched = enrich_system_prompt(clean_base)
            
            system_msg_idx = -1
            for idx, msg in enumerate(messages):
                if msg["role"] == "system":
                    system_msg_idx = idx
                    break
            
            if enriched:
                if system_msg_idx >= 0:
                    messages[system_msg_idx]["content"] = enriched
                else:
                    messages.insert(0, {"role": "system", "content": enriched})
            else:
                if system_msg_idx >= 0:
                    if clean_base:
                        messages[system_msg_idx]["content"] = clean_base
                    else:
                        messages.pop(system_msg_idx)
            
            system_prompt = enriched or clean_base
            usage_data = get_session_usage(db, session_id)
            total_tokens_in = usage_data["total_in"]
            total_tokens_out = usage_data["total_out"]
            console.print(f"\n[dim]Session reprise : {session_id}[/dim]")
            console.print(f"[dim]{usage_data['message_count']} messages chargés.[/dim]")
        else:
            system_prompt = system_override or _load_system_prompt()
            clean_base = strip_memory_tags(system_prompt)
            enriched_system_prompt = enrich_system_prompt(clean_base)
            
            chat_session = create_chat_session(
                db,
                backend=active_backend,
                model=model_name,
                system_prompt=clean_base,  # Conserver le prompt de base "propre" dans l'en-tête de session
            )
            session_id = chat_session.id
            if enriched_system_prompt:
                messages.append({"role": "system", "content": enriched_system_prompt})
                # Mettre à jour l'enregistrement du message système initial dans la DB
                from jvl.db.models import ChatMessage
                system_msg = db.query(ChatMessage).filter_by(session_id=session_id, role="system").first()
                if system_msg:
                    system_msg.content = enriched_system_prompt
                    db.commit()
            system_prompt = enriched_system_prompt or clean_base

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
        session_think_override: bool | str | None = None

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
                    new_model = None
                    if args:
                        new_backend = args[0]
                        new_model = args[1] if len(args) > 1 else None
                    else:
                        result = await pick_provider_and_model_async(
                            current_provider=active_backend,
                            current_model=model_name,
                        )
                        if result:
                            new_backend, new_model = result

                    if new_backend:
                        new_cfg = getattr(repo_config.backends, new_backend, None)
                        if new_cfg is None:
                            # Vérifier dans la user config
                            ucfg = load_user_config()
                            if new_backend not in ucfg.providers and new_backend not in {"ollama", "openai", "anthropic", "azure", "gemini"}:
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
                        if new_model:
                            model_name = new_model
                        elif new_cfg:
                            model_name = getattr(new_cfg, "model", "?")
                        else:
                            model_name = "?"

                        router.switch(new_backend, model=model_name if model_name != "?" else None)
                        active_backend = new_backend
                        set_session_info(active_backend, model_name if model_name != "?" else None)
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

                elif cmd == "/think":
                    if args:
                        val = args[0].lower()
                        if val in ("clear", "reset", "default", "none"):
                            session_think_override = None
                            console.print("[green]Thinking mode réinitialisé aux paramètres par défaut de la configuration.[/green]")
                        elif val == "false":
                            session_think_override = False
                            console.print("[green]Thinking mode désactivé pour cette session.[/green]")
                        elif val == "true":
                            session_think_override = True
                            console.print("[green]Thinking mode activé (True) pour cette session.[/green]")
                        else:
                            session_think_override = val
                            console.print(f"[green]Thinking mode défini sur '{val}' pour cette session.[/green]")
                    else:
                        if session_think_override is None:
                            console.print("[dim]Aucun override de thinking mode actif pour cette session (utilise la config).[/dim]")
                        else:
                            console.print(f"[green]Override de thinking mode actif : {session_think_override}[/green]")

                elif cmd == "/nothink":
                    session_think_override = False
                    console.print("[green]Thinking mode désactivé pour cette session.[/green]")

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
            thinking_response = ""
            content_response = ""
            full_response = ""
            try:
                if not no_markdown:
                    import time
                    from rich.live import Live
                    from rich.console import Group
                    from rich.panel import Panel

                    def make_renderable():
                        parts = []
                        if thinking_response:
                            parts.append(Panel(Markdown(thinking_response.strip()), title="[dim]Réflexion[/dim]", border_style="dim"))
                        if content_response:
                            parts.append(Markdown(content_response))
                        return Group(*parts) if parts else ""

                    console.print()
                    last_update = 0.0
                    with Live(make_renderable(), console=console, refresh_per_second=8, vertical_overflow="visible") as live:
                        async for chunk_type, chunk in router.stream_with_thinking(
                            messages, temperature=temperature,
                            think_override=session_think_override,
                        ):
                            if chunk_type == "thinking":
                                thinking_response += chunk
                            else:
                                content_response += chunk
                            
                            now = time.monotonic()
                            if now - last_update > 0.08:
                                live.update(make_renderable())
                                last_update = now
                        live.update(make_renderable())
                else:
                    in_thinking = False
                    async for chunk_type, chunk in router.stream_with_thinking(
                        messages, temperature=temperature,
                        think_override=session_think_override,
                    ):
                        if chunk_type == "thinking":
                            if not in_thinking:
                                print("<think>\n", end="", flush=True)
                                in_thinking = True
                            thinking_response += chunk
                            print(chunk, end="", flush=True)
                        else:
                            if in_thinking:
                                print("\n</think>\n", end="", flush=True)
                                in_thinking = False
                            content_response += chunk
                            print(chunk, end="", flush=True)
                    if in_thinking:
                        print("\n</think>\n", end="", flush=True)
                    print()

                if thinking_response:
                    full_response = f"<think>\n{thinking_response.strip()}\n</think>\n{content_response}"
                else:
                    full_response = content_response

            except (BackendNotAvailable, Exception) as e:
                console.print(f"[red]Erreur : {e}[/red]")
                partial_prefix = ""
                if thinking_response:
                    partial_prefix = f"<think>\n{thinking_response.strip()}\n</think>\n"
                if content_response:
                    partial_prefix += content_response
                error_msg = f"{partial_prefix}\n[Erreur : {e}]".strip()
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
