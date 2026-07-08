"""Sous-commandes `jvl history` — consultation et recherche de l'historique des conversations."""
from __future__ import annotations

from datetime import datetime
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from sqlalchemy import select

from jvl.db import get_db
from jvl.db.models import ChatMessage, ChatSession

console = Console()


def history(
    session_id: str | None = typer.Argument(None, help="ID de la session à afficher"),
    limit: int = typer.Option(10, "--limit", "-l", help="Nombre maximum de sessions à lister"),
    backend: str | None = typer.Option(None, "--backend", "-b", help="Filtrer par backend"),
    search: str | None = typer.Option(None, "--search", "-q", help="Rechercher dans le contenu des messages"),
) -> None:
    """Consulter l'historique des sessions et des conversations."""
    db = get_db()
    with db:
        if session_id:
            # Afficher une session complète par ID
            session = db.execute(
                select(ChatSession).filter_by(id=session_id)
            ).scalar_one_or_none()

            if not session:
                console.print(f"[red]Session '{session_id}' introuvable.[/red]")
                raise typer.Exit(1)

            console.print()
            console.rule(f"[bold cyan]Détails de la Session : {session.id}[/bold cyan]")
            console.print(f"[bold]Backend  :[/bold] [cyan]{session.backend}[/cyan]")
            console.print(f"[bold]Modèle   :[/bold] [cyan]{session.model}[/cyan]")
            
            # Formater les dates locales
            start_str = session.started_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            end_str = session.ended_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"[bold]Débutée  :[/bold] [dim]{start_str}[/dim]")
            console.print(f"[bold]Terminée :[/bold] [dim]{end_str}[/dim]")
            console.print()

            for msg in session.messages:
                role_label = f"[bold cyan]👤 {msg.role.upper()}[/bold cyan]"
                if msg.role == "assistant":
                    role_label = f"[bold green]🤖 ASSISTANT[/bold green]"
                elif msg.role == "system":
                    role_label = f"[bold magenta]⚙️ SYSTEM[/bold magenta]"
                
                time_str = msg.timestamp.astimezone().strftime("%H:%M:%S")
                console.print(f"{role_label} [dim]({time_str})[/dim]")
                
                # Rendre le Markdown si c'est du contenu textuel complexe
                if msg.role == "system":
                    console.print(f"[dim]{msg.content}[/dim]")
                else:
                    console.print(Markdown(msg.content))
                console.print()
            
            console.rule()
        else:
            # Lister les sessions
            query = select(ChatSession)
            if backend:
                query = query.filter(ChatSession.backend == backend)
            if search:
                query = (
                    query.join(ChatSession.messages)
                    .filter(ChatMessage.content.like(f"%{search}%"))
                    .distinct()
                )

            # Trier par la session commencée le plus récemment
            query = query.order_by(ChatSession.started_at.desc()).limit(limit)
            sessions = db.execute(query).scalars().all()

            if not sessions:
                console.print("[yellow]Aucune session trouvée dans l'historique.[/yellow]")
                return

            table = Table(title=f"Historique des {len(sessions)} dernières sessions")
            table.add_column("Session ID", style="cyan")
            table.add_column("Backend / Modèle", style="bold")
            table.add_column("Début", style="dim")
            table.add_column("Fin", style="dim")
            table.add_column("Messages", justify="right")

            for s in sessions:
                backend_model = f"{s.backend} / {s.model}"
                start_str = s.started_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
                end_str = s.ended_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
                msg_count = str(len(s.messages))
                table.add_row(s.id, backend_model, start_str, end_str, msg_count)

            console.print(table)
            console.print("\n[dim]Pour voir les détails d'une conversation : jvl history <session_id>[/dim]")
