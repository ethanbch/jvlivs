"""Gestion de la mémoire utilisateur structurée en Markdown."""
from __future__ import annotations

import json
from pathlib import Path
import typer

# Définition des chemins de base de la mémoire
MEMORY_DIR = Path.home() / ".jvl" / "memory"
STATE_FILE = Path.home() / ".jvl" / "state.json"

DEFAULT_GLOBAL_MD = """# Mémoire Globale JVLIVS

## Identité & Préférences
- Rôle : Développeur
- Langages principaux : Python

## Style de Code
- Préférer le typage statique (Type Hints)
- Rédiger des tests unitaires concrets avec pytest
"""


def initialize_memory_structure() -> None:
    """Initialise l'arborescence de la mémoire utilisateur si elle est absente."""
    # Créer les dossiers de base
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    (MEMORY_DIR / "projects").mkdir(parents=True, exist_ok=True)

    # Initialiser global.md s'il n'existe pas
    global_file = MEMORY_DIR / "global.md"
    if not global_file.exists():
        try:
            global_file.write_text(DEFAULT_GLOBAL_MD, encoding="utf-8")
        except Exception as e:
            # Ne pas faire planter l'application mais notifier de manière discrète
            pass


def get_active_project() -> str | None:
    """Lit le projet actif depuis le fichier d'état state.json."""
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data.get("active_project")
    except Exception:
        return None


def truncate_to_max_tokens(text: str, max_tokens: int = 2000) -> str:
    """Tronque la mémoire combinée pour respecter une limite de tokens (1 token ~ 4 caractères)."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline != -1:
        truncated = truncated[:last_newline]

    return (
        truncated
        + "\n\n... [Contenu de la mémoire tronqué pour respecter la limite de contexte]"
    )


def load_memory(project: str | None = None, max_tokens: int = 2000) -> str:
    """Charge et combine la mémoire globale, les partagés, et celle du projet actif."""
    initialize_memory_structure()

    # Si aucun projet n'est passé en paramètre, on tente de récupérer le projet actif global
    if project is None:
        project = get_active_project()

    content_parts = []

    # 1. Charger global.md (obligatoire si présent)
    global_path = MEMORY_DIR / "global.md"
    if global_path.exists():
        try:
            content_parts.append(f"### Mémoire Globale :\n{global_path.read_text(encoding='utf-8').strip()}")
        except Exception:
            pass

    # 2. Charger wedr-internal.md (optionnel, si présent)
    wedr_path = MEMORY_DIR / "wedr-internal.md"
    if wedr_path.exists():
        try:
            content_parts.append(f"### Directives Internes :\n{wedr_path.read_text(encoding='utf-8').strip()}")
        except Exception:
            pass

    # 3. Charger les fichiers spécifiques au projet
    if project:
        project_dir = MEMORY_DIR / "projects" / project
        
        # Charger memory.md du projet
        proj_mem_path = project_dir / "memory.md"
        if proj_mem_path.exists():
            try:
                content_parts.append(
                    f"### Contexte Projet '{project}' :\n{proj_mem_path.read_text(encoding='utf-8').strip()}"
                )
            except Exception:
                pass

        # Charger pending.md du projet
        proj_pending_path = project_dir / "pending.md"
        if proj_pending_path.exists():
            try:
                content_parts.append(
                    f"### Tâches en suspens Projet '{project}' :\n{proj_pending_path.read_text(encoding='utf-8').strip()}"
                )
            except Exception:
                pass

    if not content_parts:
        return ""

    combined_text = "\n\n".join(content_parts)
    return truncate_to_max_tokens(combined_text, max_tokens)
