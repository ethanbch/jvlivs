"""Gestion de la mémoire utilisateur structurée en Markdown."""
from __future__ import annotations

import json
import logging
from pathlib import Path

# Configurer le logger pour ce module
logger = logging.getLogger(__name__)

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
    try:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        (MEMORY_DIR / "projects").mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Impossible de créer l'arborescence de mémoire : %s", e)
        return

    # Initialiser global.md s'il n'existe pas
    global_file = MEMORY_DIR / "global.md"
    if not global_file.exists():
        try:
            global_file.write_text(DEFAULT_GLOBAL_MD, encoding="utf-8")
        except OSError as e:
            logger.warning("Impossible d'initialiser global.md : %s", e)


def get_active_project() -> str | None:
    """Lit le projet actif depuis le fichier d'état state.json."""
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data.get("active_project")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Impossible de lire le fichier d'état : %s", e)
        return None


def truncate_to_max_tokens(text: str, max_tokens: int = 2000) -> str:
    """Tronque la mémoire combinée pour respecter une limite de tokens (1 token ~ 4 caractères)."""
    # TODO: Utiliser un tokenizer précis (ex: tiktoken) si un modèle spécifique est configuré
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


def _safe_read_md_section(path: Path, title: str) -> str | None:
    """Lit de manière sécurisée une section de mémoire Markdown."""
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        return f"### {title} :\n{content}" if content else None
    except OSError as e:
        logger.warning("Impossible de lire la section de mémoire %s : %s", path, e)
        return None


def load_memory(project: str | None = None, max_tokens: int = 2000) -> str:
    """Charge et combine la mémoire globale, les partagés, et celle du projet actif."""
    initialize_memory_structure()

    # Si aucun projet n'est passé en paramètre, on tente de récupérer le projet actif global
    if project is None:
        project = get_active_project()

    content_parts = []

    # 1. Charger global.md (obligatoire si présent)
    global_part = _safe_read_md_section(MEMORY_DIR / "global.md", "Mémoire Globale")
    if global_part:
        content_parts.append(global_part)

    # 2. Charger wedr-internal.md (optionnel, si présent)
    wedr_part = _safe_read_md_section(MEMORY_DIR / "wedr-internal.md", "Directives Internes")
    if wedr_part:
        content_parts.append(wedr_part)

    # 3. Charger les fichiers spécifiques au projet
    if project:
        project_dir = MEMORY_DIR / "projects" / project
        
        # Charger memory.md du projet
        proj_mem_part = _safe_read_md_section(project_dir / "memory.md", f"Contexte Projet '{project}'")
        if proj_mem_part:
            content_parts.append(proj_mem_part)

        # Charger pending.md du projet
        proj_pending_part = _safe_read_md_section(project_dir / "pending.md", f"Tâches en suspens Projet '{project}'")
        if proj_pending_part:
            content_parts.append(proj_pending_part)

    if not content_parts:
        return ""

    combined_text = "\n\n".join(content_parts)
    return truncate_to_max_tokens(combined_text, max_tokens)

