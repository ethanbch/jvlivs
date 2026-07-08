from __future__ import annotations

import json
import os
from pathlib import Path

SESSION_DIR = Path.home() / ".jvl"


def _get_tty_id() -> str:
    for fd in (0, 1, 2):
        try:
            return str(os.ttyname(fd))
        except Exception:
            continue
    return "notty"


def session_file_for_tty() -> Path:
    tty_id = _get_tty_id().replace("/", "_")
    return SESSION_DIR / f"session_{tty_id}"


def get_session_info() -> dict | None:
    """Lit les informations de session (provider + model) pour le TTY courant.

    Rétrocompatibilité : si le fichier contient une chaîne brute (ancien
    format), elle est interprétée comme ``{"provider": <chaîne>, "model": None}``.

    Renvoie ``None`` si le fichier n'existe pas ou est vide.
    """
    f = session_file_for_tty()
    if not f.exists():
        return None
    raw = f.read_text().strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    # Rétrocompatibilité : ancien format texte brut
    return {"provider": raw, "model": None}


def set_session_info(provider: str, model: str | None = None) -> None:
    """Écrit les informations de session au format JSON pour le TTY courant."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    f = session_file_for_tty()
    data = {"provider": provider, "model": model}
    f.write_text(json.dumps(data))


def get_session_backend() -> str | None:
    """Lit le backend de session pour le TTY courant."""
    info = get_session_info()
    if info is None:
        return None
    return info.get("provider")
