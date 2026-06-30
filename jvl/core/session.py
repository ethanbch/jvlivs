from __future__ import annotations

import os
from pathlib import Path

SESSION_DIR = Path.home() / ".jvl"


def _get_tty_id() -> str:
    try:
        return str(os.ttyname(0))
    except Exception:
        return "notty"


def session_file_for_tty() -> Path:
    tty_id = _get_tty_id().replace("/", "_")
    return SESSION_DIR / f"session_{tty_id}"


def get_session_backend() -> str | None:
    """Lit le backend de session pour le TTY courant."""
    f = session_file_for_tty()
    if f.exists():
        return f.read_text().strip() or None
    return None
