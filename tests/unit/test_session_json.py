import json
from unittest.mock import patch

import pytest

from jvl.core.session import get_session_backend, get_session_info, set_session_info


# ── get_session_info ────────────────────────────────────────────────


def test_get_session_info_json(tmp_path):
    """Un fichier JSON valide est correctement parsé."""
    session_file = tmp_path / "session_notty"
    session_file.write_text(json.dumps({"provider": "openai", "model": "gpt-4o"}))
    with patch("jvl.core.session.session_file_for_tty", return_value=session_file):
        info = get_session_info()
        assert info == {"provider": "openai", "model": "gpt-4o"}


def test_get_session_info_plain_string_retrocompat(tmp_path):
    """Un fichier au format texte brut (ancien format) est converti."""
    session_file = tmp_path / "session_notty"
    session_file.write_text("gemini")
    with patch("jvl.core.session.session_file_for_tty", return_value=session_file):
        info = get_session_info()
        assert info == {"provider": "gemini", "model": None}


def test_get_session_info_missing_file(tmp_path):
    """Fichier absent → None."""
    with patch(
        "jvl.core.session.session_file_for_tty",
        return_value=tmp_path / "session_notty",
    ):
        assert get_session_info() is None


def test_get_session_info_empty_file(tmp_path):
    """Fichier vide → None."""
    session_file = tmp_path / "session_notty"
    session_file.write_text("   ")
    with patch("jvl.core.session.session_file_for_tty", return_value=session_file):
        assert get_session_info() is None


# ── set_session_info ────────────────────────────────────────────────


def test_set_session_info_writes_json(tmp_path):
    """set_session_info écrit du JSON valide."""
    with patch("jvl.core.session.SESSION_DIR", tmp_path):
        with patch(
            "jvl.core.session.session_file_for_tty",
            return_value=tmp_path / "session_notty",
        ):
            set_session_info("openai", "gpt-4o")
            raw = (tmp_path / "session_notty").read_text()
            data = json.loads(raw)
            assert data == {"provider": "openai", "model": "gpt-4o"}


def test_set_session_info_model_none(tmp_path):
    """set_session_info sans modèle écrit model=null."""
    with patch("jvl.core.session.SESSION_DIR", tmp_path):
        with patch(
            "jvl.core.session.session_file_for_tty",
            return_value=tmp_path / "session_notty",
        ):
            set_session_info("gemini")
            raw = (tmp_path / "session_notty").read_text()
            data = json.loads(raw)
            assert data == {"provider": "gemini", "model": None}


# ── Roundtrips ──────────────────────────────────────────────────────


def test_roundtrip_set_then_get_info(tmp_path):
    """set_session_info → get_session_info roundtrip."""
    with patch("jvl.core.session.SESSION_DIR", tmp_path):
        with patch(
            "jvl.core.session.session_file_for_tty",
            return_value=tmp_path / "session_notty",
        ):
            set_session_info("openai", "gpt-4o")
            info = get_session_info()
            assert info == {"provider": "openai", "model": "gpt-4o"}


def test_roundtrip_set_then_get_backend(tmp_path):
    """set_session_info → get_session_backend renvoie le provider."""
    with patch("jvl.core.session.SESSION_DIR", tmp_path):
        with patch(
            "jvl.core.session.session_file_for_tty",
            return_value=tmp_path / "session_notty",
        ):
            set_session_info("ollama", "llama3.2:3b")
            assert get_session_backend() == "ollama"


def test_model_name_with_colon_roundtrip(tmp_path):
    """Les noms de modèle contenant ':' (ex. llama3.2:3b) survivent au roundtrip."""
    with patch("jvl.core.session.SESSION_DIR", tmp_path):
        with patch(
            "jvl.core.session.session_file_for_tty",
            return_value=tmp_path / "session_notty",
        ):
            set_session_info("ollama", "llama3.2:3b")
            info = get_session_info()
            assert info == {"provider": "ollama", "model": "llama3.2:3b"}
