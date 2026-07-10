from pathlib import Path
from unittest.mock import patch

import pytest

from jvl.core.session import get_session_backend, session_file_for_tty


def test_session_file_no_tty():
    with patch("jvl.core.session.os.ttyname", side_effect=Exception("no tty"), create=True):
        f = session_file_for_tty()
        assert "notty" in f.name


def test_get_session_backend_absent(tmp_path):
    with patch("jvl.core.session.SESSION_DIR", tmp_path):
        with patch(
            "jvl.core.session.session_file_for_tty",
            return_value=tmp_path / "session_notty",
        ):
            result = get_session_backend()
            assert result is None


def test_get_session_backend_present(tmp_path):
    session_file = tmp_path / "session_notty"
    session_file.write_text("gemini")
    with patch("jvl.core.session.session_file_for_tty", return_value=session_file):
        result = get_session_backend()
        assert result == "gemini"


def test_get_session_backend_empty_file(tmp_path):
    session_file = tmp_path / "session_notty"
    session_file.write_text("   ")
    with patch("jvl.core.session.session_file_for_tty", return_value=session_file):
        result = get_session_backend()
        assert result is None
