import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from jvl.db.models import Base


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def minimal_config_data() -> dict:
    return {
        "default_backend": "ollama",
        "backends": {
            "ollama": {"model": "phi3:3.8b", "base_url": "http://localhost:11434"},
        },
    }


@pytest.fixture
def full_config_data() -> dict:
    return {
        "default_backend": "ollama",
        "backends": {
            "ollama": {"model": "phi3:3.8b", "base_url": "http://localhost:11434"},
            "openai": {"model": "gpt-4o", "api_key": "sk-test"},
            "anthropic": {"model": "claude-sonnet-4-5", "api_key": "ant-test"},
            "azure": {
                "model": "gpt-4o",
                "api_key": "az-test",
                "api_base": "https://test.openai.azure.com",
                "api_version": "2024-02-01",
            },
            "gemini": {"model": "gemini-2.5-flash", "api_key": "gem-test"},
        },
    }


@pytest.fixture(autouse=True)
def mock_prompt_toolkit():
    """Sets up a dummy application session for prompt_toolkit to prevent NoConsoleScreenBufferError on Windows."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as inp:
        with create_app_session(input=inp, output=DummyOutput()):
            yield


@pytest.fixture(autouse=True)
def patch_global_memory_paths(tmp_path, monkeypatch):
    """Garantit que tous les tests utilisent un dossier de mémoire temporaire vide."""
    from jvl.core import memory
    tmp_mem_dir = tmp_path / "jvl" / "memory"
    tmp_state_file = tmp_path / "jvl" / "state.json"
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_mem_dir)
    monkeypatch.setattr(memory, "STATE_FILE", tmp_state_file)


@pytest.fixture(autouse=True)
def mock_load_memory_empty(request):
    """Mocke load_memory pour renvoyer une chaîne vide par défaut dans tous les tests sauf ceux de test_memory.py."""
    if "test_memory" in request.module.__name__:
        yield
    else:
        from unittest.mock import patch
        with patch("jvl.core.memory.load_memory", return_value=""):
            yield

