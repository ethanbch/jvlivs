from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jvl.core.config import Config
from jvl.core.router import BackendRouter
from jvl.utils.errors import BackendNotAvailable, ConfigError


def make_config(default="ollama"):
    return Config(
        default_backend=default,
        backends={
            "ollama": {"model": "phi3", "base_url": "http://localhost:11434"},
            "openai": {"model": "gpt-4o", "api_key": "sk-test"},
        },
    )


@pytest.fixture
def mock_session_none():
    with patch("jvl.core.router.get_session_backend", return_value=None):
        yield


@pytest.fixture
def mock_session_openai():
    with patch("jvl.core.router.get_session_backend", return_value="openai"):
        yield


def test_router_uses_default_backend(mock_session_none):
    config = make_config("ollama")
    router = BackendRouter(config)
    assert router.active_backend == "ollama"


def test_router_uses_session_backend_over_default(mock_session_openai):
    config = make_config("ollama")
    router = BackendRouter(config)
    assert router.active_backend == "openai"


def test_router_switch(mock_session_none):
    config = make_config("ollama")
    router = BackendRouter(config)
    router.switch("openai")
    assert router.active_backend == "openai"


def test_router_build_client_unknown_backend(mock_session_none):
    config = make_config("ollama")
    router = BackendRouter(config)
    with pytest.raises(ConfigError, match="inconnu"):
        router._build_client("unknown_xyz")


def test_router_build_client_unconfigured_backend(mock_session_none):
    config = Config(
        default_backend="ollama",
        backends={"ollama": {"model": "phi3", "base_url": "http://localhost"}},
    )
    router = BackendRouter(config)
    with pytest.raises(ConfigError, match="non configuré"):
        router._build_client("openai")


@pytest.mark.asyncio
async def test_router_stream_raises_if_not_available(mock_session_none):
    config = make_config("ollama")
    router = BackendRouter(config)

    mock_client = AsyncMock()
    mock_client.validate = AsyncMock(return_value=False)
    router._clients["ollama"] = mock_client

    with pytest.raises(BackendNotAvailable):
        async for _ in router.stream([{"role": "user", "content": "test"}]):
            pass


@pytest.mark.asyncio
async def test_router_stream_yields_chunks(mock_session_none):
    config = make_config("ollama")
    router = BackendRouter(config)

    async def fake_stream(messages, **kwargs):
        yield "hello"
        yield " world"

    mock_client = AsyncMock()
    mock_client.validate = AsyncMock(return_value=True)
    mock_client.stream = fake_stream
    mock_client.last_usage = None
    router._clients["ollama"] = mock_client

    chunks = []
    async for chunk in router.stream([{"role": "user", "content": "test"}]):
        chunks.append(chunk)

    assert chunks == ["hello", " world"]
