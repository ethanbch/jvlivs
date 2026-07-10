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
async def test_router_validate(mock_session_none):
    config = make_config("ollama")
    router = BackendRouter(config)

    mock_client = AsyncMock()
    mock_client.validate = AsyncMock(return_value=False)
    router._clients["ollama"] = mock_client

    assert await router.validate() is False

    mock_client.validate = AsyncMock(return_value=True)
    assert await router.validate() is True


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


def test_router_clean_history(mock_session_none):
    config = make_config("ollama")
    router = BackendRouter(config)
    
    messages = [
        {"role": "system", "content": "Keep it short"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "<think>\nThinking process...\n</think>\nActual response content"}
    ]
    cleaned = router._clean_history(messages)
    assert cleaned[0] == messages[0]
    assert cleaned[1] == messages[1]
    assert cleaned[2] == {"role": "assistant", "content": "Actual response content"}


@pytest.mark.asyncio
async def test_router_stream_with_thinking_openai_think_override(mock_session_none):
    config = make_config("openai")
    router = BackendRouter(config)
    router.switch("openai")

    async def fake_stream(messages, **kwargs):
        assert "think_override" not in kwargs
        yield "openai response"

    mock_client = AsyncMock()
    del mock_client.stream_with_thinking
    mock_client.stream = fake_stream
    mock_client.last_usage = None
    router._clients["openai"] = mock_client

    chunks = []
    async for chunk_type, chunk in router.stream_with_thinking(
        [{"role": "user", "content": "test"}], think_override=True
    ):
        chunks.append((chunk_type, chunk))

    assert chunks == [("content", "openai response")]
