from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest

from jvl.backends.anthropic import AnthropicClient
from jvl.backends.azure import AzureOpenAIClient
from jvl.backends.gemini import GeminiClient
from jvl.backends.ollama import OllamaClient
from jvl.backends.openai import OpenAIClient

# ── OllamaClient ──────────────────────────────────────────────────────────────


class TestOllamaClient:

    @pytest.fixture
    def client(self):
        return OllamaClient(base_url="http://localhost:11434", model="phi3")

    @pytest.mark.asyncio
    async def test_validate_ok(self, client, respx_mock):
        respx_mock.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        assert await client.validate() is True

    @pytest.mark.asyncio
    async def test_validate_fail_connection_error(self, client, respx_mock):
        respx_mock.get("http://localhost:11434/api/tags").mock(
            side_effect=httpx.ConnectError("refused")
        )
        assert await client.validate() is False

    @pytest.mark.asyncio
    async def test_validate_fail_non_200(self, client, respx_mock):
        respx_mock.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(500)
        )
        assert await client.validate() is False

    @pytest.mark.asyncio
    async def test_stream_yields_content(self, client, respx_mock):
        lines = [
            json.dumps({"message": {"content": "Bonjour"}, "done": False}),
            json.dumps({"message": {"content": " monde"}, "done": False}),
            json.dumps({"done": True, "prompt_eval_count": 5, "eval_count": 10}),
        ]
        respx_mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="\n".join(lines))
        )
        chunks = []
        async for chunk in client.stream([{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        assert chunks == ["Bonjour", " monde"]

    @pytest.mark.asyncio
    async def test_stream_records_usage(self, client, respx_mock):
        lines = [
            json.dumps({"message": {"content": "ok"}, "done": False}),
            json.dumps({"done": True, "prompt_eval_count": 7, "eval_count": 42}),
        ]
        respx_mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="\n".join(lines))
        )
        async for _ in client.stream([{"role": "user", "content": "test"}]):
            pass

        assert client.last_usage == {"prompt_tokens": 7, "completion_tokens": 42}

    @pytest.mark.asyncio
    async def test_stream_with_max_tokens(self, client, respx_mock):
        lines = [
            json.dumps({"message": {"content": "ok"}, "done": False}),
            json.dumps({"done": True, "prompt_eval_count": 1, "eval_count": 1}),
        ]
        route = respx_mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="\n".join(lines))
        )
        async for _ in client.stream(
            [{"role": "user", "content": "test"}], max_tokens=50
        ):
            pass

        payload = json.loads(route.calls[0].request.content)
        assert payload["options"]["num_predict"] == 50

    @pytest.mark.asyncio
    async def test_complete_returns_full_string(self, client, respx_mock):
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": False}),
            json.dumps({"done": True, "prompt_eval_count": 2, "eval_count": 5}),
        ]
        respx_mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="\n".join(lines))
        )
        result = await client.complete([{"role": "user", "content": "test"}])
        assert result == "Hello world"


# ── OpenAIClient ──────────────────────────────────────────────────────────────


class TestOpenAIClient:

    @pytest.fixture
    def client(self):
        return OpenAIClient(api_key="sk-test", model="gpt-4o")

    @pytest.mark.asyncio
    async def test_validate_ok(self, client):
        with patch.object(client.client.models, "list", new_callable=AsyncMock):
            assert await client.validate() is True

    @pytest.mark.asyncio
    async def test_validate_fail(self, client):
        with patch.object(
            client.client.models,
            "list",
            new_callable=AsyncMock,
            side_effect=Exception("auth error"),
        ):
            assert await client.validate() is False

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, client):
        async def fake_stream():
            for text in ["Hello", " world"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                yield chunk

        mock_stream = MagicMock()
        mock_stream.__aiter__ = lambda self: fake_stream()

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_stream,
        ):
            chunks = []
            async for chunk in client.stream([{"role": "user", "content": "test"}]):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_none_delta(self, client):
        async def fake_stream():
            for text in [None, "word", None]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                yield chunk

        mock_stream = MagicMock()
        mock_stream.__aiter__ = lambda self: fake_stream()

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_stream,
        ):
            chunks = []
            async for chunk in client.stream([{"role": "user", "content": "test"}]):
                chunks.append(chunk)

        assert chunks == ["word"]


# ── AnthropicClient ───────────────────────────────────────────────────────────


class TestAnthropicClient:

    @pytest.fixture
    def client(self):
        return AnthropicClient(api_key="ant-test", model="claude-sonnet-4-5")

    @pytest.mark.asyncio
    async def test_validate_ok(self, client):
        with patch.object(client.client.models, "list", new_callable=AsyncMock):
            assert await client.validate() is True

    @pytest.mark.asyncio
    async def test_validate_fail(self, client):
        with patch.object(
            client.client.models,
            "list",
            new_callable=AsyncMock,
            side_effect=Exception("auth error"),
        ):
            assert await client.validate() is False

    @pytest.mark.asyncio
    async def test_stream_extracts_system_prompt(self, client):
        messages = [
            {"role": "system", "content": "Tu es un assistant."},
            {"role": "user", "content": "Bonjour"},
        ]

        async def fake_response():
            event = MagicMock()
            event.type = "content_block_delta"
            event.delta = MagicMock()
            event.delta.text = "Salut"
            yield event

        mock_response = MagicMock()
        mock_response.__aiter__ = lambda self: fake_response()

        with patch.object(
            client.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            chunks = []
            async for chunk in client.stream(messages):
                chunks.append(chunk)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "Tu es un assistant."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])
        assert chunks == ["Salut"]

    @pytest.mark.asyncio
    async def test_stream_skips_non_delta_events(self, client):
        async def fake_response():
            for event_type in ["message_start", "content_block_start", "message_stop"]:
                event = MagicMock()
                event.type = event_type
                yield event

        mock_response = MagicMock()
        mock_response.__aiter__ = lambda self: fake_response()

        with patch.object(
            client.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            chunks = []
            async for chunk in client.stream([{"role": "user", "content": "test"}]):
                chunks.append(chunk)

        assert chunks == []


# ── AzureOpenAIClient ─────────────────────────────────────────────────────────


class TestAzureOpenAIClient:

    @pytest.fixture
    def client(self):
        return AzureOpenAIClient(
            api_key="az-test",
            api_base="https://test.openai.azure.com",
            model="gpt-4o",
            api_version="2024-02-01",
        )

    @pytest.mark.asyncio
    async def test_validate_ok(self, client):
        with patch.object(client.client.models, "list", new_callable=AsyncMock):
            assert await client.validate() is True

    @pytest.mark.asyncio
    async def test_validate_fail(self, client):
        with patch.object(
            client.client.models,
            "list",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            assert await client.validate() is False

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, client):
        async def fake_stream():
            for text in ["Azure", " response"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                yield chunk

        mock_stream = MagicMock()
        mock_stream.__aiter__ = lambda self: fake_stream()

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_stream,
        ):
            chunks = []
            async for chunk in client.stream([{"role": "user", "content": "test"}]):
                chunks.append(chunk)

        assert chunks == ["Azure", " response"]


# ── GeminiClient ──────────────────────────────────────────────────────────────


class TestGeminiClient:

    @pytest.fixture
    def client(self):
        with patch("jvl.backends.gemini.genai.Client"):
            return GeminiClient(api_key="gem-test", model="gemini-2.5-flash")

    @pytest.mark.asyncio
    async def test_validate_ok(self, client):
        client.client.models.generate_content = MagicMock(return_value=MagicMock())
        assert await client.validate() is True

    @pytest.mark.asyncio
    async def test_validate_fail(self, client):
        client.client.models.generate_content = MagicMock(
            side_effect=Exception("auth error")
        )
        assert await client.validate() is False

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, client):
        chunks_mock = [
            MagicMock(text="Bonjour"),
            MagicMock(text=" Gemini"),
        ]
        client.client.models.generate_content_stream = MagicMock(
            return_value=iter(chunks_mock)
        )
        chunks = []
        async for chunk in client.stream([{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        assert chunks == ["Bonjour", " Gemini"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_text(self, client):
        chunks_mock = [
            MagicMock(text=""),
            MagicMock(text="ok"),
            MagicMock(text=None),
        ]
        # None n'a pas d'attribut text retournant None via getattr
        chunks_mock[2].text = None
        client.client.models.generate_content_stream = MagicMock(
            return_value=iter(chunks_mock)
        )
        chunks = []
        async for chunk in client.stream([{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        assert chunks == ["ok"]

    @pytest.mark.asyncio
    async def test_messages_to_prompt_format(self, client):
        messages = [
            {"role": "system", "content": "Tu es utile."},
            {"role": "user", "content": "Bonjour"},
        ]
        prompt = client._messages_to_prompt(messages)
        assert "SYSTEM:" in prompt
        assert "USER:" in prompt
        assert "Tu es utile." in prompt
        assert "Bonjour" in prompt
