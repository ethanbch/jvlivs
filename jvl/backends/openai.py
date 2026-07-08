from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from jvl.backends.base import BaseClient


class OpenAIClient(BaseClient):

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        self._last_usage: dict | None = None

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                self._last_usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                }
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def validate(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage
