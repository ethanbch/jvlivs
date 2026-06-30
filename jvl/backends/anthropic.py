from __future__ import annotations

from typing import AsyncIterator

from anthropic import AsyncAnthropic

from jvl.backends.base import BaseClient


class AnthropicClient(BaseClient):

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)
        self._last_usage: dict | None = None

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        system = None
        formatted_messages = []

        for message in messages:
            if message["role"] == "system":
                system = message["content"]
            else:
                formatted_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

        response = await self.client.messages.create(
            model=self.model,
            messages=formatted_messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens or 1024,
            stream=True,
        )

        async for event in response:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                yield event.delta.text

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        chunks = []
        async for chunk in self.stream(messages, temperature=temperature):
            chunks.append(chunk)
        return "".join(chunks)

    async def validate(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage
