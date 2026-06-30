from __future__ import annotations

from typing import AsyncIterator

from google import genai

from jvl.backends.base import BaseClient


class GeminiClient(BaseClient):

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self._last_usage: dict | None = None

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        parts: list[str] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")

        return "\n\n".join(parts)

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        prompt = self._messages_to_prompt(messages)

        config = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=config,
        )

        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

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
            self.client.models.generate_content(
                model=self.model,
                contents="ping",
            )
            return True
        except Exception:
            return False

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage
