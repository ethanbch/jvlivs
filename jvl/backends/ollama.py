from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from jvl.backends.base import BaseClient


class OllamaClient(BaseClient):

    def __init__(self, base_url: str, model: str, think: bool | str | None = None) -> None:
        self.base_url = base_url
        self.model = model
        self.think = think
        self._last_usage: dict | None = None

    def _build_payload(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        think_override: bool | str | None = None,
    ) -> dict:
        effective_think = think_override if think_override is not None else self.think
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if effective_think is not None:
            payload["think"] = effective_think
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        return payload

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        think_override: bool | str | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            messages, temperature=temperature, max_tokens=max_tokens, think_override=think_override
        )

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        self._last_usage = {
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                        }
                        return
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content

    async def stream_with_thinking(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        think_override: bool | str | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        payload = self._build_payload(
            messages, temperature=temperature, max_tokens=max_tokens, think_override=think_override
        )

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        self._last_usage = {
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                        }
                        return
                    
                    msg = data.get("message", {})
                    thinking = msg.get("thinking", "")
                    content = msg.get("content", "")
                    
                    if thinking:
                        yield "thinking", thinking
                    if content:
                        yield "content", content

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        result = ""
        async for chunk in self.stream(messages, temperature=temperature):
            result += chunk
        return result

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage
