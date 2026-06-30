from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from jvl.backends.base import BaseClient


class OllamaClient(BaseClient):

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._last_usage: dict | None = None

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

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
