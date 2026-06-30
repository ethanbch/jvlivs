from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseClient(ABC):

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream les chunks de réponse un par un."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """Retourne la réponse complète en une fois."""
        ...

    @abstractmethod
    async def validate(self) -> bool:
        """Vérifie que le backend est joignable et configuré."""
        ...

    @property
    def last_usage(self) -> dict | None:
        """Usage tokens après le stream. None si non disponible."""
        return None
