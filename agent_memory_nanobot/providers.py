"""Adapters for bridging nanobot providers to agent-memory protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider as NanobotProvider


class NanobotLLMAdapter:
    """Wrap a nanobot provider to satisfy agent-memory's LLM protocol."""

    def __init__(self, provider: "NanobotProvider", default_model: str):
        self._provider = provider
        self._default_model = default_model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        response = await self._provider.chat(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content or ""
