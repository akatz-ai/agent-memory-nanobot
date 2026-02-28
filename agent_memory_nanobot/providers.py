"""Adapters for bridging nanobot providers to agent-memory protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider as NanobotProvider

# Model prefixes that need routing through litellm instead of Anthropic
_NON_ANTHROPIC_PREFIXES = ("openai/", "groq/", "azure/", "bedrock/", "vertex_ai/")


class NanobotLLMAdapter:
    """Wrap a nanobot provider to satisfy agent-memory's LLM protocol.

    When the main provider is AnthropicDirect but the requested model is
    a non-Anthropic model (e.g. openai/gpt-4o-mini), routes through litellm
    instead.
    """

    def __init__(self, provider: "NanobotProvider", default_model: str):
        self._provider = provider
        self._default_model = default_model

    def _needs_litellm_fallback(self, model: str) -> bool:
        """Check if this model should bypass the main provider."""
        try:
            from nanobot.providers.anthropic_direct_provider import AnthropicDirectProvider
            if not isinstance(self._provider, AnthropicDirectProvider):
                return False
        except ImportError:
            return False
        return any(model.lower().startswith(p) for p in _NON_ANTHROPIC_PREFIXES)

    async def _litellm_complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Direct litellm call for non-Anthropic models."""
        import litellm
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        resolved_model = model or self._default_model
        if self._needs_litellm_fallback(resolved_model):
            try:
                return await self._litellm_complete(
                    messages, resolved_model, temperature, max_tokens,
                )
            except Exception as e:
                logger.warning("litellm fallback failed for {}: {}", resolved_model, e)
                # Fall through to main provider as last resort

        response = await self._provider.chat(
            messages=messages,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.content or ""

        # Detect error responses that the provider wraps as content text.
        # Without this check, error messages like "Anthropic API error: 529 - [overloaded]"
        # get passed to JSON parsers and cause confusing parse failures.
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason == "error":
            logger.warning(
                "LLM adapter: provider returned error response (model={}): {}",
                resolved_model,
                content[:300],
            )
            return ""

        return content
