"""Token pricing for OpenAI-compatible models.

Prices are USD per one million tokens.  Local models are intentionally free
(``0.0``) and unknown models also return ``0.0`` so cost reporting never
crashes a run.
"""

from __future__ import annotations

# Prefix -> (input_usd_per_1m, output_usd_per_1m)
PRICING: dict[str, tuple[float, float]] = {
    # Local / self-hosted models.
    "qwen": (0.0, 0.0),
    "lmstudio": (0.0, 0.0),
    "lm-studio": (0.0, 0.0),
    "ollama": (0.0, 0.0),
    "llama": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "local": (0.0, 0.0),
    # OpenAI.
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Common OpenAI-compatible providers.
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "deepseek-chat": (0.27, 1.10),
}

# Longest prefix wins, so ``gpt-4o-mini`` is not priced as ``gpt-4o``.
_PRICING_BY_LENGTH = sorted(PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def _lookup(model: str) -> tuple[float, float] | None:
    normalized = (model or "").strip().lower()
    if not normalized:
        return None
    for prefix, prices in _PRICING_BY_LENGTH:
        if normalized == prefix or normalized.startswith(prefix):
            return prices
    return None


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate USD cost for a model call, returning ``0.0`` if unknown."""
    prices = _lookup(model)
    if prices is None:
        return 0.0

    input_price, output_price = prices
    prompt = max(0, int(prompt_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    return (prompt / 1_000_000) * input_price + (
        completion / 1_000_000
    ) * output_price
