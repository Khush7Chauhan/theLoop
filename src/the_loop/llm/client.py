"""The only module allowed to make model calls over the network.

Wraps the OpenAI-compatible SDK, adds bounded exponential backoff with jitter
for transient failures, and returns a small typed response so the agent does
not depend on SDK response objects.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_S = 0.5
DEFAULT_MAX_DELAY_S = 8.0


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call independent of the OpenAI SDK object shape."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    def to_assistant_message(self) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            normalized = []
            for call in self.tool_calls:
                if isinstance(call, dict):
                    function = call.get("function", {})
                    normalized.append(
                        {
                            "id": call.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": function.get("arguments", "{}"),
                            },
                        }
                    )
                else:
                    normalized.append(
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                    )
            message["tool_calls"] = normalized
        return message


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(error: Exception) -> bool:
    """True only for rate limits, 5xx responses, timeouts, and connections."""
    if isinstance(
        error,
        (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError),
    ):
        return True

    status = _status_code(error)
    if status is not None:
        return status == 429 or 500 <= status <= 599

    name = type(error).__name__.lower()
    return "timeout" in name or "connection" in name


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_tool_calls(raw_calls: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls or []):
        function = _value(raw, "function", {})
        name = _value(function, "name", "")
        arguments = _value(function, "arguments", "{}")
        call_id = _value(raw, "id", "") or f"call_{index}"
        calls.append(
            ToolCall(
                id=str(call_id),
                name=str(name or ""),
                arguments=str(arguments or "{}"),
            )
        )
    return calls


class LLMClient:
    """OpenAI-compatible chat client with explicit retry and token capture."""

    def __init__(
        self,
        config,
        *,
        client: Any | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleep_fn=None,
        jitter_fn=None,
        base_delay_s: float = DEFAULT_BASE_DELAY_S,
        max_delay_s: float = DEFAULT_MAX_DELAY_S,
    ):
        self.config = config
        self.model = config.model
        self.max_retries = max(0, int(max_retries))
        self.sleep_fn = sleep_fn or time.sleep
        self.jitter_fn = jitter_fn or random.random
        self.base_delay_s = max(0.0, float(base_delay_s))
        self.max_delay_s = max(self.base_delay_s, float(max_delay_s))
        self._client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            max_retries=0,
        )

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return delay + delay * self.jitter_fn()

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                }
                if tools is not None:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice

                response = self._client.chat.completions.create(**kwargs)
            except Exception as error:
                last_error = error
                if not is_retryable(error) or attempt >= self.max_retries:
                    raise
                self.sleep_fn(self._backoff_delay(attempt))
                continue

            latency_ms = max(0, int(round((time.monotonic() - started) * 1000)))
            choices = _value(response, "choices", []) or []
            if not choices:
                raise ValueError("LLM response contained no choices.")

            message = _value(choices[0], "message", {})
            usage = _value(response, "usage", None)

            return LLMResponse(
                content=_value(message, "content", None),
                tool_calls=_normalize_tool_calls(_value(message, "tool_calls", None)),
                prompt_tokens=_as_int(_value(usage, "prompt_tokens", 0)),
                completion_tokens=_as_int(_value(usage, "completion_tokens", 0)),
                latency_ms=latency_ms,
            )

        # Unreachable: the loop either returns or raises.
        raise RuntimeError("LLM call failed without an exception") from last_error
