"""Append-only structured event traces (one JSON object per line)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SECRET_KEYS = {"api_key", "apikey", "authorization", "password", "secret"}


def utc_timestamp() -> str:
    """UTC ISO 8601 timestamp with a trailing ``Z`` (second precision)."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SECRET_KEYS or "api_key" in normalized:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class EventLogger:
    """Write JSONL events, flushing after every line so crashes lose nothing."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def emit(self, event_type: str, **fields: Any) -> dict:
        event = {
            "ts": utc_timestamp(),
            "type": event_type,
            **_redact(fields),
        }
        self._file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self._file.flush()
        return event

    # Convenience aliases for callers that prefer ``log``/``write``.
    log = emit
    write = emit

    def run_start(
        self,
        *,
        task: str,
        model: str,
        config_hash: str,
        headless: bool,
        max_turns: int,
        max_tool_calls: int,
        max_tokens: int | None,
        max_wall_clock_s: float | None,
    ) -> dict:
        return self.emit(
            "run_start",
            task=task,
            model=model,
            config_hash=config_hash,
            headless=headless,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            max_wall_clock_s=max_wall_clock_s,
        )

    def llm_turn(
        self,
        *,
        turn: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        tool_call_count: int,
    ) -> dict:
        return self.emit(
            "llm_turn",
            turn=turn,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            tool_call_count=tool_call_count,
        )

    def policy_decision(
        self,
        *,
        tool: str,
        decision: str,
        reason: str | None = None,
        rule_hint: str | None = None,
    ) -> dict:
        return self.emit(
            "policy_decision",
            tool=tool,
            decision=decision,
            reason=reason,
            rule_hint=rule_hint,
        )

    def tool_result(
        self,
        *,
        tool: str,
        ok: bool,
        error_type: str | None = None,
        retryable: bool = False,
        duration_ms: int = 0,
        exit_code: int | None = None,
    ) -> dict:
        return self.emit(
            "tool_result",
            tool=tool,
            ok=ok,
            error_type=error_type,
            retryable=retryable,
            duration_ms=duration_ms,
            exit_code=exit_code,
        )

    def run_end(
        self,
        *,
        status: str,
        turns: int,
        tool_calls: int,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float,
        wall_clock_s: float,
        config_hash: str,
    ) -> dict:
        return self.emit(
            "run_end",
            status=status,
            turns=turns,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
            wall_clock_s=wall_clock_s,
            config_hash=config_hash,
        )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
