"""Decoupled plan -> act -> observe agent loop.

The agent deliberately knows nothing about Docker, subprocesses, filesystem
paths, policy rules, or approvals.  It talks only to the injected LLM client,
tool registry, event logger, and config object.
"""

from __future__ import annotations

import json
import time
from typing import Any

from the_loop.llm.pricing import estimate_cost_usd
from the_loop.observability.summary import RunSummary

SYSTEM_PROMPT = (
    "You are a tiny coding agent. Use tools when you need to inspect "
    "or change the workspace. Paths must stay inside the workspace. "
    "When you have enough information, answer clearly."
)


class Agent:
    """Run one task against injected collaborators."""

    def __init__(
        self,
        *,
        llm_client: Any,
        tool_registry: Any,
        event_logger: Any,
        config: Any,
        trace_path: str,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.event_logger = event_logger
        self.config = config
        self.trace_path = trace_path
        self.system_prompt = system_prompt
        self.messages: list[dict] = []

    def _wall_clock_exceeded(self, started: float) -> bool:
        limit = getattr(self.config, "max_wall_clock_s", None)
        return limit is not None and (time.monotonic() - started) >= limit

    def _config_hash(self) -> str:
        hasher = getattr(self.config, "config_hash", None)
        if callable(hasher):
            return str(hasher())
        return str(getattr(self.config, "config_hash", ""))

    def _emit(self, event_type: str, **fields: Any) -> None:
        emitter = getattr(self.event_logger, "emit", None)
        if callable(emitter):
            emitter(event_type, **fields)
            return
        method = getattr(self.event_logger, event_type, None)
        if callable(method):
            method(**fields)
            return
        logger = getattr(self.event_logger, "log", None)
        if callable(logger):
            logger(event_type, **fields)

    @staticmethod
    def _tool_call_id(tool_call: Any) -> str:
        if isinstance(tool_call, dict):
            return str(tool_call.get("id", ""))
        return str(getattr(tool_call, "id", ""))

    def run(self, task: str) -> RunSummary:
        started = time.monotonic()
        status = "error"
        state = {
            "turns": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        max_turns = int(getattr(self.config, "max_turns", 8))
        max_tool_calls = int(getattr(self.config, "max_tool_calls", 12))
        max_tokens = getattr(self.config, "max_tokens", None)

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        try:
            self._emit(
                "run_start",
                task=task,
                model=getattr(self.config, "model", ""),
                config_hash=self._config_hash(),
                headless=bool(getattr(self.config, "headless", False)),
                max_turns=max_turns,
                max_tool_calls=max_tool_calls,
                max_tokens=max_tokens,
                max_wall_clock_s=getattr(self.config, "max_wall_clock_s", None),
            )

            for turn in range(1, max_turns + 1):
                if self._wall_clock_exceeded(started):
                    status = "budget_exceeded"
                    break

                state["turns"] = turn
                response = self.llm_client.complete(
                    self.messages,
                    tools=self.tool_registry.tool_schemas(),
                )
                state["prompt_tokens"] += int(getattr(response, "prompt_tokens", 0) or 0)
                state["completion_tokens"] += int(
                    getattr(response, "completion_tokens", 0) or 0
                )

                tool_calls = list(getattr(response, "tool_calls", []) or [])
                self._emit(
                    "llm_turn",
                    turn=turn,
                    prompt_tokens=int(getattr(response, "prompt_tokens", 0) or 0),
                    completion_tokens=int(
                        getattr(response, "completion_tokens", 0) or 0
                    ),
                    latency_ms=int(getattr(response, "latency_ms", 0) or 0),
                    tool_call_count=len(tool_calls),
                )

                self.messages.append(response.to_assistant_message())

                total_tokens = state["prompt_tokens"] + state["completion_tokens"]
                if max_tokens is not None and total_tokens >= int(max_tokens):
                    status = "budget_exceeded"
                    break

                if self._wall_clock_exceeded(started):
                    status = "budget_exceeded"
                    break

                if not tool_calls:
                    content = getattr(response, "content", None)
                    status = (
                        "resolved"
                        if isinstance(content, str) and content.strip()
                        else "gave_up"
                    )
                    break

                budget_hit = False
                for tool_call in tool_calls:
                    if self._wall_clock_exceeded(started):
                        status = "budget_exceeded"
                        budget_hit = True
                        break

                    if state["tool_calls"] >= max_tool_calls:
                        status = "budget_exceeded"
                        budget_hit = True
                        break

                    outcome = self.tool_registry.execute(tool_call)
                    state["tool_calls"] += 1

                    self._emit(
                        "policy_decision",
                        tool=outcome.tool,
                        decision=outcome.decision,
                        reason=outcome.reason,
                        rule_hint=outcome.rule_hint,
                    )
                    self._emit(
                        "tool_result",
                        tool=outcome.tool,
                        ok=outcome.ok,
                        error_type=outcome.error_type,
                        retryable=outcome.retryable,
                        duration_ms=outcome.duration_ms,
                        exit_code=outcome.exit_code,
                    )

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": self._tool_call_id(tool_call),
                            "content": json.dumps(outcome.payload, default=str),
                        }
                    )

                if budget_hit:
                    break
            else:
                status = "budget_exceeded"

        except Exception as error:  # noqa: BLE001 - the run must still summarize
            status = "error"
            try:
                self._emit(
                    "error",
                    error_type=type(error).__name__,
                    message=str(error),
                )
            except Exception:  # noqa: BLE001 - never mask the original failure
                pass
        finally:
            wall_clock_s = max(0.0, time.monotonic() - started)
            estimated_cost = estimate_cost_usd(
                getattr(self.config, "model", ""),
                state["prompt_tokens"],
                state["completion_tokens"],
            )
            summary = RunSummary(
                status=status,
                turns=state["turns"],
                tool_calls=state["tool_calls"],
                prompt_tokens=state["prompt_tokens"],
                completion_tokens=state["completion_tokens"],
                estimated_cost_usd=estimated_cost,
                wall_clock_s=wall_clock_s,
                config_hash=self._config_hash(),
                trace_path=self.trace_path,
            )
            self._emit(
                "run_end",
                status=summary.status,
                turns=summary.turns,
                tool_calls=summary.tool_calls,
                prompt_tokens=summary.prompt_tokens,
                completion_tokens=summary.completion_tokens,
                estimated_cost_usd=summary.estimated_cost_usd,
                wall_clock_s=summary.wall_clock_s,
                config_hash=summary.config_hash,
            )

        return summary
