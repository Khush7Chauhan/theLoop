"""Composition root for one agent run.

The CLI parses flags, builds the run directory outside the agent's writable
policy area, wires the concrete LLM client / tool registry / event logger into
the decoupled agent, and always writes a summary.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from the_loop.agent import Agent
from the_loop.config import load_config
from the_loop.errors import ConfigError
from the_loop.llm.client import LLMClient
from the_loop.observability.events import EventLogger
from the_loop.observability.summary import RunSummary, write_summary
from the_loop.policy.approvals import HeadlessApprover, InteractiveApprover
from the_loop.tools.registry import ToolRegistry

_DEFAULT_LLM_CLIENT = LLMClient


def _new_llm_client(config):
    """Resolve the client class so tests can patch either import location."""
    import the_loop.llm.client as client_module

    if client_module.LLMClient is not _DEFAULT_LLM_CLIENT:
        return client_module.LLMClient(config)
    return LLMClient(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the tiny coding agent loop.")
    parser.add_argument("task", help="The task for the agent to perform.")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory the agent can inspect and edit (default: cwd).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Never prompt; resolve approvals from --policy-file and deny unresolved asks.",
    )
    parser.add_argument(
        "--policy-file",
        default=None,
        help="JSON file with headless allow/deny overrides.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum model turns (default: 8).",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Maximum tool calls across the run (default: 12).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional total token budget.",
    )
    parser.add_argument(
        "--max-wall-clock-s",
        type=float,
        default=None,
        help="Optional wall-clock budget in seconds.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature flags (normalized to a sorted set).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for trace.jsonl and summary.json "
            "(default: <workspace>/.loop/runs/<run_id>)."
        ),
    )
    return parser


def _run_directory(config) -> Path:
    if config.output_dir is not None:
        return config.output_dir
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    return config.workspace / ".loop" / "runs" / run_id


def _error_summary(config, trace_path: Path, started: float) -> RunSummary:
    return RunSummary(
        status="error",
        turns=0,
        tool_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        estimated_cost_usd=0.0,
        wall_clock_s=max(0.0, time.monotonic() - started),
        config_hash=config.config_hash(),
        trace_path=str(trace_path),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        config = load_config(argv)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 3

    run_dir = _run_directory(config)
    trace_path = run_dir / "trace.jsonl"
    summary_path = run_dir / "summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = EventLogger(trace_path)
    started = time.monotonic()
    summary: RunSummary | None = None

    try:
        approver = (
            HeadlessApprover(config.policy, workspace=config.workspace)
            if config.headless
            else InteractiveApprover()
        )
        tool_registry = ToolRegistry(
            workspace=config.workspace,
            approver=approver,
        )
        llm_client = _new_llm_client(config)
        agent = Agent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            event_logger=logger,
            config=config,
            trace_path=str(trace_path),
        )
        summary = agent.run(config.task)
    except Exception as error:  # noqa: BLE001 - always emit a valid summary
        try:
            logger.emit(
                "error",
                error_type=type(error).__name__,
                message=str(error),
            )
        except Exception:  # noqa: BLE001
            pass
        summary = _error_summary(config, trace_path, started)
    finally:
        if summary is None:
            summary = _error_summary(config, trace_path, started)
        try:
            write_summary(summary, summary_path)
        finally:
            logger.close()

    print(f"Workspace: {config.workspace}")
    print(f"Stopped because: {summary.status}")
    if config.headless:
        print(f"Summary: {summary_path}")

    if summary.status == "resolved":
        return 0
    if summary.status == "budget_exceeded":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
