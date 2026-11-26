"""Command-line interface for the agent loop.

This module only parses flags, builds the runtime configuration, chooses an
approval strategy, and wires those into the runner.  The agent loop itself
lives in ``loop.py`` and is imported lazily to keep the entrypoint thin.
"""

from __future__ import annotations

import argparse
import sys

from the_loop.config import load_config
from the_loop.errors import ConfigError
from the_loop.policy.approvals import HeadlessApprover, InteractiveApprover

_BUDGET_STOP_REASONS = {
    "max_turns",
    "tool_budget_exceeded",
    "token_budget_exceeded",
    "wall_clock_budget_exceeded",
}


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
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        config = load_config(argv)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 3

    # Imported here so `loop` can re-export `main` without an import cycle.
    import loop as loop_module

    approver = (
        HeadlessApprover(config.policy, workspace=config.workspace)
        if config.headless
        else InteractiveApprover()
    )

    loop_module.configure(config)
    print(f"Workspace: {config.workspace}")

    result = loop_module.run_agent(config.task, config=config, approver=approver)
    print(f"\nStopped because: {result['stop_reason']}")

    if result["stop_reason"] == "final_answer":
        return 0
    if result["stop_reason"] in _BUDGET_STOP_REASONS:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
