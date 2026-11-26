"""Tiny policy-guarded coding agent: compatibility entrypoint.

The real loop lives in :mod:`the_loop.agent`; the CLI composition root lives in
:mod:`the_loop.cli`.  This module re-exports the historical symbols so existing
imports keep working and delegates execution to the new modules.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from the_loop.cli import build_parser
from the_loop.cli import main as cli_main
from the_loop.errors import ToolError
from the_loop.policy import approvals, paths
from the_loop.policy.approvals import session_approvals
from the_loop.policy.paths import path_starts_with
from the_loop.policy.paths import safe_path as _safe_path
from the_loop.policy.rules import (
    INTERNAL_PATHS,
    SENSITIVE_PATHS,
    SOURCE_PATHS,
    allow,
    ask,
    check_policy,
    deny,
)
from the_loop.tools.base import ToolResult
from the_loop.tools.fs import (
    IGNORED_NAMES,
    edit_file,
    list_dir,
    read_file,
    write_file,
)
from the_loop.tools.registry import (
    REQUIRED_ARGUMENTS,
    TOOL_ARGUMENTS,
    TOOL_FUNCTIONS,
    ToolOutcome,
    ToolRegistry,
    error_result,
    normalize_result,
    parse_arguments,
    policy_to_tool_result,
    require,
    tools,
    validate_tool_call,
)
from the_loop.tools.shell import run_shell, run_tests

__all__ = [
    "IGNORED_NAMES",
    "INTERNAL_PATHS",
    "REQUIRED_ARGUMENTS",
    "SENSITIVE_PATHS",
    "SOURCE_PATHS",
    "TOOL_ARGUMENTS",
    "TOOL_FUNCTIONS",
    "ToolError",
    "ToolOutcome",
    "ToolResult",
    "allow",
    "ask",
    "build_parser",
    "check_policy",
    "configure",
    "deny",
    "edit_file",
    "error_result",
    "handle_tool_call",
    "list_dir",
    "main",
    "normalize_result",
    "parse_args",
    "parse_arguments",
    "path_starts_with",
    "policy_to_tool_result",
    "prompt_for_approval",
    "read_file",
    "require",
    "run_agent",
    "run_shell",
    "run_tests",
    "safe_path",
    "session_approvals",
    "tools",
    "validate_tool_call",
    "write_file",
]

BASE_URL = os.getenv("LOOP_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LOOP_API_KEY", "lm-studio")
MODEL = os.getenv("LOOP_MODEL", "qwen/qwen2.5-vl-7b")
MAX_TURNS = 8
MAX_TOOL_CALLS = 12

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
WORKSPACE = Path.cwd().resolve()
paths.set_workspace(WORKSPACE)

_clients: dict[tuple[str, str], OpenAI] = {}


def _client_for_config(config) -> OpenAI:
    key = (config.base_url, config.api_key)
    cached = _clients.get(key)
    if cached is None:
        cached = OpenAI(base_url=config.base_url, api_key=config.api_key)
        _clients[key] = cached
    return cached


def configure(config) -> None:
    """Apply a RunConfig to module-level compatibility settings."""
    global WORKSPACE, MAX_TURNS, MAX_TOOL_CALLS, MODEL, BASE_URL, API_KEY, client

    WORKSPACE = config.workspace
    MAX_TURNS = config.max_turns
    MAX_TOOL_CALLS = config.max_tool_calls
    MODEL = config.model
    BASE_URL = config.base_url
    API_KEY = config.api_key
    client = _client_for_config(config)
    paths.set_workspace(WORKSPACE)


def safe_path(path: str) -> Path:
    """Compatibility wrapper: sandbox against this module's WORKSPACE."""
    return _safe_path(path, workspace=WORKSPACE)


def prompt_for_approval(policy: dict) -> bool:
    return approvals.request_approval(policy)


def execute_tool(name: str, arguments: dict):
    return TOOL_FUNCTIONS[name](**arguments)


def handle_tool_call(tool_call, *, config=None, approver=None, workspace=None) -> dict:
    """Compatibility wrapper around the policy-guarded ToolRegistry."""
    root = workspace or (config.workspace if config is not None else WORKSPACE)
    if approver is None:
        registry = ToolRegistry(workspace=root, legacy_approver=prompt_for_approval)
    else:
        registry = ToolRegistry(workspace=root, approver=approver)
    return registry.execute(tool_call).payload


def log_event(event: dict) -> None:
    print(json.dumps(event))


class _NullEventLogger:
    def emit(self, *_args, **_kwargs) -> None:
        return None


def run_agent(task: str, *, config=None, approver=None) -> dict:
    """Deprecated compatibility shim; new code should use :class:`Agent`."""
    from the_loop.agent import Agent
    from the_loop.config import RunConfig
    from the_loop.llm.client import LLMClient

    if config is None:
        config = RunConfig(
            task=task,
            workspace=WORKSPACE,
            base_url=BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            max_turns=MAX_TURNS,
            max_tool_calls=MAX_TOOL_CALLS,
        )

    registry = ToolRegistry(
        workspace=config.workspace,
        approver=approver,
        legacy_approver=prompt_for_approval if approver is None else None,
    )
    agent = Agent(
        llm_client=LLMClient(config),
        tool_registry=registry,
        event_logger=_NullEventLogger(),
        config=config,
        trace_path="",
    )
    summary = agent.run(task)
    stop_reason = {
        "resolved": "final_answer",
        "gave_up": "final_answer",
        "budget_exceeded": "max_turns",
        "error": "error",
    }.get(summary.status, summary.status)
    return {
        "messages": agent.messages,
        "stop_reason": stop_reason,
        "tool_call_count": summary.tool_calls,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main(argv: list[str] | None = None) -> int:
    """Delegate to the CLI module while keeping ``loop.main`` importable."""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
