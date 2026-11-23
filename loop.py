"""Tiny policy-guarded coding agent.

This module is the CLI entrypoint and stays importable for backwards
compatibility: the policy engine and filesystem tools live in
``src/the_loop/`` and the symbols below are re-exported from there.
"""

import argparse
import json
import os
import time
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
from the_loop.tools.shell import run_shell, run_tests

__all__ = [
    "IGNORED_NAMES",
    "INTERNAL_PATHS",
    "SENSITIVE_PATHS",
    "SOURCE_PATHS",
    "ToolError",
    "ToolResult",
    "allow",
    "ask",
    "build_parser",
    "check_policy",
    "configure",
    "deny",
    "edit_file",
    "handle_tool_call",
    "list_dir",
    "main",
    "parse_args",
    "path_starts_with",
    "prompt_for_approval",
    "read_file",
    "run_agent",
    "run_shell",
    "run_tests",
    "safe_path",
    "session_approvals",
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


TOOL_FUNCTIONS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_shell": run_shell,
    "run_tests": run_tests,
}

REQUIRED_ARGUMENTS = {
    "list_dir": ["path"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "old_text", "new_text"],
    "run_shell": ["command"],
    "run_tests": [],
}

TOOL_ARGUMENTS = {
    "list_dir": {"path"},
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "edit_file": {"path", "old_text", "new_text"},
    "run_shell": {"command", "timeout_s"},
    "run_tests": {"args", "timeout_s"},
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List visible files and folders inside the workspace. "
                "Use this to inspect project structure. "
                "Path must be relative to the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file from the workspace. "
                "Use this when you need exact file contents. "
                "Path must be relative to the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a UTF-8 text file inside the workspace. "
                "Use this only when the task requires changing or creating a file. "
                "Path must be relative to the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace exactly one occurrence of old_text with new_text in an "
                "existing UTF-8 text file. old_text must match the file exactly, "
                "including whitespace, and must be unique; include surrounding "
                "lines when needed. Use write_file to create a new file. "
                "Path must be relative to the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace; must match once.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command inside an isolated Docker sandbox with no "
                "network access, a read-only root filesystem, and the workspace "
                "mounted at /workspace. Use this to inspect the environment or "
                "run commands that are not covered by the other tools. The host "
                "never executes the command directly. Requires human approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Shell command to run inside the sandbox via sh -c."
                        ),
                    },
                    "timeout_s": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 300,
                        "description": "Maximum runtime in seconds (default 30).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run pytest inside the Docker sandbox with the workspace mounted "
                "at /workspace. Arguments are passed directly to pytest as argv "
                "(never through a shell). Use this to check whether changes pass "
                "the test suite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": (
                            "pytest arguments, for example '-q tests/' or "
                            "'-q -k test_name' (default '-q')."
                        ),
                    },
                    "timeout_s": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 300,
                        "description": "Maximum runtime in seconds (default 60).",
                    },
                },
                "required": [],
            },
        },
    },
]


def parse_arguments(raw_arguments: str) -> dict:
    try:
        return json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as error:
        raise ToolError(
            "invalid_tool_arguments",
            f"Arguments were not valid JSON: {error}",
            retryable=True,
        )


def require(arguments: dict, *names: str) -> None:
    missing = [name for name in names if name not in arguments]
    if missing:
        raise ToolError(
            "missing_arguments",
            f"Missing required arguments: {', '.join(missing)}",
            retryable=True,
        )


def validate_tool_call(name: str, arguments: dict) -> None:
    if name not in REQUIRED_ARGUMENTS:
        raise ToolError("unknown_tool", f"Unknown tool: {name}", False)
    require(arguments, *REQUIRED_ARGUMENTS[name])
    unexpected = sorted(set(arguments) - TOOL_ARGUMENTS[name])
    if unexpected:
        raise ToolError(
            "unknown_arguments",
            f"Unknown arguments for {name}: {', '.join(unexpected)}",
            retryable=True,
        )


def policy_to_tool_result(name: str, policy: dict, *, reason: str | None = None) -> dict:
    result = {
        "ok": False,
        "tool": name,
        "error_type": "permission",
        "message": policy["message"],
        "retryable": policy.get("retryable", False),
        "needs_approval": policy["decision"] == "ask",
        "risk": policy.get("risk"),
        "rule_hint": policy.get("rule_hint"),
        "approval_scopes": policy.get("approval_scopes", []),
    }
    if reason is not None:
        result["reason"] = reason
    return result


def normalize_result(name: str, result) -> dict:
    if isinstance(result, ToolResult):
        payload = {
            "ok": result.ok,
            "tool": name,
        }
        if result.ok:
            payload["result"] = result.data
        else:
            payload.update(
                {
                    "error_type": result.error_type,
                    "message": result.message,
                    "retryable": result.retryable,
                }
            )
            if result.data is not None:
                payload["result"] = result.data
        if result.exit_code is not None:
            payload["exit_code"] = result.exit_code
        payload["duration_ms"] = result.duration_ms
        return payload
    return {"ok": True, "tool": name, "result": result}


def error_result(name: str, error: ToolError) -> dict:
    return {
        "ok": False,
        "tool": name,
        "error_type": error.error_type,
        "message": error.message,
        "retryable": error.retryable,
    }


def execute_tool(name: str, arguments: dict):
    return TOOL_FUNCTIONS[name](**arguments)


def handle_tool_call(tool_call, *, config=None, approver=None, workspace=None) -> dict:
    name = tool_call.function.name
    try:
        arguments = parse_arguments(tool_call.function.arguments)
        validate_tool_call(name, arguments)

        # Keep the package sandbox in sync with the active workspace so
        # --workspace, and code that swaps WORKSPACE, applies to every tool.
        root = workspace or (config.workspace if config is not None else WORKSPACE)
        paths.set_workspace(root)

        policy = check_policy(name, arguments)

        if policy["decision"] == "deny":
            return policy_to_tool_result(name, policy)

        if policy["decision"] == "ask":
            if approver is None:
                approved = prompt_for_approval(policy)
                reason = None if approved else "interactive_declined"
            else:
                decision = approver.approve(
                    policy,
                    tool=name,
                    arguments=arguments,
                )
                approved = decision.approved
                reason = decision.reason
            if not approved:
                return policy_to_tool_result(name, policy, reason=reason)

        if policy["decision"] not in {"allow", "ask"}:
            # Fail closed if a future policy returns an unknown decision.
            return policy_to_tool_result(
                name,
                deny(f"Unrecognized policy decision: {policy['decision']}"),
            )

        # Sandbox tools re-check policy before invoking Docker. This context
        # makes a one-shot interactive approval visible to that check without
        # trusting a model-supplied argument.
        with approvals.approve_current_call():
            result = execute_tool(name, arguments)
        return normalize_result(name, result)
    except ToolError as error:
        return error_result(name, error)


def log_event(event: dict) -> None:
    print(json.dumps(event))


def _call_model_with_usage(messages: list[dict], *, config=None):
    api_client = _client_for_config(config) if config is not None else client
    model = config.model if config is not None else MODEL
    response = api_client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return response.choices[0].message, total_tokens or 0


def call_model(messages: list[dict]):
    message, _ = _call_model_with_usage(messages)
    return message


def run_agent(task: str, *, config=None, approver=None) -> dict:
    workspace = config.workspace if config is not None else WORKSPACE
    max_turns = config.max_turns if config is not None else MAX_TURNS
    max_tool_calls = config.max_tool_calls if config is not None else MAX_TOOL_CALLS
    max_tokens = config.max_tokens if config is not None else None
    max_wall_clock_s = config.max_wall_clock_s if config is not None else None

    started = time.monotonic()
    token_count = 0

    messages = [
        {
            "role": "system",
            "content": (
                "You are a tiny coding agent. Use tools when you need to inspect "
                "or change the workspace. Paths must stay inside the workspace. "
                "When you have enough information, answer clearly."
            ),
        },
        {"role": "user", "content": task},
    ]

    tool_call_count = 0
    stop_reason = "max_turns"

    for turn in range(max_turns):
        if (
            max_wall_clock_s is not None
            and time.monotonic() - started >= max_wall_clock_s
        ):
            stop_reason = "wall_clock_budget_exceeded"
            break

        log_event({"type": "model_call", "turn": turn})
        message, usage = _call_model_with_usage(messages, config=config)
        token_count += usage
        messages.append(message.model_dump(exclude_none=True))

        if message.content:
            print("\nassistant:")
            print(message.content)

        if max_tokens is not None and token_count >= max_tokens:
            stop_reason = "token_budget_exceeded"
            break

        if not message.tool_calls:
            stop_reason = "final_answer"
            break

        for tool_call in message.tool_calls:
            if (
                max_wall_clock_s is not None
                and time.monotonic() - started >= max_wall_clock_s
            ):
                stop_reason = "wall_clock_budget_exceeded"
                break

            tool_call_count += 1
            if tool_call_count > max_tool_calls:
                stop_reason = "tool_budget_exceeded"
                break

            result = handle_tool_call(
                tool_call,
                config=config,
                approver=approver,
                workspace=workspace,
            )
            log_event(
                {
                    "type": "tool_result",
                    "turn": turn,
                    "tool": result.get("tool"),
                    "ok": result["ok"],
                    "error_type": result.get("error_type"),
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        if stop_reason != "max_turns":
            break

    return {
        "messages": messages,
        "stop_reason": stop_reason,
        "tool_call_count": tool_call_count,
    }


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main(argv: list[str] | None = None) -> int:
    """Delegate to the CLI module while keeping ``loop.main`` importable."""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
