"""Tiny policy-guarded coding agent.

This module is the CLI entrypoint and stays importable for backwards
compatibility: the policy engine and filesystem tools live in
``src/the_loop/`` and the symbols below are re-exported from there.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

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

__all__ = [
    "IGNORED_NAMES",
    "INTERNAL_PATHS",
    "SENSITIVE_PATHS",
    "SOURCE_PATHS",
    "ToolError",
    "ToolResult",
    "allow",
    "ask",
    "check_policy",
    "deny",
    "edit_file",
    "list_dir",
    "path_starts_with",
    "prompt_for_approval",
    "read_file",
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
}

REQUIRED_ARGUMENTS = {
    "list_dir": ["path"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "old_text", "new_text"],
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


def policy_to_tool_result(name: str, policy: dict) -> dict:
    return {
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


def normalize_result(name: str, result) -> dict:
    if isinstance(result, ToolResult):
        if not result.ok:
            return {
                "ok": False,
                "tool": name,
                "error_type": result.error_type,
                "message": result.message,
                "retryable": result.retryable,
                "duration_ms": result.duration_ms,
            }
        return {
            "ok": True,
            "tool": name,
            "result": result.data,
            "duration_ms": result.duration_ms,
        }
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


def handle_tool_call(tool_call) -> dict:
    name = tool_call.function.name
    try:
        arguments = parse_arguments(tool_call.function.arguments)
        validate_tool_call(name, arguments)

        # Keep the package sandbox in sync with this module's WORKSPACE so
        # --workspace, and code that swaps WORKSPACE, applies to every tool.
        paths.set_workspace(WORKSPACE)

        policy = check_policy(name, arguments)

        if policy["decision"] == "deny":
            return policy_to_tool_result(name, policy)

        if policy["decision"] == "ask" and not prompt_for_approval(policy):
            return policy_to_tool_result(name, policy)

        if policy["decision"] != "allow":
            # Fail closed if a future policy returns an unknown decision.
            return policy_to_tool_result(
                name,
                deny(f"Unrecognized policy decision: {policy['decision']}"),
            )

        result = execute_tool(name, arguments)
        return normalize_result(name, result)
    except ToolError as error:
        return error_result(name, error)


def log_event(event: dict) -> None:
    print(json.dumps(event))


def call_model(messages: list[dict]):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0].message


def run_agent(task: str) -> dict:
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

    for turn in range(MAX_TURNS):
        log_event({"type": "model_call", "turn": turn})
        message = call_model(messages)
        messages.append(message.model_dump(exclude_none=True))

        if message.content:
            print("\nassistant:")
            print(message.content)

        if not message.tool_calls:
            stop_reason = "final_answer"
            break

        for tool_call in message.tool_calls:
            tool_call_count += 1
            if tool_call_count > MAX_TOOL_CALLS:
                stop_reason = "tool_budget_exceeded"
                break

            result = handle_tool_call(tool_call)
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
    parser = argparse.ArgumentParser(description="Run the tiny coding agent loop.")
    parser.add_argument("task", help="The task for the agent to perform.")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace directory the agent can inspect and edit.",
    )
    return parser.parse_args()


def main() -> int:
    global WORKSPACE

    args = parse_args()
    WORKSPACE = Path(args.workspace).expanduser().resolve()
    if not WORKSPACE.exists() or not WORKSPACE.is_dir():
        print(f"Workspace does not exist: {WORKSPACE}", file=sys.stderr)
        return 3

    paths.set_workspace(WORKSPACE)
    print(f"Workspace: {WORKSPACE}")
    result = run_agent(args.task)
    print(f"\nStopped because: {result['stop_reason']}")

    if result["stop_reason"] == "final_answer":
        return 0
    if result["stop_reason"] in {"max_turns", "tool_budget_exceeded"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
