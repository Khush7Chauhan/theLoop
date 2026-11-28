"""Tool schemas, argument validation, policy enforcement, and execution.

The agent sees only this registry interface; policy decisions, approvals,
Docker sandboxing, and filesystem access stay behind it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from the_loop.errors import ToolError
from the_loop.policy import approvals, paths
from the_loop.policy.rules import allow, check_policy, deny
from the_loop.tools.base import ToolResult
from the_loop.tools.fs import edit_file, list_dir, read_file, write_file
from the_loop.tools.shell import run_shell, run_tests

TOOL_FUNCTIONS: dict[str, Callable[..., ToolResult]] = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_shell": run_shell,
    "run_tests": run_tests,
}

REQUIRED_ARGUMENTS: dict[str, list[str]] = {
    "list_dir": ["path"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "old_text", "new_text"],
    "run_shell": ["command"],
    "run_tests": [],
}

TOOL_ARGUMENTS: dict[str, set[str]] = {
    "list_dir": {"path"},
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "edit_file": {"path", "old_text", "new_text"},
    "run_shell": {"command", "timeout_s"},
    "run_tests": {"args", "timeout_s"},
}

tools: list[dict] = [
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


@dataclass(frozen=True)
class ToolOutcome:
    """Result of one registry call, including the effective policy decision."""

    tool: str
    ok: bool
    payload: dict
    decision: str
    reason: str | None = None
    rule_hint: str | None = None
    error_type: str | None = None
    retryable: bool = False
    duration_ms: int = 0
    exit_code: int | None = None


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


def normalize_result(name: str, result: Any) -> dict:
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


def _tool_call_parts(tool_call: Any) -> tuple[str, str]:
    function = getattr(tool_call, "function", None)
    if function is not None:
        return (
            str(getattr(function, "name", "") or ""),
            str(getattr(function, "arguments", "{}") or "{}"),
        )
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        if isinstance(function, dict):
            return (
                str(function.get("name", "") or ""),
                str(function.get("arguments", "{}") or "{}"),
            )
        return (
            str(tool_call.get("name", "") or ""),
            str(tool_call.get("arguments", "{}") or "{}"),
        )
    return (
        str(getattr(tool_call, "name", "") or ""),
        str(getattr(tool_call, "arguments", "{}") or "{}"),
    )


class ToolRegistry:
    """Policy-guarded tool dispatch with no logging side effects."""

    def __init__(
        self,
        *,
        workspace: str | Path | None = None,
        approver: Any | None = None,
        legacy_approver: Callable[[dict], bool] | None = None,
        tool_functions: dict[str, Callable[..., ToolResult]] | None = None,
    ):
        self.workspace = (
            Path(workspace).expanduser().resolve() if workspace is not None else None
        )
        self.approver = approver
        self.legacy_approver = legacy_approver
        self.tool_functions = dict(tool_functions or TOOL_FUNCTIONS)

    def tool_schemas(self) -> list[dict]:
        return list(tools)

    def execute(self, tool_call: Any) -> ToolOutcome:
        name, raw_arguments = _tool_call_parts(tool_call)
        try:
            arguments = parse_arguments(raw_arguments)
            validate_tool_call(name, arguments)
        except ToolError as error:
            return ToolOutcome(
                tool=name,
                ok=False,
                payload=error_result(name, error),
                decision="error",
                reason=error.error_type,
                error_type=error.error_type,
                retryable=error.retryable,
            )

        if self.workspace is not None:
            paths.set_workspace(self.workspace)

        policy = check_policy(name, arguments)
        decision = policy["decision"]
        rule_hint = policy.get("rule_hint")

        if decision == "deny":
            return ToolOutcome(
                tool=name,
                ok=False,
                payload=policy_to_tool_result(name, policy),
                decision="deny",
                reason=policy.get("message"),
                rule_hint=rule_hint,
                error_type="permission",
                retryable=False,
            )

        reason: str | None = None
        if decision == "ask":
            approved = False
            if self.approver is not None:
                approval = self.approver.approve(
                    policy,
                    tool=name,
                    arguments=arguments,
                )
                approved = bool(approval.approved)
                reason = approval.reason
            elif self.legacy_approver is not None:
                approved = bool(self.legacy_approver(policy))
                reason = None if approved else "approval_declined"
            else:
                reason = "no_approver"

            if not approved:
                return ToolOutcome(
                    tool=name,
                    ok=False,
                    payload=policy_to_tool_result(name, policy, reason=reason),
                    decision="deny",
                    reason=reason,
                    rule_hint=rule_hint,
                    error_type="permission",
                    retryable=False,
                )
        elif decision != "allow":
            policy = deny(f"Unrecognized policy decision: {decision}")
            return ToolOutcome(
                tool=name,
                ok=False,
                payload=policy_to_tool_result(name, policy),
                decision="deny",
                reason="unknown_policy_decision",
                rule_hint=rule_hint,
                error_type="permission",
                retryable=False,
            )

        try:
            with approvals.approve_current_call():
                result = self.tool_functions[name](**arguments)
            payload = normalize_result(name, result)
        except Exception as error:  # noqa: BLE001 - tools must not crash the loop
            tool_error = ToolError(
                "internal_error",
                f"{type(error).__name__}: {error}",
                retryable=False,
            )
            payload = error_result(name, tool_error)
            return ToolOutcome(
                tool=name,
                ok=False,
                payload=payload,
                decision="allow",
                reason=reason,
                rule_hint=rule_hint,
                error_type=tool_error.error_type,
                retryable=False,
            )

        return ToolOutcome(
            tool=name,
            ok=bool(payload.get("ok")),
            payload=payload,
            decision="allow",
            reason=reason,
            rule_hint=rule_hint,
            error_type=payload.get("error_type"),
            retryable=bool(payload.get("retryable", False)),
            duration_ms=int(payload.get("duration_ms", 0) or 0),
            exit_code=payload.get("exit_code"),
        )


__all__ = [
    "REQUIRED_ARGUMENTS",
    "TOOL_ARGUMENTS",
    "TOOL_FUNCTIONS",
    "ToolOutcome",
    "ToolRegistry",
    "allow",
    "error_result",
    "normalize_result",
    "parse_arguments",
    "policy_to_tool_result",
    "require",
    "tools",
    "validate_tool_call",
]
