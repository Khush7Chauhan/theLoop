"""Sandbox-only shell and test tools.

Neither tool ever runs a command on the host. ``run_shell`` executes
``sh -c <command>`` inside the Docker sandbox; ``run_tests`` runs pytest
strictly via argv (never through a shell). Both re-check the policy engine
immediately before invoking Docker.
"""

from __future__ import annotations

import shlex
import time

from the_loop.policy.approvals import is_approved, is_call_approved
from the_loop.policy.paths import get_workspace
from the_loop.policy.rules import check_policy
from the_loop.sandbox.docker import (
    DEFAULT_TIMEOUT_S,
    MAX_TIMEOUT_S,
    SandboxResult,
    run_in_sandbox,
)
from the_loop.tools.base import ToolResult

DEFAULT_TEST_TIMEOUT_S = 60

# run_tests is argv-only, but rejecting obvious shell syntax keeps a confused
# model from thinking it can use pipes, redirects, or command substitution.
_SHELL_METACHARACTERS = frozenset(";|&$`><\n\r\0*?~!")


def _elapsed_ms(start: float) -> int:
    return max(0, int(round((time.perf_counter() - start) * 1000)))


def _validate_timeout(timeout_s: int, start: float) -> ToolResult | None:
    if not isinstance(timeout_s, int) or isinstance(timeout_s, bool):
        return ToolResult.failure(
            "invalid_arguments",
            "timeout_s must be an integer.",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )
    if timeout_s < 1 or timeout_s > MAX_TIMEOUT_S:
        return ToolResult.failure(
            "invalid_arguments",
            f"timeout_s must be between 1 and {MAX_TIMEOUT_S}.",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )
    return None


def _policy_failure(
    name: str,
    arguments: dict,
    start: float,
) -> ToolResult | None:
    """Fail closed unless the policy allows this exact call.

    The interactive loop grants a one-shot approval for the duration of the
    call via ``approvals.approve_current_call()``. Session approvals are
    honored as well, so a direct tool call can still be authorized explicitly.
    """
    policy = check_policy(name, arguments)
    decision = policy["decision"]

    if decision == "deny":
        return ToolResult.failure(
            "permission",
            policy["message"],
            retryable=policy.get("retryable", False),
            duration_ms=_elapsed_ms(start),
        )

    if decision == "ask":
        if is_call_approved() or is_approved(policy.get("rule_hint")):
            return None
        return ToolResult.failure(
            "approval_required",
            policy["message"],
            retryable=False,
            duration_ms=_elapsed_ms(start),
        )

    if decision != "allow":
        return ToolResult.failure(
            "permission",
            f"Unrecognized policy decision: {decision}",
            retryable=False,
            duration_ms=_elapsed_ms(start),
        )

    return None


def _to_tool_result(result: SandboxResult, start: float) -> ToolResult:
    data = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "truncated": result.truncated,
        "container": result.container_name,
        "command": result.command,
    }
    duration_ms = _elapsed_ms(start)

    if result.ok:
        return ToolResult(
            ok=True,
            data=data,
            message=result.message,
            exit_code=result.exit_code,
            duration_ms=duration_ms,
        )

    return ToolResult(
        ok=False,
        data=data,
        error_type=result.error_type or "sandbox_error",
        retryable=result.retryable,
        message=result.message,
        exit_code=result.exit_code,
        duration_ms=duration_ms,
    )


def run_shell(command: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> ToolResult:
    """Run an arbitrary shell command inside the Docker sandbox.

    The host executes only ``docker run``; the command itself runs as
    ``sh -c <command>`` inside a network-less, read-only container.
    """
    start = time.perf_counter()

    if not isinstance(command, str) or not command.strip():
        return ToolResult.failure(
            "invalid_arguments",
            "command must be a non-empty string.",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )

    timeout_failure = _validate_timeout(timeout_s, start)
    if timeout_failure is not None:
        return timeout_failure

    policy_failure = _policy_failure("run_shell", {"command": command}, start)
    if policy_failure is not None:
        return policy_failure

    result = run_in_sandbox(
        ["sh", "-c", command],
        get_workspace(),
        timeout_s=timeout_s,
    )
    return _to_tool_result(result, start)


def run_tests(args: str = "-q", timeout_s: int = DEFAULT_TEST_TIMEOUT_S) -> ToolResult:
    """Run pytest inside the Docker sandbox using argv, never ``sh -c``."""
    start = time.perf_counter()

    if not isinstance(args, str):
        return ToolResult.failure(
            "invalid_arguments",
            "args must be a string of pytest arguments.",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )

    bad_characters = sorted(set(args) & _SHELL_METACHARACTERS)
    if bad_characters:
        return ToolResult.failure(
            "invalid_arguments",
            (
                "args must not contain shell metacharacters "
                f"({''.join(bad_characters)!r}). Pass pytest arguments as a "
                "plain string, for example '-q tests/'."
            ),
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )

    try:
        parsed_args = shlex.split(args)
    except ValueError as error:
        return ToolResult.failure(
            "invalid_arguments",
            f"Could not parse pytest arguments: {error}",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )

    timeout_failure = _validate_timeout(timeout_s, start)
    if timeout_failure is not None:
        return timeout_failure

    policy_failure = _policy_failure("run_tests", {"args": args}, start)
    if policy_failure is not None:
        return policy_failure

    result = run_in_sandbox(
        ["pytest", *parsed_args],
        get_workspace(),
        timeout_s=timeout_s,
    )
    return _to_tool_result(result, start)
