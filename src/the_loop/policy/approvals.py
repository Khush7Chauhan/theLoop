"""Approval strategies for ``ask`` policy decisions.

The policy engine decides *whether* an action needs approval; an approver
decides *whether to grant it*.  Interactive runs ask a human, while headless
runs consult a static allow/deny policy and fail closed for anything left
unresolved.
"""

from __future__ import annotations

import fnmatch
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from the_loop.errors import ToolError
from the_loop.policy.paths import get_workspace, safe_path
from the_loop.policy.rules import INTERNAL_PATHS, SENSITIVE_PATHS, path_starts_with

session_approvals: set[str] = set()

# Hard denials are enforced before any policy-file override is consulted, so a
# permissive allow-list can never expose secrets, internal state, or paths that
# escape the workspace.
HARD_DENY_PATHS = SENSITIVE_PATHS + INTERNAL_PATHS

# Approval granted by the interactive loop for the duration of a single tool
# call. This keeps "yes, once" truly one-shot: the tool can re-check policy
# without weakening the approval semantics or trusting model-supplied flags.
_call_approved: ContextVar[bool] = ContextVar(
    "the_loop_call_approved",
    default=False,
)


@dataclass(frozen=True)
class ApprovalDecision:
    """Outcome of an approval request, including a machine-readable reason."""

    approved: bool
    reason: str | None = None
    rule_hint: str | None = None


def reset_session_approvals() -> None:
    session_approvals.clear()


def is_approved(rule_hint: str | None) -> bool:
    return bool(rule_hint) and rule_hint in session_approvals


def is_call_approved() -> bool:
    """Return True while an approved tool call is executing."""
    return _call_approved.get()


@contextmanager
def approve_current_call():
    """Mark the current tool call as approved by the interactive loop."""
    token = _call_approved.set(True)
    try:
        yield
    finally:
        _call_approved.reset(token)


def request_approval(policy: dict, *, input_fn=input, output_fn=print) -> bool:
    """Ask the human to approve an ``ask`` policy decision.

    Returns True for "once" and "session" approvals; a "session" approval is
    remembered by its rule hint so the same path is not asked twice.
    """
    rule_hint = policy.get("rule_hint")
    if is_approved(rule_hint):
        return True

    output_fn("\nApproval needed")
    output_fn(policy["message"])
    output_fn(f"Risk: {policy.get('risk', 'unknown')}")
    output_fn("\nAllow?")
    output_fn("  y  yes, once")
    output_fn("  s  yes, for this session")
    output_fn("  n  no")

    choice = input_fn("> ").strip().lower()
    if choice == "s" and rule_hint:
        session_approvals.add(rule_hint)
        return True
    return choice == "y"


class InteractiveApprover:
    """Prompt a human on stdin, preserving the Phase 1 approval semantics."""

    def __init__(self, *, input_fn=input, output_fn=print):
        self.input_fn = input_fn
        self.output_fn = output_fn

    def approve(
        self,
        policy: dict,
        *,
        tool: str | None = None,
        arguments: dict | None = None,
    ) -> ApprovalDecision:
        approved = request_approval(
            policy,
            input_fn=self.input_fn,
            output_fn=self.output_fn,
        )
        return ApprovalDecision(
            approved=approved,
            reason=None if approved else "interactive_declined",
            rule_hint=policy.get("rule_hint"),
        )


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _has_hard_deny_segment(path: str) -> bool:
    for segment in _normalize_path(path).split("/"):
        if segment == ".env" or segment.startswith(".env."):
            return True
        if segment in {"secrets", ".loop"}:
            return True
    return False


def _matches_path(path: str, patterns: list[str]) -> bool:
    normalized = _normalize_path(path)
    if not normalized:
        return False
    for pattern in patterns:
        candidate = _normalize_path(pattern)
        if not candidate:
            continue
        if fnmatch.fnmatchcase(normalized, candidate):
            return True
        if not any(char in candidate for char in "*?["):
            if normalized == candidate or normalized.startswith(candidate.rstrip("/") + "/"):
                return True
    return False


class HeadlessApprover:
    """Resolve approvals from a static policy file without prompting.

    Unresolved ``ask`` decisions are denied with ``headless_unresolved_ask``.
    Hard denials and path escapes are checked first and cannot be overridden
    by an allow entry.
    """

    def __init__(
        self,
        policy: dict | str | Path | None = None,
        *,
        workspace: str | Path | None = None,
    ):
        if isinstance(policy, (str, Path)):
            from the_loop.config import load_policy_file

            policy = load_policy_file(policy)
        policy = policy or {}
        allow = policy.get("allow", {})
        deny = policy.get("deny", {})
        self.allow_tools = set(allow.get("tools", []))
        self.allow_paths = list(allow.get("paths", []))
        self.deny_tools = set(deny.get("tools", []))
        self.deny_paths = list(deny.get("paths", []))
        self.workspace = (
            Path(workspace).expanduser().resolve()
            if workspace is not None
            else get_workspace()
        )

    def _relative_path(self, raw: str) -> str | None:
        """Return a workspace-relative POSIX path, or None if it escapes."""
        try:
            resolved = safe_path(raw, workspace=self.workspace)
        except ToolError:
            return None
        return resolved.relative_to(self.workspace).as_posix()

    def _hard_denial(self, tool: str | None, arguments: dict) -> str | None:
        raw_path = arguments.get("path")
        if isinstance(raw_path, str) and raw_path:
            relative = self._relative_path(raw_path)
            if relative is None:
                return "headless_path_escape"
            if path_starts_with(relative, HARD_DENY_PATHS):
                return "headless_hard_deny"
            if _has_hard_deny_segment(relative):
                return "headless_hard_deny"
            if path_starts_with(_normalize_path(raw_path), HARD_DENY_PATHS):
                return "headless_hard_deny"

        if tool == "run_shell":
            command = arguments.get("command")
            if isinstance(command, str):
                lowered = command.lower()
                if any(token in lowered for token in (".env", "secrets/", ".loop/")):
                    return "headless_hard_deny"
        return None

    def approve(
        self,
        policy: dict,
        *,
        tool: str | None = None,
        arguments: dict | None = None,
    ) -> ApprovalDecision:
        arguments = arguments or {}
        rule_hint = policy.get("rule_hint")

        if policy.get("decision") != "ask":
            return ApprovalDecision(True, reason=None, rule_hint=rule_hint)

        hard_reason = self._hard_denial(tool, arguments)
        if hard_reason is not None:
            return ApprovalDecision(False, reason=hard_reason, rule_hint=rule_hint)

        if tool and tool in self.deny_tools:
            return ApprovalDecision(False, reason="headless_denied_tool", rule_hint=rule_hint)

        raw_path = arguments.get("path")
        if isinstance(raw_path, str) and raw_path:
            relative = self._relative_path(raw_path)
            if relative is not None:
                if _matches_path(relative, self.deny_paths) or _matches_path(
                    raw_path, self.deny_paths
                ):
                    return ApprovalDecision(
                        False, reason="headless_denied_path", rule_hint=rule_hint
                    )

        if tool and tool in self.allow_tools:
            return ApprovalDecision(True, reason="headless_allowed_tool", rule_hint=rule_hint)

        if isinstance(raw_path, str) and raw_path:
            relative = self._relative_path(raw_path)
            if relative is not None and (
                _matches_path(relative, self.allow_paths)
                or _matches_path(raw_path, self.allow_paths)
            ):
                return ApprovalDecision(
                    True, reason="headless_allowed_path", rule_hint=rule_hint
                )

        return ApprovalDecision(
            False,
            reason="headless_unresolved_ask",
            rule_hint=rule_hint,
        )
