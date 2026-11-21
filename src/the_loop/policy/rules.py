from the_loop.errors import ToolError
from the_loop.policy.paths import path_starts_with, safe_path

SENSITIVE_PATHS = (".env", "secrets/")
INTERNAL_PATHS = (".loop/",)
SOURCE_PATHS = ("src/", "tests/")

READ_ACTIONS = frozenset({"read_file"})
WRITE_ACTIONS = frozenset({"write_file", "edit_file"})
KNOWN_ACTIONS = frozenset({"list_dir", "read_file", "write_file", "edit_file"})


def allow() -> dict:
    return {"decision": "allow"}


def deny(message: str) -> dict:
    return {
        "decision": "deny",
        "message": message,
        "retryable": False,
    }


def ask(message: str, risk: str, rule_hint: str) -> dict:
    return {
        "decision": "ask",
        "message": message,
        "risk": risk,
        "rule_hint": rule_hint,
        "approval_scopes": ["once", "session"],
        "retryable": False,
    }


def check_policy(name: str, arguments: dict) -> dict:
    """Decide whether a tool call may run.

    Anything the policy does not explicitly recognise is denied: unknown
    actions, malformed paths, and paths that leave the workspace (including
    through symlinks) never reach a tool.
    """
    if name not in KNOWN_ACTIONS:
        return deny(f"Unrecognized action: {name}")

    path = arguments.get("path", "")
    if not isinstance(path, str):
        return deny("The path argument must be a string.")

    try:
        safe_path(path)
    except ToolError as error:
        return deny(error.message)

    if name in READ_ACTIONS and path_starts_with(path, SENSITIVE_PATHS):
        return deny("The agent may not read sensitive files.")

    if name in WRITE_ACTIONS and path_starts_with(path, INTERNAL_PATHS):
        return deny("The agent may not write internal loop state.")

    if name in WRITE_ACTIONS and path_starts_with(path, SENSITIVE_PATHS):
        return deny("The agent may not write secrets.")

    if name in WRITE_ACTIONS and path_starts_with(path, SOURCE_PATHS):
        return ask(
            message=f"The agent wants to modify source code: {path}",
            risk="changes code that may affect behavior",
            rule_hint=f"write_file:{path}",
        )

    return allow()
