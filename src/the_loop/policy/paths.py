from pathlib import Path

from the_loop.errors import ToolError

_workspace: Path = Path.cwd().resolve()


def get_workspace() -> Path:
    return _workspace


def set_workspace(path: str | Path) -> Path:
    """Point the default sandbox root at ``path`` and return it resolved."""
    global _workspace
    _workspace = Path(path).expanduser().resolve()
    return _workspace


def safe_path(path: str, *, workspace: str | Path | None = None) -> Path:
    """Resolve ``path`` and refuse anything outside the workspace.

    ``Path.resolve()`` follows every symlink and junction in the path, so a
    link inside the workspace that points outside it resolves to the outside
    target and is rejected. Non-existent tails are resolved against their
    nearest existing ancestor, which means a write through an escaping link is
    rejected before the file is created.
    """
    root = Path(workspace).expanduser().resolve() if workspace is not None else _workspace

    try:
        raw = Path(path)
        candidate = raw if raw.is_absolute() else root / raw
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ToolError(
            "permission",
            "Path could not be resolved safely.",
            retryable=False,
        ) from error

    if not resolved.is_relative_to(root):
        raise ToolError(
            "permission",
            "Path escapes the workspace.",
            retryable=False,
        )
    return resolved


def path_starts_with(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        path == prefix.rstrip("/") or path.startswith(prefix)
        for prefix in prefixes
    )
