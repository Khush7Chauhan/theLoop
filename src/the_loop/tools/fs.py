import time

from the_loop.errors import ToolError
from the_loop.policy.paths import safe_path
from the_loop.tools.base import ToolResult

IGNORED_NAMES = {
    ".git",
    ".loop",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}


def _elapsed_ms(start: float) -> int:
    return max(0, int(round((time.perf_counter() - start) * 1000)))


def _failure_from_tool_error(error: ToolError, start: float) -> ToolResult:
    return ToolResult.failure(
        error.error_type,
        error.message,
        retryable=error.retryable,
        duration_ms=_elapsed_ms(start),
    )


def _failure_from_os_error(error: OSError, path: str, start: float) -> ToolResult:
    if isinstance(error, FileNotFoundError):
        return ToolResult.failure(
            "not_found",
            f"No such file or directory: {path}",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )
    return ToolResult.failure(
        "io_error",
        f"{type(error).__name__}: {error}",
        retryable=False,
        duration_ms=_elapsed_ms(start),
    )


def list_dir(path: str) -> ToolResult:
    start = time.perf_counter()
    try:
        target = safe_path(path)
        entries = sorted(
            entry.name
            for entry in target.iterdir()
            if entry.name not in IGNORED_NAMES
        )
    except ToolError as error:
        return _failure_from_tool_error(error, start)
    except OSError as error:
        return _failure_from_os_error(error, path, start)

    return ToolResult.success(entries, duration_ms=_elapsed_ms(start))


def read_file(path: str) -> ToolResult:
    start = time.perf_counter()
    try:
        target = safe_path(path)
        text = target.read_text(encoding="utf-8")
    except ToolError as error:
        return _failure_from_tool_error(error, start)
    except UnicodeDecodeError:
        return ToolResult.failure(
            "invalid_encoding",
            f"{path} is not valid UTF-8 text.",
            retryable=False,
            duration_ms=_elapsed_ms(start),
        )
    except OSError as error:
        return _failure_from_os_error(error, path, start)

    return ToolResult.success(text, duration_ms=_elapsed_ms(start))


def write_file(path: str, content: str) -> ToolResult:
    start = time.perf_counter()
    try:
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except ToolError as error:
        return _failure_from_tool_error(error, start)
    except OSError as error:
        return _failure_from_os_error(error, path, start)

    return ToolResult.success(
        f"Wrote {path}",
        duration_ms=_elapsed_ms(start),
    )


def edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
    """Replace exactly one occurrence of ``old_text`` with ``new_text``.

    The file must already exist. Ambiguous matches are refused rather than
    guessed at, so the model has to provide more context and retry.
    """
    start = time.perf_counter()

    if not isinstance(old_text, str) or not old_text:
        return ToolResult.failure(
            "invalid_arguments",
            "old_text must be a non-empty string.",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )
    if not isinstance(new_text, str):
        return ToolResult.failure(
            "invalid_arguments",
            "new_text must be a string.",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )

    try:
        target = safe_path(path)
        content = target.read_text(encoding="utf-8")
    except ToolError as error:
        return _failure_from_tool_error(error, start)
    except FileNotFoundError:
        return ToolResult.failure(
            "not_found",
            f"Cannot edit {path}: the file does not exist.",
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )
    except UnicodeDecodeError:
        return ToolResult.failure(
            "invalid_encoding",
            f"{path} is not valid UTF-8 text.",
            retryable=False,
            duration_ms=_elapsed_ms(start),
        )
    except OSError as error:
        return _failure_from_os_error(error, path, start)

    matches = content.count(old_text)
    if matches == 0:
        return ToolResult.failure(
            "not_found",
            (
                f"old_text was not found in {path}. Read the file and copy "
                "the exact text, including whitespace and indentation."
            ),
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )
    if matches > 1:
        return ToolResult.failure(
            "not_unique",
            (
                f"old_text appears {matches} times in {path}. Include more "
                "surrounding context so it matches exactly once."
            ),
            retryable=True,
            duration_ms=_elapsed_ms(start),
        )

    updated = content.replace(old_text, new_text, 1)
    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as error:
        return _failure_from_os_error(error, path, start)

    return ToolResult.success(
        f"Edited {path}",
        duration_ms=_elapsed_ms(start),
    )
