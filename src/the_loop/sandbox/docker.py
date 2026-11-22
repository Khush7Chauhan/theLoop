"""Docker sandbox runner.

This is the only module allowed to invoke the Docker CLI. Agent-chosen
commands are never executed on the host: they are passed as arguments to
``docker run`` inside a hardened, network-less container.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IMAGE = "the-loop-sandbox:latest"
DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_OUTPUT_CHARS = 16000
MAX_TIMEOUT_S = 300
REMOVE_TIMEOUT_S = 10

_TRUNCATION_NOTICE = "\n[TRUNCATED]"

_DAEMON_ERROR_MARKERS = (
    "cannot connect to the docker daemon",
    "error during connect",
    "is the docker daemon running",
    "docker daemon is not running",
    "dockerdesktoplinuxengine",
    "permission denied while trying to connect to the docker daemon socket",
)

_MISSING_IMAGE_MARKERS = (
    "unable to find image",
    "no such image",
    "pull access denied",
    "manifest unknown",
)


@dataclass(frozen=True)
class SandboxResult:
    """Structured result from a single sandboxed command."""

    ok: bool
    command: list[str]
    container_name: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None
    message: str = ""
    retryable: bool = False
    duration_ms: int = 0
    truncated: bool = False

    @property
    def timed_out(self) -> bool:
        return self.error_type == "timeout"


def _elapsed_ms(start: float) -> int:
    return max(0, int(round((time.perf_counter() - start) * 1000)))


def _coerce_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _clip(text: str, limit: int = 2000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_NOTICE)] + _TRUNCATION_NOTICE


def _truncate_to_budget(text: str, budget: int) -> tuple[str, bool]:
    if budget <= 0:
        return "", bool(text)
    if len(text) <= budget:
        return text, False
    if budget <= len(_TRUNCATION_NOTICE):
        return _TRUNCATION_NOTICE[:budget], True
    return text[: budget - len(_TRUNCATION_NOTICE)] + _TRUNCATION_NOTICE, True


def _bound_output(
    stdout: str,
    stderr: str,
    max_chars: int,
) -> tuple[str, str, bool]:
    """Bound stdout+stderr to ``max_chars`` while keeping stderr useful.

    The ``[TRUNCATED]`` marker is counted inside the budget, so callers can
    rely on ``len(stdout) + len(stderr) <= max_chars``.
    """
    max_chars = max(1, max_chars)
    if len(stdout) + len(stderr) <= max_chars:
        return stdout, stderr, False

    # Reserve at most a quarter of the budget for stderr. If stderr is short,
    # stdout gets the remainder; if stdout is short, stderr gets the remainder.
    stderr_budget = min(len(stderr), max(1, max_chars // 4))
    stdout_budget = max_chars - stderr_budget

    if len(stdout) < stdout_budget:
        stdout_budget = len(stdout)
        stderr_budget = max_chars - stdout_budget

    stdout_out, stdout_truncated = _truncate_to_budget(stdout, stdout_budget)
    stderr_out, stderr_truncated = _truncate_to_budget(stderr, stderr_budget)
    return stdout_out, stderr_out, stdout_truncated or stderr_truncated


def _normalize_command(command: list[str] | str) -> list[str]:
    if isinstance(command, str):
        return ["sh", "-c", command]
    if isinstance(command, (list, tuple)):
        if not command:
            raise ValueError("command must not be empty.")
        if not all(isinstance(part, str) for part in command):
            raise ValueError("every command argument must be a string.")
        return list(command)
    raise ValueError("command must be a string or a list of strings.")


def _docker_run_argv(
    container_name: str,
    workspace: Path,
    image: str,
    command: list[str],
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        # Never pull implicitly: a missing image should fail fast and clearly.
        "--pull=never",
        "--network=none",
        "--cpus=1.0",
        "--memory=512m",
        "--memory-swap=512m",
        "--pids-limit=64",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{workspace}:/workspace",
        "-w",
        "/workspace",
        image,
        *command,
    ]


def _run_docker_cli(
    argv: list[str],
    *,
    timeout_s: int = 15,
) -> subprocess.CompletedProcess | None:
    """Run a short Docker CLI query, returning None if Docker is unusable."""
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _force_remove(container_name: str) -> None:
    """Best-effort removal of a container whose client timed out.

    ``--rm`` normally handles cleanup, but a killed Docker client can leave
    the container running. This is still a Docker CLI invocation; it never
    executes the agent's command on the host.
    """
    _run_docker_cli(
        ["docker", "rm", "-f", container_name],
        timeout_s=REMOVE_TIMEOUT_S,
    )


def docker_available() -> bool:
    """Return True when the Docker daemon answers ``docker info``."""
    completed = _run_docker_cli(["docker", "info"])
    return completed is not None and completed.returncode == 0


def image_available(image: str = DEFAULT_IMAGE) -> bool:
    """Return True when ``image`` exists locally."""
    completed = _run_docker_cli(["docker", "image", "inspect", image])
    return completed is not None and completed.returncode == 0


def container_exists(container_name: str) -> bool:
    """Return True when a container with exactly ``container_name`` exists."""
    completed = _run_docker_cli(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={container_name}",
            "--format",
            "{{.Names}}",
        ]
    )
    if completed is None or completed.returncode != 0:
        return False
    return container_name in completed.stdout.splitlines()


def _looks_like_daemon_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _DAEMON_ERROR_MARKERS)


def _looks_like_missing_image(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _MISSING_IMAGE_MARKERS)


def _sandbox_unavailable_message(stderr: str, image: str) -> str:
    if _looks_like_missing_image(stderr):
        return (
            f"Sandbox image '{image}' is not available. Build it with: "
            f"docker build -f Dockerfile.sandbox -t {image} ."
        )
    detail = _clip(stderr)
    return f"Docker is not available. Start Docker and retry. {detail}".strip()


def _result(
    *,
    ok: bool,
    command: list[str],
    container_name: str | None,
    start: float,
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    error_type: str | None = None,
    message: str = "",
    retryable: bool = False,
    truncated: bool = False,
) -> SandboxResult:
    return SandboxResult(
        ok=ok,
        command=command,
        container_name=container_name,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        error_type=error_type,
        message=message,
        retryable=retryable,
        duration_ms=_elapsed_ms(start),
        truncated=truncated,
    )


def run_in_sandbox(
    command: list[str] | str,
    workspace: Path,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    image: str = DEFAULT_IMAGE,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> SandboxResult:
    """Run ``command`` in a hardened Docker container.

    ``command`` is either an argv list (preferred) or a string, in which case
    it is executed as ``sh -c <command>`` *inside* the container. The host
    only ever executes the Docker CLI.
    """
    start = time.perf_counter()

    try:
        normalized_command = _normalize_command(command)
    except ValueError as error:
        return _result(
            ok=False,
            command=[],
            container_name=None,
            start=start,
            error_type="invalid_arguments",
            message=str(error),
            retryable=True,
        )

    if not isinstance(timeout_s, int) or isinstance(timeout_s, bool):
        return _result(
            ok=False,
            command=normalized_command,
            container_name=None,
            start=start,
            error_type="invalid_arguments",
            message="timeout_s must be an integer.",
            retryable=True,
        )
    if timeout_s < 1 or timeout_s > MAX_TIMEOUT_S:
        return _result(
            ok=False,
            command=normalized_command,
            container_name=None,
            start=start,
            error_type="invalid_arguments",
            message=f"timeout_s must be between 1 and {MAX_TIMEOUT_S}.",
            retryable=True,
        )

    if not isinstance(max_output_chars, int) or isinstance(max_output_chars, bool):
        return _result(
            ok=False,
            command=normalized_command,
            container_name=None,
            start=start,
            error_type="invalid_arguments",
            message="max_output_chars must be an integer.",
            retryable=True,
        )
    if max_output_chars < 1:
        return _result(
            ok=False,
            command=normalized_command,
            container_name=None,
            start=start,
            error_type="invalid_arguments",
            message="max_output_chars must be at least 1.",
            retryable=True,
        )

    if not isinstance(image, str) or not image.strip():
        return _result(
            ok=False,
            command=normalized_command,
            container_name=None,
            start=start,
            error_type="invalid_arguments",
            message="image must be a non-empty string.",
            retryable=True,
        )

    try:
        resolved_workspace = Path(workspace).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as error:
        return _result(
            ok=False,
            command=normalized_command,
            container_name=None,
            start=start,
            error_type="invalid_workspace",
            message=f"Workspace could not be resolved: {error}",
            retryable=False,
        )

    if not resolved_workspace.is_dir():
        return _result(
            ok=False,
            command=normalized_command,
            container_name=None,
            start=start,
            error_type="invalid_workspace",
            message=f"Workspace is not a directory: {resolved_workspace}",
            retryable=False,
        )

    container_name = f"the-loop-{uuid.uuid4().hex[:12]}"
    argv = _docker_run_argv(
        container_name,
        resolved_workspace,
        image,
        normalized_command,
    )

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        _force_remove(container_name)
        stdout, stderr, truncated = _bound_output(
            _coerce_output(error.stdout),
            _coerce_output(error.stderr),
            max_output_chars,
        )
        return _result(
            ok=False,
            command=normalized_command,
            container_name=container_name,
            start=start,
            stdout=stdout,
            stderr=stderr,
            error_type="timeout",
            message=(
                f"Command timed out after {timeout_s}s; "
                f"container {container_name} was removed."
            ),
            retryable=True,
            truncated=truncated,
        )
    except FileNotFoundError:
        return _result(
            ok=False,
            command=normalized_command,
            container_name=container_name,
            start=start,
            error_type="sandbox_unavailable",
            message=(
                "Docker CLI was not found. Install Docker and make sure "
                "'docker' is on PATH."
            ),
            retryable=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        _force_remove(container_name)
        return _result(
            ok=False,
            command=normalized_command,
            container_name=container_name,
            start=start,
            error_type="sandbox_error",
            message=f"Could not run the sandbox container: {error}",
            retryable=False,
        )

    stdout, stderr, truncated = _bound_output(
        _coerce_output(completed.stdout),
        _coerce_output(completed.stderr),
        max_output_chars,
    )
    exit_code = completed.returncode
    lowered_stderr = _coerce_output(completed.stderr).lower()
    docker_prefixed = lowered_stderr.lstrip().startswith("docker:")

    if exit_code == 0:
        return _result(
            ok=True,
            command=normalized_command,
            container_name=container_name,
            start=start,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            message="Command exited with status 0.",
            truncated=truncated,
        )

    if exit_code == 125 or docker_prefixed:
        if _looks_like_daemon_error(lowered_stderr) or _looks_like_missing_image(
            lowered_stderr
        ):
            return _result(
                ok=False,
                command=normalized_command,
                container_name=container_name,
                start=start,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                error_type="sandbox_unavailable",
                message=_sandbox_unavailable_message(
                    _coerce_output(completed.stderr),
                    image,
                ),
                retryable=False,
                truncated=truncated,
            )

        return _result(
            ok=False,
            command=normalized_command,
            container_name=container_name,
            start=start,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            error_type="sandbox_error",
            message=(
                "Docker could not start the sandbox container: "
                f"{_clip(_coerce_output(completed.stderr))}"
            ).strip(),
            retryable=False,
            truncated=truncated,
        )

    return _result(
        ok=False,
        command=normalized_command,
        container_name=container_name,
        start=start,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        error_type="command_failed",
        message=f"Command exited with status {exit_code}.",
        retryable=True,
        truncated=truncated,
    )
