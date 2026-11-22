import re
import subprocess

import pytest

import the_loop.sandbox.docker as docker


def completed(argv, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        argv,
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


def capture_run(monkeypatch, result):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return result(argv)

    monkeypatch.setattr(docker.subprocess, "run", fake_run)
    return calls


# --- hardening ---------------------------------------------------------------


def test_run_in_sandbox_uses_strict_hardening_flags(workspace, monkeypatch):
    calls = capture_run(
        monkeypatch,
        lambda argv: completed(argv, stdout="ok\n"),
    )

    result = docker.run_in_sandbox(["echo", "hi"], workspace)

    assert result.ok is True
    assert result.exit_code == 0
    assert result.stdout == "ok\n"

    argv = calls[0]
    assert argv[:2] == ["docker", "run"]
    for flag in (
        "--rm",
        "--pull=never",
        "--network=none",
        "--cpus=1.0",
        "--memory=512m",
        "--memory-swap=512m",
        "--pids-limit=64",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
    ):
        assert flag in argv

    assert argv[argv.index("--tmpfs") + 1] == "/tmp:rw,noexec,nosuid,size=64m"
    assert argv[argv.index("-v") + 1] == f"{workspace}:/workspace"
    assert argv[argv.index("-w") + 1] == "/workspace"
    assert argv[-3:] == ["the-loop-sandbox:latest", "echo", "hi"]

    container_name = argv[argv.index("--name") + 1]
    assert re.fullmatch(r"the-loop-[0-9a-f]{12}", container_name)
    assert result.container_name == container_name


def test_run_in_sandbox_wraps_string_command_in_sh_c(workspace, monkeypatch):
    calls = capture_run(
        monkeypatch,
        lambda argv: completed(argv),
    )

    docker.run_in_sandbox("echo hi", workspace)

    assert calls[0][-4:] == [
        "the-loop-sandbox:latest",
        "sh",
        "-c",
        "echo hi",
    ]


# --- structured failures -----------------------------------------------------


def test_missing_docker_cli_is_sandbox_unavailable(workspace, monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(docker.subprocess, "run", fake_run)

    result = docker.run_in_sandbox(["echo", "hi"], workspace)

    assert result.ok is False
    assert result.error_type == "sandbox_unavailable"
    assert result.retryable is False
    assert "Docker CLI" in result.message


def test_unreachable_daemon_is_sandbox_unavailable(workspace, monkeypatch):
    stderr = (
        "docker: Cannot connect to the Docker daemon at "
        "unix:///var/run/docker.sock. Is the docker daemon running?"
    )
    capture_run(
        monkeypatch,
        lambda argv: completed(argv, returncode=125, stderr=stderr),
    )

    result = docker.run_in_sandbox(["echo", "hi"], workspace)

    assert result.ok is False
    assert result.error_type == "sandbox_unavailable"
    assert result.retryable is False
    assert "Docker is not available" in result.message


def test_missing_image_is_sandbox_unavailable(workspace, monkeypatch):
    stderr = (
        "Unable to find image 'the-loop-sandbox:latest' locally\n"
        "docker: Error response from daemon: No such image: "
        "the-loop-sandbox:latest"
    )
    capture_run(
        monkeypatch,
        lambda argv: completed(argv, returncode=125, stderr=stderr),
    )

    result = docker.run_in_sandbox(["echo", "hi"], workspace)

    assert result.ok is False
    assert result.error_type == "sandbox_unavailable"
    assert result.retryable is False
    assert "docker build -f Dockerfile.sandbox" in result.message


def test_command_failure_preserves_exit_code(workspace, monkeypatch):
    capture_run(
        monkeypatch,
        lambda argv: completed(
            argv,
            returncode=1,
            stdout="1 failed\n",
            stderr="",
        ),
    )

    result = docker.run_in_sandbox(["pytest"], workspace)

    assert result.ok is False
    assert result.error_type == "command_failed"
    assert result.retryable is True
    assert result.exit_code == 1
    assert result.stdout == "1 failed\n"


# --- timeout and cleanup -----------------------------------------------------


def test_timeout_forcibly_removes_the_container(workspace, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
        return completed(argv)

    monkeypatch.setattr(docker.subprocess, "run", fake_run)

    result = docker.run_in_sandbox(["sleep", "30"], workspace, timeout_s=2)

    assert result.ok is False
    assert result.error_type == "timeout"
    assert result.retryable is True
    assert result.timed_out is True
    assert result.duration_ms >= 0

    container_name = result.container_name
    assert container_name is not None
    assert calls[1] == ["docker", "rm", "-f", container_name]


def test_timeout_result_is_truncated(workspace, monkeypatch):
    def fake_run(argv, **kwargs):
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(
                argv,
                kwargs.get("timeout"),
                output="x" * 5000,
                stderr="",
            )
        return completed(argv)

    monkeypatch.setattr(docker.subprocess, "run", fake_run)

    result = docker.run_in_sandbox(
        ["sleep", "30"],
        workspace,
        timeout_s=2,
        max_output_chars=200,
    )

    assert result.error_type == "timeout"
    assert result.truncated is True
    assert "[TRUNCATED]" in result.stdout
    assert len(result.stdout) + len(result.stderr) <= 200


# --- output bounding ---------------------------------------------------------


def test_output_is_bounded_and_marked(workspace, monkeypatch):
    capture_run(
        monkeypatch,
        lambda argv: completed(argv, stdout="x" * 10000, stderr=""),
    )

    result = docker.run_in_sandbox(
        ["echo", "hi"],
        workspace,
        max_output_chars=100,
    )

    assert result.ok is True
    assert result.truncated is True
    assert "[TRUNCATED]" in result.stdout
    assert len(result.stdout) + len(result.stderr) <= 100


def test_stderr_is_kept_when_stdout_is_large(workspace, monkeypatch):
    capture_run(
        monkeypatch,
        lambda argv: completed(
            argv,
            returncode=1,
            stdout="x" * 10000,
            stderr="important error\n",
        ),
    )

    result = docker.run_in_sandbox(
        ["pytest"],
        workspace,
        max_output_chars=200,
    )

    assert result.error_type == "command_failed"
    assert result.truncated is True
    assert "important error" in result.stderr
    assert len(result.stdout) + len(result.stderr) <= 200


# --- validation --------------------------------------------------------------


def test_missing_workspace_is_rejected_before_docker(workspace, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Docker must not run for an invalid workspace")

    monkeypatch.setattr(docker.subprocess, "run", fail_if_called)

    result = docker.run_in_sandbox(["echo", "hi"], workspace / "missing")

    assert result.ok is False
    assert result.error_type == "invalid_workspace"
    assert result.retryable is False


@pytest.mark.parametrize("timeout_s", [0, -1, 301, "30", True, None])
def test_invalid_timeout_is_rejected(workspace, monkeypatch, timeout_s):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Docker must not run for an invalid timeout")

    monkeypatch.setattr(docker.subprocess, "run", fail_if_called)

    result = docker.run_in_sandbox(["echo", "hi"], workspace, timeout_s=timeout_s)

    assert result.ok is False
    assert result.error_type == "invalid_arguments"
    assert result.retryable is True


@pytest.mark.parametrize("command", [[], [1, 2], 123])
def test_invalid_command_is_rejected(workspace, command):
    result = docker.run_in_sandbox(command, workspace)

    assert result.ok is False
    assert result.error_type == "invalid_arguments"
    assert result.retryable is True
