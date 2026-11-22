import shlex

import pytest

from the_loop.sandbox.docker import container_exists, run_in_sandbox
from the_loop.tools.fs import edit_file
from the_loop.tools.shell import run_shell, run_tests

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("docker_ready"),
]


def _combined_output(result) -> str:
    data = result.data or {}
    return (data.get("stdout", "") + data.get("stderr", "")).lower()


# 1. Edit + pytest flow -------------------------------------------------------


def test_edit_file_then_pytest_passes(loop_workspace):
    (loop_workspace / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    (loop_workspace / "test_calc.py").write_text(
        "from calc import add\n\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    edit_result = edit_file("calc.py", "return a - b", "return a + b")
    assert edit_result.ok is True, edit_result.message

    result = run_tests("test_calc.py -q", timeout_s=60)

    assert result.ok is True, result.message + "\n" + _combined_output(result)
    assert result.exit_code == 0
    assert "1 passed" in result.data["stdout"]


# 2. No-network enforcement ---------------------------------------------------


def test_sandbox_has_no_network(loop_workspace, approved_shell):
    code = "import socket; socket.create_connection(('1.1.1.1', 80), timeout=2)"
    result = run_shell(f"python -c {shlex.quote(code)}", timeout_s=15)

    output = _combined_output(result)
    assert result.ok is False, "outbound network access unexpectedly succeeded"
    assert result.exit_code != 0
    assert (
        "network" in output
        or "unreachable" in output
        or "timed out" in output
    ), output


# 3. Hard timeout + cleanup ---------------------------------------------------


def test_timeout_removes_the_container(loop_workspace, approved_shell):
    result = run_shell("sleep 30", timeout_s=2)

    assert result.ok is False
    assert result.error_type == "timeout"
    assert result.retryable is True
    assert result.duration_ms >= 1900

    container_name = result.data["container"]
    assert container_name
    assert container_name.startswith("the-loop-")
    assert container_exists(container_name) is False


# 4. Non-root check -----------------------------------------------------------


def test_sandbox_runs_as_non_root(loop_workspace, approved_shell):
    result = run_shell("id -u", timeout_s=15)

    assert result.ok is True, result.message + "\n" + _combined_output(result)
    assert result.exit_code == 0
    uid = result.data["stdout"].strip()
    assert uid != "0"
    assert uid == "1000"


# 5. Output truncation --------------------------------------------------------


def test_output_is_truncated(loop_workspace):
    result = run_in_sandbox(
        ["python", "-c", "print('x' * 5000)"],
        loop_workspace,
        timeout_s=15,
        max_output_chars=500,
    )

    assert result.ok is True, result.message + "\n" + result.stderr
    assert result.exit_code == 0
    assert result.truncated is True
    assert "[TRUNCATED]" in result.stdout
    assert len(result.stdout) + len(result.stderr) <= 500
