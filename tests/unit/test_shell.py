import json
from types import SimpleNamespace

import pytest

import loop as loop_module
import the_loop.tools.shell as shell
from the_loop.policy.approvals import approve_current_call, is_call_approved
from the_loop.policy.rules import check_policy
from the_loop.sandbox.docker import SandboxResult


def make_tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_test",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def fake_sandbox(
    *,
    ok=True,
    exit_code=0,
    stdout="",
    stderr="",
    error_type=None,
    message="",
    retryable=False,
):
    def _fake(command, workspace, *, timeout_s=30, **kwargs):
        return SandboxResult(
            ok=ok,
            command=command,
            container_name="the-loop-fake",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            error_type=error_type,
            message=message,
            retryable=retryable,
            duration_ms=5,
        )

    return _fake


# --- policy ------------------------------------------------------------------


def test_run_shell_policy_requires_approval():
    policy = check_policy("run_shell", {"command": "ls"})

    assert policy["decision"] == "ask"
    assert policy["rule_hint"] == "run_shell"
    assert "session" in policy["approval_scopes"]


def test_run_shell_policy_denies_empty_command():
    assert check_policy("run_shell", {"command": "   "})["decision"] == "deny"


def test_run_tests_policy_is_allowed():
    assert check_policy("run_tests", {"args": "-q"}) == {"decision": "allow"}


# --- run_shell ---------------------------------------------------------------


def test_run_shell_requires_approval_before_docker(workspace, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Docker must not run without approval")

    monkeypatch.setattr(shell, "run_in_sandbox", fail_if_called)

    result = shell.run_shell("echo hi")

    assert result.ok is False
    assert result.error_type == "approval_required"
    assert result.retryable is False


def test_run_shell_with_approval_uses_sh_c(workspace, monkeypatch):
    captured = {}

    def fake(command, sandbox_workspace, *, timeout_s=30, **kwargs):
        captured["command"] = command
        captured["workspace"] = sandbox_workspace
        captured["timeout_s"] = timeout_s
        return fake_sandbox(stdout="hi\n")(command, sandbox_workspace)

    monkeypatch.setattr(shell, "run_in_sandbox", fake)

    with approve_current_call():
        result = shell.run_shell("echo hi", timeout_s=7)

    assert result.ok is True
    assert result.exit_code == 0
    assert captured["command"] == ["sh", "-c", "echo hi"]
    assert captured["workspace"] == workspace
    assert captured["timeout_s"] == 7
    assert result.data["stdout"] == "hi\n"
    assert result.data["container"] == "the-loop-fake"


def test_run_shell_timeout_is_retryable(workspace, monkeypatch):
    monkeypatch.setattr(
        shell,
        "run_in_sandbox",
        fake_sandbox(
            ok=False,
            exit_code=None,
            error_type="timeout",
            message="timed out",
            retryable=True,
        ),
    )

    with approve_current_call():
        result = shell.run_shell("sleep 30", timeout_s=2)

    assert result.ok is False
    assert result.error_type == "timeout"
    assert result.retryable is True


# --- run_tests ---------------------------------------------------------------


def test_run_tests_uses_argv_without_a_shell(workspace, monkeypatch):
    captured = {}

    def fake(command, sandbox_workspace, *, timeout_s=30, **kwargs):
        captured["command"] = command
        captured["timeout_s"] = timeout_s
        return fake_sandbox(stdout="1 passed\n")(command, sandbox_workspace)

    monkeypatch.setattr(shell, "run_in_sandbox", fake)

    result = shell.run_tests("-q tests/ -k 'foo or bar'", timeout_s=45)

    assert result.ok is True
    assert captured["command"] == [
        "pytest",
        "-q",
        "tests/",
        "-k",
        "foo or bar",
    ]
    assert captured["timeout_s"] == 45


@pytest.mark.parametrize(
    "args",
    [
        "-q; rm -rf /",
        "-q | tee out.txt",
        "$(id)",
        "`id`",
        "tests/*.py",
        "a > b",
        "line\nbreak",
    ],
)
def test_run_tests_rejects_shell_metacharacters(workspace, monkeypatch, args):
    def fail_if_called(*call_args, **kwargs):
        raise AssertionError("Docker must not run for invalid args")

    monkeypatch.setattr(shell, "run_in_sandbox", fail_if_called)

    result = shell.run_tests(args)

    assert result.ok is False
    assert result.error_type == "invalid_arguments"
    assert result.retryable is True


def test_run_tests_rejects_unmatched_quote(workspace, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Docker must not run for invalid args")

    monkeypatch.setattr(shell, "run_in_sandbox", fail_if_called)

    result = shell.run_tests("'unterminated")

    assert result.ok is False
    assert result.error_type == "invalid_arguments"


@pytest.mark.parametrize("timeout_s", [0, -1, 301, "30", True, None])
def test_run_tests_rejects_invalid_timeout(workspace, timeout_s):
    result = shell.run_tests(timeout_s=timeout_s)

    assert result.ok is False
    assert result.error_type == "invalid_arguments"
    assert result.retryable is True


def test_run_tests_failure_preserves_exit_code_and_output(workspace, monkeypatch):
    monkeypatch.setattr(
        shell,
        "run_in_sandbox",
        fake_sandbox(
            ok=False,
            exit_code=1,
            stdout="1 failed\n",
            error_type="command_failed",
            message="Command exited with status 1.",
            retryable=True,
        ),
    )

    result = shell.run_tests()

    assert result.ok is False
    assert result.exit_code == 1
    assert result.error_type == "command_failed"
    assert result.retryable is True
    assert result.data["stdout"] == "1 failed\n"


# --- loop approval wiring ----------------------------------------------------


def test_handle_tool_call_grants_one_shot_shell_approval(workspace, monkeypatch):
    monkeypatch.setattr(loop_module, "WORKSPACE", workspace)
    monkeypatch.setattr(loop_module, "prompt_for_approval", lambda policy: True)

    captured = {}

    def fake(command, sandbox_workspace, *, timeout_s=30, **kwargs):
        captured["command"] = command
        return fake_sandbox(stdout="ok\n")(command, sandbox_workspace)

    monkeypatch.setattr(shell, "run_in_sandbox", fake)

    result = loop_module.handle_tool_call(
        make_tool_call("run_shell", {"command": "echo hi"})
    )

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["result"]["stdout"] == "ok\n"
    assert captured["command"] == ["sh", "-c", "echo hi"]
    # The approval is scoped to the call, not left behind afterwards.
    assert is_call_approved() is False


def test_handle_tool_call_blocks_shell_when_approval_declined(workspace, monkeypatch):
    monkeypatch.setattr(loop_module, "WORKSPACE", workspace)
    monkeypatch.setattr(loop_module, "prompt_for_approval", lambda policy: False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Docker must not run when approval is declined")

    monkeypatch.setattr(shell, "run_in_sandbox", fail_if_called)

    result = loop_module.handle_tool_call(
        make_tool_call("run_shell", {"command": "echo hi"})
    )

    assert result["ok"] is False
    assert result["error_type"] == "permission"
    assert result["needs_approval"] is True


def test_handle_tool_call_runs_tests_without_approval(workspace, monkeypatch):
    monkeypatch.setattr(loop_module, "WORKSPACE", workspace)

    def fail_if_asked(policy):
        raise AssertionError("run_tests should not require approval")

    monkeypatch.setattr(loop_module, "prompt_for_approval", fail_if_asked)

    def fake(command, sandbox_workspace, *, timeout_s=60, **kwargs):
        return fake_sandbox(stdout="2 passed\n")(command, sandbox_workspace)

    monkeypatch.setattr(shell, "run_in_sandbox", fake)

    result = loop_module.handle_tool_call(
        make_tool_call("run_tests", {"args": "-q tests/"})
    )

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["result"]["stdout"] == "2 passed\n"


def test_handle_tool_call_rejects_unknown_arguments(workspace, monkeypatch):
    monkeypatch.setattr(loop_module, "WORKSPACE", workspace)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Docker must not run for unknown arguments")

    monkeypatch.setattr(shell, "run_in_sandbox", fail_if_called)

    result = loop_module.handle_tool_call(
        make_tool_call(
            "run_tests",
            {"args": "-q", "approved": True, "max_output_chars": 1},
        )
    )

    assert result["ok"] is False
    assert result["error_type"] == "unknown_arguments"
    assert result["retryable"] is True
