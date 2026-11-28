import pytest

from the_loop.policy.approvals import (
    HeadlessApprover,
    InteractiveApprover,
    reset_session_approvals,
)
from the_loop.policy.rules import ask


@pytest.fixture(autouse=True)
def clear_session_approvals():
    reset_session_approvals()
    yield
    reset_session_approvals()


def make_policy(rule_hint="run_shell"):
    return ask(
        "The agent wants to run a shell command.",
        risk="runs a command",
        rule_hint=rule_hint,
    )


def test_interactive_approver_grants_once_and_remembers_session():
    answers = iter(["y", "s"])
    output = []
    approver = InteractiveApprover(
        input_fn=lambda _prompt: next(answers),
        output_fn=output.append,
    )
    policy = make_policy()

    first = approver.approve(policy)
    second = approver.approve(policy)

    assert first.approved is True
    assert second.approved is True
    assert any("Approval needed" in line for line in output)


def test_interactive_approver_declines_and_reports_reason():
    approver = InteractiveApprover(
        input_fn=lambda _prompt: "n",
        output_fn=lambda _line: None,
    )

    decision = approver.approve(make_policy())

    assert decision.approved is False
    assert decision.reason == "interactive_declined"


def test_headless_allows_configured_tool_without_prompting(monkeypatch, workspace):
    def fail_if_asked(*args, **kwargs):
        raise AssertionError("headless mode must never read stdin")

    monkeypatch.setattr("builtins.input", fail_if_asked)
    approver = HeadlessApprover(
        {"allow": {"tools": ["run_shell"], "paths": []}},
        workspace=workspace,
    )

    decision = approver.approve(
        make_policy(),
        tool="run_shell",
        arguments={"command": "echo hi"},
    )

    assert decision.approved is True
    assert decision.reason == "headless_allowed_tool"


def test_headless_allows_matching_path_glob(workspace):
    approver = HeadlessApprover(
        {"allow": {"tools": [], "paths": ["src/*"]}},
        workspace=workspace,
    )

    decision = approver.approve(
        make_policy("write_file:src/main.py"),
        tool="edit_file",
        arguments={"path": "src/main.py", "old_text": "a", "new_text": "b"},
    )

    assert decision.approved is True
    assert decision.reason == "headless_allowed_path"


def test_headless_approver_can_load_a_policy_path(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        '{"allow": {"tools": ["run_shell"], "paths": []}}',
        encoding="utf-8",
    )

    approver = HeadlessApprover(policy_path, workspace=tmp_path)

    decision = approver.approve(
        make_policy(),
        tool="run_shell",
        arguments={"command": "echo hi"},
    )

    assert decision.approved is True
    assert decision.reason == "headless_allowed_tool"


def test_headless_denies_unresolved_ask(workspace):
    approver = HeadlessApprover({"allow": {"tools": [], "paths": []}}, workspace=workspace)

    decision = approver.approve(
        make_policy("write_file:src/main.py"),
        tool="edit_file",
        arguments={"path": "src/main.py", "old_text": "a", "new_text": "b"},
    )

    assert decision.approved is False
    assert decision.reason == "headless_unresolved_ask"


@pytest.mark.parametrize(
    "path, reason",
    [
        (".env", "headless_hard_deny"),
        ("secrets/token.txt", "headless_hard_deny"),
        (".loop/state.json", "headless_hard_deny"),
        ("docs/.env", "headless_hard_deny"),
        ("../outside.txt", "headless_path_escape"),
    ],
)
def test_headless_hard_denials_cannot_be_overridden(workspace, path, reason):
    approver = HeadlessApprover(
        {"allow": {"tools": ["write_file"], "paths": ["*", "**"]}},
        workspace=workspace,
    )

    decision = approver.approve(
        make_policy(f"write_file:{path}"),
        tool="write_file",
        arguments={"path": path, "content": "x"},
    )

    assert decision.approved is False
    assert decision.reason == reason


def test_headless_denies_shell_commands_touching_secrets(workspace):
    approver = HeadlessApprover(
        {"allow": {"tools": ["run_shell"], "paths": []}},
        workspace=workspace,
    )

    decision = approver.approve(
        make_policy(),
        tool="run_shell",
        arguments={"command": "cat .env"},
    )

    assert decision.approved is False
    assert decision.reason == "headless_hard_deny"


def test_headless_deny_overrides_allow(workspace):
    approver = HeadlessApprover(
        {
            "allow": {"tools": ["run_shell"], "paths": ["src/*"]},
            "deny": {"tools": ["run_shell"], "paths": ["src/*"]},
        },
        workspace=workspace,
    )

    tool_decision = approver.approve(
        make_policy(),
        tool="run_shell",
        arguments={"command": "echo hi"},
    )
    path_decision = approver.approve(
        make_policy("write_file:src/main.py"),
        tool="edit_file",
        arguments={"path": "src/main.py", "old_text": "a", "new_text": "b"},
    )

    assert tool_decision.approved is False
    assert tool_decision.reason == "headless_denied_tool"
    assert path_decision.approved is False
    assert path_decision.reason == "headless_denied_path"
