import json
from types import SimpleNamespace

import pytest

import loop as loop_module
from the_loop.policy.rules import check_policy
from the_loop.tools.fs import edit_file


def make_tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_test",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.fixture
def loop_workspace(workspace, monkeypatch):
    monkeypatch.setattr(loop_module, "WORKSPACE", workspace)
    return workspace


# --- successful edits --------------------------------------------------------


def test_edit_file_replaces_a_single_occurrence(workspace):
    target = workspace / "notes.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = edit_file("notes.txt", "beta", "delta")

    assert result.ok is True
    assert result.error_type is None
    assert target.read_text(encoding="utf-8") == "alpha\ndelta\ngamma\n"


def test_edit_file_replaces_only_the_matched_occurrence(workspace):
    target = workspace / "notes.txt"
    target.write_text("one two one\n", encoding="utf-8")

    result = edit_file("notes.txt", "two one", "two three")

    assert result.ok is True
    assert target.read_text(encoding="utf-8") == "one two three\n"


# --- retryable failures ------------------------------------------------------


def test_edit_file_missing_file_is_retryable_not_found(workspace):
    result = edit_file("missing.txt", "old", "new")

    assert result.ok is False
    assert result.error_type == "not_found"
    assert result.retryable is True
    assert "does not exist" in result.message
    assert not (workspace / "missing.txt").exists()


def test_edit_file_absent_old_text_is_retryable_not_found(workspace):
    target = workspace / "notes.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = edit_file("notes.txt", "not present", "new")

    assert result.ok is False
    assert result.error_type == "not_found"
    assert result.retryable is True
    assert "not found" in result.message
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_edit_file_ambiguous_old_text_is_retryable_not_unique(workspace):
    target = workspace / "notes.txt"
    target.write_text("repeat\nkeep\nrepeat\n", encoding="utf-8")

    result = edit_file("notes.txt", "repeat", "changed")

    assert result.ok is False
    assert result.error_type == "not_unique"
    assert result.retryable is True
    assert "2 times" in result.message
    assert "context" in result.message
    assert target.read_text(encoding="utf-8") == "repeat\nkeep\nrepeat\n"


def test_edit_file_rejects_empty_old_text(workspace):
    target = workspace / "notes.txt"
    target.write_text("alpha\n", encoding="utf-8")

    result = edit_file("notes.txt", "", "new")

    assert result.ok is False
    assert result.error_type == "invalid_arguments"
    assert result.retryable is True
    assert target.read_text(encoding="utf-8") == "alpha\n"


# --- policy denials ----------------------------------------------------------


@pytest.mark.parametrize("path", [".env", "secrets/token.txt", ".loop/state.json"])
def test_edit_file_policy_denies_sensitive_paths(path):
    policy = check_policy(
        "edit_file", {"path": path, "old_text": "a", "new_text": "b"}
    )

    assert policy["decision"] == "deny"
    assert policy["retryable"] is False


@pytest.mark.parametrize("path", [".env", ".loop/state.json"])
def test_handle_tool_call_blocks_edit_file_policy_denials(loop_workspace, path):
    target = loop_workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret=1\n", encoding="utf-8")

    result = loop_module.handle_tool_call(
        make_tool_call(
            "edit_file",
            {"path": path, "old_text": "secret=1", "new_text": "secret=2"},
        )
    )

    assert result["ok"] is False
    assert result["error_type"] == "permission"
    assert result["retryable"] is False
    assert target.read_text(encoding="utf-8") == "secret=1\n"


def test_edit_file_source_change_requires_approval(workspace):
    policy = check_policy(
        "edit_file",
        {"path": "src/main.py", "old_text": "a", "new_text": "b"},
    )

    assert policy["decision"] == "ask"
    assert "session" in policy["approval_scopes"]
