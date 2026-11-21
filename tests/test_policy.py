import pytest

import loop
from loop import ToolError, check_policy, safe_path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(loop, "WORKSPACE", root.resolve())
    return root.resolve()


# --- safe_path ---------------------------------------------------------------


def test_safe_path_accepts_relative_path_inside_workspace(workspace):
    assert safe_path("notes.txt") == workspace / "notes.txt"
    assert safe_path("src/main.py") == workspace / "src" / "main.py"
    assert safe_path(".") == workspace


def test_safe_path_rejects_parent_traversal(workspace):
    with pytest.raises(ToolError) as excinfo:
        safe_path("../outside.txt")

    assert excinfo.value.error_type == "permission"
    assert excinfo.value.retryable is False


def test_safe_path_rejects_nested_parent_traversal(workspace):
    with pytest.raises(ToolError):
        safe_path("src/../../../outside.txt")


def test_safe_path_rejects_absolute_path_outside_workspace(workspace, tmp_path):
    with pytest.raises(ToolError):
        safe_path(str(tmp_path / "outside.txt"))


# --- policy: sensitive reads -------------------------------------------------


def test_read_env_is_denied():
    policy = check_policy("read_file", {"path": ".env"})

    assert policy["decision"] == "deny"
    assert policy["retryable"] is False


def test_read_secrets_is_denied():
    policy = check_policy("read_file", {"path": "secrets/api_key.txt"})

    assert policy["decision"] == "deny"


# --- policy: sensitive writes ------------------------------------------------


def test_write_env_is_denied():
    assert check_policy("write_file", {"path": ".env"})["decision"] == "deny"


def test_write_secrets_is_denied():
    policy = check_policy(
        "write_file", {"path": "secrets/token.txt", "content": "x"}
    )["decision"]

    assert policy == "deny"


def test_write_internal_loop_state_is_denied():
    policy = check_policy(
        "write_file", {"path": ".loop/state.json", "content": "{}"}
    )

    assert policy["decision"] == "deny"


# --- policy: source writes need approval, everything else is allowed --------


def test_write_source_asks_for_approval():
    policy = check_policy(
        "write_file", {"path": "src/main.py", "content": "print('hi')"}
    )

    assert policy["decision"] == "ask"
    assert policy["rule_hint"] == "write_file:src/main.py"
    assert "session" in policy["approval_scopes"]


def test_write_test_file_asks_for_approval():
    policy = check_policy(
        "write_file", {"path": "tests/test_app.py", "content": "pass"}
    )

    assert policy["decision"] == "ask"


def test_write_docs_is_allowed():
    assert check_policy("write_file", {"path": "README.md", "content": "hi"}) == {
        "decision": "allow"
    }


def test_read_regular_file_is_allowed():
    assert check_policy("read_file", {"path": "notes.txt"}) == {"decision": "allow"}


def test_list_dir_is_allowed():
    assert check_policy("list_dir", {"path": "."}) == {"decision": "allow"}
