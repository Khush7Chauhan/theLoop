import pytest

from the_loop.errors import ToolError
from the_loop.policy.paths import safe_path


def test_symlink_directory_escape_is_denied(workspace, tmp_path, make_link):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")

    make_link(workspace / "link", outside)

    with pytest.raises(ToolError) as excinfo:
        safe_path("link/secret.txt")

    assert excinfo.value.error_type == "permission"
    assert excinfo.value.retryable is False


def test_write_through_symlink_escape_is_denied(workspace, tmp_path, make_link):
    outside = tmp_path / "outside"
    outside.mkdir()

    make_link(workspace / "link", outside)

    with pytest.raises(ToolError):
        safe_path("link/new-file.txt")
    assert not (outside / "new-file.txt").exists()


def test_symlinked_file_escape_is_denied(workspace, tmp_path, make_link):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    make_link(workspace / "escape.txt", outside_file)

    with pytest.raises(ToolError) as excinfo:
        safe_path("escape.txt")

    assert excinfo.value.error_type == "permission"


def test_symlink_inside_workspace_is_allowed(workspace, make_link):
    real = workspace / "real"
    real.mkdir()
    (real / "notes.txt").write_text("hi", encoding="utf-8")

    make_link(workspace / "alias", real)

    assert safe_path("alias/notes.txt") == real / "notes.txt"
