import os
import subprocess
from pathlib import Path

import pytest

from the_loop.policy import paths


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the package sandbox at a fresh workspace directory."""
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(paths, "_workspace", root.resolve())
    return root.resolve()


@pytest.fixture
def make_link():
    """Create a symlink, falling back to a Windows junction when needed."""

    def _make(link: Path, target: Path) -> Path:
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
            return link
        except (OSError, NotImplementedError):
            pass

        if os.name == "nt":
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                return link

        pytest.skip("symlinks are not available in this environment")

    return _make
