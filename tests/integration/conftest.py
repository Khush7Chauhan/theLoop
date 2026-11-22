import pytest

from the_loop.policy import paths
from the_loop.policy.approvals import approve_current_call
from the_loop.sandbox.docker import (
    DEFAULT_IMAGE,
    docker_available,
    image_available,
)


@pytest.fixture(scope="session")
def docker_ready():
    """Skip the integration suite cleanly when Docker is not usable."""
    if not docker_available():
        pytest.skip("Docker daemon is not available")

    if not image_available(DEFAULT_IMAGE):
        pytest.skip(
            f"Docker image {DEFAULT_IMAGE} is not built. Run: "
            f"docker build -f Dockerfile.sandbox -t {DEFAULT_IMAGE} ."
        )
    return True


@pytest.fixture
def sandbox_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def loop_workspace(sandbox_workspace):
    """Point the fs tools and sandbox tools at a fresh workspace."""
    previous = paths.get_workspace()
    paths.set_workspace(sandbox_workspace)
    try:
        yield sandbox_workspace
    finally:
        paths.set_workspace(previous)


@pytest.fixture
def approved_shell():
    """Grant the one-shot approval the interactive loop normally provides."""
    with approve_current_call():
        yield
