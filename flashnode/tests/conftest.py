"""Fixtures shared across the whole test tree.

Kept at the tests/ root (not per-directory) so both the unit-level real-
Docker smoke test (test_docker_runner.py) and tests/integration/ share one
definition instead of two copies drifting apart.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture()
def docker_workdir():
    """A workdir the Docker VM can actually see.

    Do NOT use pytest's `tmp_path` for anything that gets bind-mounted into
    a container here. On macOS, colima/Docker Desktop only share $HOME by
    default — pytest's /var/folders tmp dir bind-mounts as an EMPTY
    directory inside the VM (or the mount fails outright), so a task that
    reads/writes under /work either sees nothing or the `docker run` call
    itself errors before the workload ever runs. Either way, an assertion
    like `pytest.raises(TaskExecutionError)` still passes — for the wrong
    reason, proving nothing about the sandbox flag under test. Same
    constraint applies to real devices: ExecutorLoop's workdir_base /
    FLASHNODE_WORKDIR exists for exactly this reason (see AGENTS.md).
    """
    base = Path.home() / ".cache" / "flashnode-tests" / uuid.uuid4().hex[:8]
    base.mkdir(parents=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)
