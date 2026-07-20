"""Integration-test harness: auto-marking and environment detection.

Every test in this folder gets the `integration` marker (excluded from the
default `pytest` run by pyproject's addopts). Fixtures detect the required
infrastructure and skip — with a reason — when it is absent, so
`pytest -m integration` is always safe: it runs what the machine can support
and reports what it can't.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_collection_modifyitems(items):
    # This hook receives ALL collected items (conftest hooks are not
    # directory-scoped) — mark only the tests that live in this folder.
    for item in items:
        if str(item.fspath).startswith(_HERE):
            item.add_marker(pytest.mark.integration)


def _run_ok(cmd: list[str], timeout: float = 10) -> bool:
    try:
        return (
            subprocess.run(cmd, capture_output=True, timeout=timeout, check=False).returncode == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def docker_available() -> None:
    if shutil.which("docker") is None or not _run_ok(["docker", "info"]):
        pytest.skip("no reachable Docker daemon (install Docker Desktop or colima)")


@pytest.fixture(scope="session")
def kubernetes_available() -> None:
    if shutil.which("kubectl") is None or not _run_ok(["kubectl", "version", "--request-timeout=5s"]):
        pytest.skip("no reachable Kubernetes cluster (workspace: make poc-local-up)")


@pytest.fixture(scope="session")
def kuberay_available(kubernetes_available) -> None:
    if not _run_ok(["kubectl", "get", "crd", "rayjobs.ray.io"]):
        pytest.skip("KubeRay operator not installed (workspace: make poc-local-up)")


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str


@pytest.fixture(scope="session")
def minio_config() -> MinioConfig:
    endpoint = os.environ.get("FLASHML_TEST_MINIO_ENDPOINT")
    if not endpoint:
        pytest.skip(
            "FLASHML_TEST_MINIO_ENDPOINT not set "
            "(workspace: make poc-local-up && make poc-local-forward, then "
            "export FLASHML_TEST_MINIO_ENDPOINT=localhost:9000)"
        )
    return MinioConfig(
        endpoint=endpoint,
        access_key=os.environ.get("FLASHML_TEST_MINIO_ACCESS_KEY", "flashml"),
        secret_key=os.environ.get("FLASHML_TEST_MINIO_SECRET_KEY", "flashml-dev"),
        bucket=os.environ.get("FLASHML_TEST_MINIO_BUCKET", "flashml-artifacts"),
    )
