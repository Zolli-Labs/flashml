"""`flashnode work` must not claim a lease on a host that cannot run tasks.

Before this gate, docker_runner raised TaskExecutionError, loop.py called
fail() on the lease and kept claiming — so a misconfigured machine burned
task after task while looking healthy to its owner and to the coordinator.
"""

from __future__ import annotations

import pytest

from flashnode.agent.cli import _work
from flashnode.doctor import CheckResult


@pytest.fixture()
def never_construct_a_loop(monkeypatch):
    """Fail loudly if the gate lets execution through."""
    import flashnode.executor as executor

    def boom(*a, **k):
        raise AssertionError("ExecutorLoop was constructed on an unhealthy host")

    monkeypatch.setattr(executor, "ExecutorLoop", boom)
    return boom


def test_work_refuses_to_start_when_a_check_fails(monkeypatch, capsys, never_construct_a_loop):
    monkeypatch.setattr(
        "flashnode.doctor.run_checks",
        lambda **kw: [CheckResult("docker engine reachable", "fail",
                                  detail="_ping 500", fix="Start Docker Desktop.")],
    )
    rc = _work(["--runner", "docker", "--coordinator", "http://localhost:8100"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "flashnode doctor" in err
    assert "Start Docker Desktop." in err


def test_work_gate_does_not_pull(monkeypatch, never_construct_a_loop):
    seen = {}

    def fake_run_checks(**kw):
        seen.update(kw)
        return [CheckResult("docker engine reachable", "fail")]

    monkeypatch.setattr("flashnode.doctor.run_checks", fake_run_checks)
    _work(["--runner", "docker"])
    assert seen["pull"] is False


class _Reached(Exception):
    """Raised where the subprocess tier should get to, so the test asserts on
    a specific point in _work rather than on any exception at all."""


def test_work_runs_no_doctor_for_the_subprocess_tier(monkeypatch):
    """The subprocess tier has no engine, no registry and no mounts, so the
    gate must not run for it."""
    called = []
    monkeypatch.setattr("flashnode.doctor.run_checks",
                        lambda **kw: called.append(kw) or [])

    def stop_here(*a, **k):
        raise _Reached

    monkeypatch.setattr("flashnode.identity.store.load_or_create_node_id", stop_here)
    with pytest.raises(_Reached):
        _work(["--runner", "subprocess", "--coordinator", "http://127.0.0.1:1"])
    assert called == []
