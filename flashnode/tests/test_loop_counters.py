"""What counts as "this host is broken" — and what does not.

execute_one returns False for three different things and only one of them
implicates the machine. Treating them alike would quarantine a healthy host
that lost three lease races, which is the same trap the contributions ledger
hit: the coordinator answers HTTP 200 with {"accepted": false} when an output
fails its hash check or a commit loses a race.
"""

from __future__ import annotations

import pytest

import flashnode.executor.loop as loop_mod
from flashnode.executor.loop import ExecutorLoop
from flashnode.executor.runner import TaskExecutionError


class _Lease:
    lease_id = "L1"
    task_id = "T1"
    attempt_number = 1
    payload: dict = {}
    inputs: dict = {}


def _loop(**kw):
    return ExecutorLoop(client=object(), node_id="n1", **kw)


def _always_fails(monkeypatch, loop):
    monkeypatch.setattr(
        loop, "_execute_inner",
        lambda lease: (_ for _ in ()).throw(TaskExecutionError("boom")))


def test_a_fresh_loop_starts_with_clean_counters():
    loop = _loop()
    assert loop.tasks_failed == 0
    assert loop.consecutive_failures == 0
    assert loop.current_task is None
    assert loop.quarantined is False


def test_task_execution_error_increments_the_host_failure_counter(monkeypatch):
    loop = _loop()
    monkeypatch.setattr(
        loop, "_execute_inner",
        lambda lease: (_ for _ in ()).throw(TaskExecutionError("docker is unavailable")))
    assert loop.execute_one(_Lease()) is False
    assert loop.consecutive_failures == 1
    assert loop.tasks_failed == 1


def test_lease_lost_does_not_count_against_the_host(monkeypatch):
    """Someone else got the work. That says nothing about this machine."""
    loop = _loop()
    monkeypatch.setattr(
        loop, "_execute_inner",
        lambda lease: (_ for _ in ()).throw(loop_mod.LeaseLost()))
    assert loop.execute_one(_Lease()) is False
    assert loop.consecutive_failures == 0


def test_a_rejected_output_does_not_count_against_the_host(monkeypatch):
    """complete() -> accepted=False is HTTP 200. The task RAN here fine; the
    coordinator declined the result or lost a race."""
    loop = _loop()
    monkeypatch.setattr(loop, "_execute_inner", lambda lease: False)
    assert loop.execute_one(_Lease()) is False
    assert loop.consecutive_failures == 0
    assert loop.tasks_failed == 0


def test_one_success_resets_the_streak(monkeypatch):
    loop = _loop()
    _always_fails(monkeypatch, loop)
    loop.execute_one(_Lease())
    loop.execute_one(_Lease())
    assert loop.consecutive_failures == 2
    monkeypatch.setattr(loop, "_execute_inner", lambda lease: True)
    loop.execute_one(_Lease())
    assert loop.consecutive_failures == 0


def test_current_task_is_set_while_running_and_cleared_after(monkeypatch):
    loop = _loop()
    seen = {}

    def inner(lease):
        seen["task"] = loop.current_task
        seen["attempt"] = loop.current_attempt
        seen["started"] = loop.current_task_started
        return True

    monkeypatch.setattr(loop, "_execute_inner", inner)
    loop.execute_one(_Lease())
    assert seen["task"] == "T1"
    assert seen["attempt"] == 1
    assert seen["started"] is not None
    assert loop.current_task is None


def test_current_task_is_cleared_even_when_the_task_fails(monkeypatch):
    loop = _loop()
    _always_fails(monkeypatch, loop)
    loop.execute_one(_Lease())
    assert loop.current_task is None
    assert loop.current_task_started is None


def test_the_loop_module_does_not_import_the_doctor():
    """doctor.py imports flashnode.executor.hardening, which initialises the
    flashnode.executor package, whose __init__ imports loop. loop -> doctor
    -> executor -> loop resolves or explodes by import order — the worst kind
    of bug to ship to machines we cannot reach. Inject, never import."""
    import pathlib

    source = pathlib.Path(loop_mod.__file__).read_text()
    assert "flashnode.doctor" not in source
    assert "from flashnode import doctor" not in source


# ---------------------------------------------------------------- quarantine


class _Check:
    """health_check returns BLOCKING problems only, so a test needs just a
    name to print — the loop never reads a status."""

    name = "docker engine reachable"
    detail = "_ping 500"
    fix = "Start Docker Desktop."


class _Brake(BaseException):
    """Stops a loop that is SUPPOSED to keep claiming, so a test asserting
    'it did not stop' cannot hang the suite.

    BaseException, NOT Exception: `run()` wraps the claim in
    `except Exception` and treats anything it catches as "coordinator
    unreachable", then retries with a growing backoff — so an
    Exception-derived brake is swallowed and the test hangs forever instead
    of failing. Same reasoning as KeyboardInterrupt."""


class _FailingClient:
    """Hands out the same lease forever; the runner always fails."""

    def __init__(self, brake_after=5):
        self.claims = 0
        self.brake_after = brake_after

    def claim(self, node_id):
        self.claims += 1
        if self.claims > self.brake_after:
            raise _Brake
        return _Lease()

    def node_heartbeat(self, node_id):
        return True

    def fail(self, lease_id, msg):
        pass


def _quarantine_loop(health, threshold=3):
    return ExecutorLoop(client=_FailingClient(), node_id="n1",
                        health_check=health, max_consecutive_failures=threshold,
                        poll_seconds=0)


def test_an_unhealthy_host_stops_claiming(monkeypatch):
    loop = _quarantine_loop(lambda: [_Check()])   # one blocking problem
    _always_fails(monkeypatch, loop)
    loop.run()  # returns on its own — no brake needed, that is the point
    assert loop.quarantined is True
    # Three failures to reach the threshold, and then it STOPS. The bug this
    # exists to fix is a host that keeps claiming forever.
    assert loop.client.claims == 3
    assert loop.health_report and loop.health_report[0].name


def test_a_healthy_host_keeps_working_because_the_JOBS_are_failing(monkeypatch):
    """Same three failures, but the machine checks out. The jobs are broken,
    not the host — reset and carry on."""
    loop = _quarantine_loop(lambda: [])           # nothing blocking
    _always_fails(monkeypatch, loop)
    with pytest.raises(_Brake):
        loop.run()
    assert loop.quarantined is False
    assert loop.client.claims == 6  # kept going well past the threshold


def test_no_health_check_means_no_quarantine(monkeypatch):
    """Every existing caller passes nothing and must behave exactly as before."""
    loop = ExecutorLoop(client=_FailingClient(), node_id="n1", poll_seconds=0)
    _always_fails(monkeypatch, loop)
    with pytest.raises(_Brake):
        loop.run()
    assert loop.quarantined is False


def test_threshold_zero_disables_the_quarantine(monkeypatch):
    loop = _quarantine_loop(lambda: [_Check()], threshold=0)
    _always_fails(monkeypatch, loop)
    with pytest.raises(_Brake):
        loop.run()
    assert loop.quarantined is False


def test_the_health_check_is_not_called_before_the_threshold(monkeypatch):
    """Checking on every failure would shell out to docker three times per
    unlucky job."""
    calls = []
    loop = _quarantine_loop(lambda: calls.append(1) or [])
    _always_fails(monkeypatch, loop)
    with pytest.raises(_Brake):
        loop.run()
    # Claims 1-3 fail -> check at 3, healthy, counter resets to 0.
    # Claims 4-5 fail -> streak is only 2. Claim 6 hits the brake.
    # So exactly one check, not one per failure.
    assert len(calls) == 1
