"""cli.py owns every policy decision the view and the quarantine need:
TTY detection, the threshold, the health-check closure, the exit code.
"""

from __future__ import annotations

import pytest

from flashnode.agent.cli import _work
from flashnode.doctor import CheckResult


@pytest.fixture()
def healthy(monkeypatch, tmp_path):
    monkeypatch.setattr("flashnode.doctor.run_checks", lambda **kw: [])
    monkeypatch.setenv("FLASHNODE_STATE_DIR", str(tmp_path))


def _spy_loop(monkeypatch, seen, **attrs):
    import flashnode.executor as executor

    class Spy:
        quarantined = False
        health_report = None

        def __init__(self, *a, **kw):
            seen.update(kw)
            for k, v in attrs.items():
                setattr(self, k, v)

        def run(self, max_tasks=None):
            return 0

    monkeypatch.setattr(executor, "ExecutorLoop", Spy)
    monkeypatch.setattr(executor.CoordinatorClient, "register", lambda self, r: None)
    return Spy


def test_the_loop_is_given_an_injected_health_check(monkeypatch, healthy):
    """loop.py must never import the doctor; cli wires it (spec §2.1.1)."""
    seen: dict = {}
    _spy_loop(monkeypatch, seen)
    _work(["--runner", "subprocess", "--coordinator", "http://localhost:1"])
    assert callable(seen["health_check"])
    assert seen["max_consecutive_failures"] == 3


def test_the_health_check_returns_only_blocking_problems(monkeypatch, healthy):
    """The GPU check reports "info" and never fails. If cli handed the raw
    results to the loop, every CPU-only volunteer would be quarantined on
    their third unlucky job."""
    seen: dict = {}
    _spy_loop(monkeypatch, seen)
    monkeypatch.setattr("flashnode.doctor.run_checks", lambda **kw: [
        CheckResult("docker engine reachable", "ok"),
        CheckResult("GPU devices", "info", detail="no GPU detected"),
    ])
    _work(["--runner", "subprocess", "--coordinator", "http://localhost:1"])
    assert seen["health_check"]() == []


def test_a_blocking_problem_survives_the_filter(monkeypatch, healthy):
    seen: dict = {}
    _spy_loop(monkeypatch, seen)
    monkeypatch.setattr("flashnode.doctor.run_checks", lambda **kw: [
        CheckResult("GPU devices", "info"),
        CheckResult("docker engine reachable", "fail", fix="Start Docker."),
    ])
    _work(["--runner", "subprocess", "--coordinator", "http://localhost:1"])
    problems = seen["health_check"]()
    assert [p.name for p in problems] == ["docker engine reachable"]


def test_max_consecutive_failures_is_configurable(monkeypatch, healthy):
    seen: dict = {}
    _spy_loop(monkeypatch, seen)
    _work(["--runner", "subprocess", "--coordinator", "http://localhost:1",
           "--max-consecutive-failures", "7"])
    assert seen["max_consecutive_failures"] == 7


def test_a_quarantined_run_exits_2_and_prints_the_failing_checks(
    monkeypatch, healthy, capsys
):
    seen: dict = {}
    _spy_loop(monkeypatch, seen, quarantined=True, health_report=[
        CheckResult("docker engine reachable", "fail", detail="_ping 500",
                    fix="Start Docker Desktop."),
    ])
    rc = _work(["--runner", "subprocess", "--coordinator", "http://localhost:1"])
    assert rc == 2
    assert "Start Docker Desktop." in capsys.readouterr().err


def test_no_status_view_when_stdout_is_not_a_tty(monkeypatch, healthy):
    """Redrawing with ANSI into a pipe or a systemd journal is corruption."""
    seen: dict = {}
    _spy_loop(monkeypatch, seen)
    started = []
    monkeypatch.setattr("flashnode.status.StatusView.start",
                        lambda self: started.append(True))
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    _work(["--runner", "subprocess", "--coordinator", "http://localhost:1"])
    assert started == []


def test_log_json_suppresses_the_view_even_on_a_tty(monkeypatch, healthy):
    seen: dict = {}
    _spy_loop(monkeypatch, seen)
    started = []
    monkeypatch.setattr("flashnode.status.StatusView.start",
                        lambda self: started.append(True))
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    _work(["--runner", "subprocess", "--coordinator", "http://localhost:1",
           "--log-json"])
    assert started == []


def test_the_view_starts_and_stops_on_a_tty(monkeypatch, healthy):
    seen: dict = {}
    _spy_loop(monkeypatch, seen)
    events = []
    monkeypatch.setattr("flashnode.status.StatusView.start",
                        lambda self: events.append("start"))
    monkeypatch.setattr("flashnode.status.StatusView.stop",
                        lambda self: events.append("stop"))
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    _work(["--runner", "subprocess", "--coordinator", "http://localhost:1"])
    assert events == ["start", "stop"]
