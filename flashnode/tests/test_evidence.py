"""Execution evidence: what this agent measures and reports at commit time.

Slice 2 of the result-verification design, agent side. The coordinator
accepts an optional `ExecutionEvidence` block on `complete`; this is the
half that fills it in.

The rule every test here exists to defend: **never fabricate a measurement.**
If the sampler could not read a value the agent sends `None`. A `0.0`
standing in for "unknown" is precisely the value that later reads as "this
node did nothing" and gets an honest volunteer flagged — and the cheapest
way to produce one is by accident, which is why absence is asserted
explicitly everywhere below rather than left to inspection.

Every probe takes its subprocess runner as a parameter, so the whole suite
runs on a machine with no NVIDIA driver and no docker daemon — the same
reason `inventory/gpu.py` and `doctor.py` are shaped that way.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flashnode.executor import CoordinatorClient, ExecutorLoop, SubprocessRunner


# ---------------------------------------------------------------------------
# ResourceSampler: means over a run, and the difference between 0 and unknown
# ---------------------------------------------------------------------------


def _fixed(*values):
    """A probe returning each value in turn, then repeating the last."""
    seen = []

    def probe():
        seen.append(len(seen))
        return values[min(len(seen) - 1, len(values) - 1)]

    probe.calls = seen
    return probe


def _sampler(cpu_probe, gpu_probe, **over):
    from flashnode.executor.evidence import ResourceSampler

    kw = dict(cpu_probe=cpu_probe, gpu_probe=gpu_probe,
              cpu_interval_s=0.01, gpu_interval_s=0.01)
    kw.update(over)
    return ResourceSampler(**kw)


def test_means_are_averaged_over_the_run():
    sampler = _sampler(_fixed(10.0, 30.0, 50.0), _fixed(20.0, 40.0, 60.0))
    sampler.start()
    while len(sampler._cpu) < 3:  # noqa: SLF001 — pinning the sampling itself
        pass
    cpu, gpu = sampler.stop()

    assert cpu is not None and 10.0 <= cpu <= 50.0
    assert gpu is not None and 20.0 <= gpu <= 60.0


def test_a_probe_that_knows_nothing_yields_none_and_never_zero():
    """The load-bearing case. No psutil, no driver, no sampler — the honest
    report is "not measured", and it must not arrive as 0.0."""
    sampler = _sampler(_fixed(None), _fixed(None))
    sampler.start()
    cpu, gpu = sampler.stop()

    assert cpu is None
    assert gpu is None


def test_a_probe_that_reads_zero_yields_zero_and_never_none():
    """The mirror, and the reason the two must be different values: an idle
    GPU on a task that asked for one is the strongest signal slice 2 carries.
    Rounding it to "unknown" throws that away."""
    sampler = _sampler(_fixed(0.0), _fixed(0.0))
    sampler.start()
    cpu, gpu = sampler.stop()

    assert cpu == 0.0
    assert gpu == 0.0


def test_a_sampler_that_never_ran_reports_nothing():
    """Zero samples is not a mean of zero. A task shorter than one tick, or a
    sampler that failed to start, must report absence."""
    sampler = _sampler(_fixed(None), _fixed(None), cpu_interval_s=3600.0)
    assert sampler.stop() == (None, None)


def test_a_short_task_is_still_measured_by_the_final_reading():
    """Sub-tick tasks are the common case on a sweep. Taking one last reading
    at stop() is a genuine measurement of the window that just elapsed, not a
    guess — so a fast task reports a number rather than pushing the agent
    towards inventing one."""
    sampler = _sampler(_fixed(42.0), _fixed(7.0), cpu_interval_s=3600.0,
                       gpu_interval_s=3600.0)
    sampler.start()
    cpu, gpu = sampler.stop()

    assert cpu == 42.0
    assert gpu == 7.0


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param("87%", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="none"),
    ],
)
def test_a_probe_returning_junk_contributes_nothing_rather_than_poisoning_the_mean(value):
    """One NaN turns an entire run's mean into NaN, which serialises to
    something no verifier can read. Drop the reading, keep the run."""
    sampler = _sampler(_fixed(value), _fixed(value))
    sampler.start()
    assert sampler.stop() == (None, None)


def test_a_raising_probe_never_reaches_the_task():
    """This thread lives beside a live task. A probe that raises must cost a
    field, never the run."""

    def boom():
        raise RuntimeError("sensor on fire")

    sampler = _sampler(boom, boom)
    sampler.start()
    assert sampler.stop() == (None, None)


def test_the_gpu_probe_stops_being_called_on_a_host_that_has_no_driver():
    """The overwhelmingly common case is no GPU at all. Asking every few
    seconds for the length of a job means thousands of doomed subprocess
    spawns on a machine somebody lent us."""
    gpu = _fixed(None)
    sampler = _sampler(_fixed(1.0), gpu, cpu_interval_s=0.001, gpu_interval_s=0.001)
    sampler.start()
    import time

    time.sleep(0.05)
    sampler.stop()

    assert len(gpu.calls) == 1  # asked once, told no, never asked again


def test_a_transient_gpu_failure_does_not_disable_a_real_gpu_host():
    """The mirror of the above: once a reading has succeeded the driver
    demonstrably exists, so a single blip must not cost the rest of the run's
    GPU evidence."""
    gpu = _fixed(50.0, None, 70.0)
    sampler = _sampler(_fixed(1.0), gpu, cpu_interval_s=0.001, gpu_interval_s=0.001)
    sampler.start()
    while len(gpu.calls) < 3:
        pass
    _, mean = sampler.stop()

    assert len(gpu.calls) >= 3
    assert mean is not None and mean > 0.0


def test_the_default_cpu_probe_discards_psutils_priming_reading(monkeypatch):
    """`psutil.cpu_percent(interval=None)` measures since the LAST call, so
    its first answer in a process is 0.0 by construction. Reporting that
    would be a fabricated zero of exactly the kind this slice must never
    produce — the first reading is burned at construction, before the run."""
    import psutil

    from flashnode.executor.evidence import default_cpu_probe

    readings = iter([0.0, 61.0, 63.0])
    monkeypatch.setattr(psutil, "cpu_percent", lambda **kw: next(readings))

    probe = default_cpu_probe()  # burns the 0.0
    assert probe() == 61.0
    assert probe() == 63.0


def test_the_default_cpu_probe_degrades_to_none_without_psutil(monkeypatch):
    import builtins

    from flashnode.executor.evidence import default_cpu_probe

    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    assert default_cpu_probe()() is None


# ---------------------------------------------------------------------------
# image_digest: which bytes actually ran
# ---------------------------------------------------------------------------


DIGEST = "sha256:" + "3f" * 32


def _docker(stdout: bytes = b"", returncode: int = 0):
    calls: list[tuple[tuple, dict]] = []

    def run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(list(argv), returncode, stdout, b"")

    run.calls = calls
    return run


def test_the_digest_of_the_image_that_ran_is_reported():
    from flashnode.executor.evidence import image_digest

    assert image_digest("ghcr.io/x/y:v1", run=_docker(DIGEST.encode() + b"\n")) == DIGEST


def test_the_digest_query_names_the_image_and_is_bounded_in_time():
    from flashnode.executor.evidence import image_digest

    run = _docker(DIGEST.encode())
    image_digest("ghcr.io/x/y:v1", run=run)
    argv, kwargs = run.calls[0]

    assert argv[:3] == ("docker", "image", "inspect")
    assert "ghcr.io/x/y:v1" in argv
    assert kwargs["timeout"] <= 10


@pytest.mark.parametrize(
    "stdout, returncode",
    [
        pytest.param(b"", 1, id="unknown-image"),
        pytest.param(b"", 0, id="empty-output"),
        pytest.param(b"<no value>\n", 0, id="template-miss"),
        pytest.param(b"Error: No such image\n", 0, id="prose-error"),
        pytest.param(b"sha256:short\n", 0, id="truncated-digest"),
    ],
)
def test_anything_but_a_digest_is_reported_as_unknown(stdout, returncode):
    """`""` is the unknown sentinel for this field. An unparsed error string
    forwarded as a digest would be a fabricated measurement that also looks
    convincing."""
    from flashnode.executor.evidence import image_digest

    assert image_digest("x:1", run=_docker(stdout, returncode)) == ""


@pytest.mark.parametrize(
    "boom",
    [
        lambda argv, **kw: (_ for _ in ()).throw(FileNotFoundError("no docker")),
        lambda argv, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("docker", 5)),
        lambda argv, **kw: (_ for _ in ()).throw(RuntimeError("kaboom")),
        lambda argv, **kw: object(),
    ],
)
def test_the_digest_probe_never_raises(boom):
    from flashnode.executor.evidence import image_digest

    assert image_digest("x:1", run=boom) == ""


def test_a_stub_runner_returning_a_non_string_stdout_is_unknown_not_a_crash():
    """The runner is a parameter, so `stdout` is whatever the caller's double
    put there — a `Mock` in several existing suites. A duck-typed `.decode`
    returning another object must not walk a non-string into the digest
    check and raise out of a best-effort probe."""
    from unittest import mock

    from flashnode.executor.evidence import image_digest

    assert image_digest("x:1", run=lambda argv, **kw: mock.Mock(returncode=0)) == ""


def test_no_image_reference_means_no_digest_and_no_subprocess():
    from flashnode.executor.evidence import image_digest

    run = _docker(DIGEST.encode())
    assert image_digest("", run=run) == ""
    assert run.calls == []


# ---------------------------------------------------------------------------
# Runners report what they actually did
# ---------------------------------------------------------------------------


TASK_MODULE = """
import argparse, json, pathlib
p = argparse.ArgumentParser(); p.add_argument("--spec"); p.add_argument("--out")
a = p.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text("{}")
"""


@pytest.fixture()
def fake_module(tmp_path, monkeypatch):
    pkg = tmp_path / "evidence_workloads"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "trial.py").write_text(TASK_MODULE)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return "evidence_workloads.trial"


def test_the_subprocess_runner_reports_the_real_exit_code_and_no_image(tmp_path, fake_module):
    """Tier 1 runs `python -m` on the host: there is no image, so the digest
    stays the unknown sentinel rather than echoing back the payload's."""
    runner = SubprocessRunner(allowed_modules=frozenset({fake_module}))
    runner.run({"module": fake_module, "task_id": "t1", "image": "ghcr.io/x/y:v1"}, tmp_path, {})

    assert runner.last_exit_code == 0
    assert runner.last_image_digest == ""


def test_a_refused_task_never_leaves_the_previous_tasks_measurement_behind(
    tmp_path, fake_module
):
    """Stale evidence IS fabricated evidence: it is a real measurement of a
    different run, wearing this one's name. Every run resets first."""
    runner = SubprocessRunner(allowed_modules=frozenset({fake_module}))
    runner.run({"module": fake_module, "task_id": "t1"}, tmp_path, {})
    assert runner.last_exit_code == 0

    from flashnode.executor.runner import TaskExecutionError

    with pytest.raises(TaskExecutionError):
        runner.run({"module": "os.system"}, tmp_path, {})

    assert runner.last_exit_code is None


def test_the_docker_runner_reports_the_digest_of_the_image_it_ran(tmp_path, monkeypatch):
    from flashnode.executor.docker_runner import DockerRunner

    image = "ghcr.io/example/trial:v1"

    def fake_run(argv, **kw):
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, DIGEST.encode(), b"")
        for i, part in enumerate(argv):
            if part == "-v":
                out = Path(argv[i + 1].split(":", 1)[0]) / "out"
                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text("{}")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr("flashnode.executor.docker_runner.subprocess.run", fake_run)
    runner = DockerRunner(
        allowed_images=frozenset({image}),
        allowed_modules=frozenset({"flashml_workloads.sklearn_trial"}),
    )
    runner.run(
        {"module": "flashml_workloads.sklearn_trial", "task_id": "t1", "image": image},
        tmp_path, {},
    )

    assert runner.last_image_digest == DIGEST
    assert runner.last_exit_code == 0


def test_the_argv_runner_reports_the_digest_of_the_image_it_ran(tmp_path, monkeypatch):
    from flashnode.executor.argv_runner import ArgvDockerRunner

    image = "ghcr.io/example/trial:v1"

    def fake_run(argv, **kw):
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, DIGEST.encode(), b"")
        for i, part in enumerate(argv):
            if part == "-v":
                out = Path(argv[i + 1].split(":", 1)[0]) / "out"
                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text("{}")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr("flashnode.executor.argv_runner.subprocess.run", fake_run)
    runner = ArgvDockerRunner(allowed_images=frozenset({image}))
    runner.run({"argv": ["python", "train.py"], "task_id": "t1", "image": image}, tmp_path, {})

    assert runner.last_image_digest == DIGEST
    assert runner.last_exit_code == 0


def test_a_docker_host_with_no_inspectable_image_reports_no_digest(tmp_path, monkeypatch):
    """Best effort, like every other field: an image the daemon will not
    describe costs the digest and nothing else."""
    from flashnode.executor.docker_runner import DockerRunner

    image = "ghcr.io/example/trial:v1"

    def fake_run(argv, **kw):
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 1, b"", b"No such image")
        for i, part in enumerate(argv):
            if part == "-v":
                out = Path(argv[i + 1].split(":", 1)[0]) / "out"
                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text("{}")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr("flashnode.executor.docker_runner.subprocess.run", fake_run)
    runner = DockerRunner(
        allowed_images=frozenset({image}),
        allowed_modules=frozenset({"flashml_workloads.sklearn_trial"}),
    )
    runner.run(
        {"module": "flashml_workloads.sklearn_trial", "task_id": "t1", "image": image},
        tmp_path, {},
    )

    assert runner.last_image_digest == ""


# ---------------------------------------------------------------------------
# The loop attaches it to the commit
# ---------------------------------------------------------------------------


class Stub:
    """A coordinator double recording the exact JSON body of every commit."""

    def __init__(self):
        self.leases: list[dict] = []
        self.artifacts: dict[str, bytes] = {}
        self.complete_bodies: list[dict] = []

    def make_lease(self, task_id="t1", payload=None):
        self.leases.append({
            "lease_id": f"ls-{task_id}", "task_id": task_id, "job_id": "j1",
            "node_id": "n1", "attempt_number": 1,
            "deadline": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(),
            "payload": payload or {},
        })

    def request(self, method, path, body=None, content_type=None, headers=None):
        if path == "/v1alpha1/leases/claim":
            if not self.leases:
                return 204, b""
            return 200, json.dumps(self.leases.pop(0)).encode()
        if path.endswith("/heartbeat"):
            return 200, b"{}"
        if path.endswith("/complete"):
            self.complete_bodies.append(json.loads(body))
            return 200, b'{"accepted": true}'
        if path.endswith("/fail"):
            return 200, b"{}"
        if path.startswith("/v1alpha1/artifacts/"):
            key = path.removeprefix("/v1alpha1/artifacts/")
            if method == "PUT":
                self.artifacts[key] = body
                return 200, b"{}"
            return (200, self.artifacts[key]) if key in self.artifacts else (404, b"{}")
        raise AssertionError(f"unexpected {method} {path}")


@pytest.fixture()
def stub(monkeypatch):
    stub = Stub()
    client = CoordinatorClient("http://stub")
    monkeypatch.setattr(
        client, "_request",
        lambda m, p, body=None, content_type="", headers=None: stub.request(m, p, body),
    )
    return stub, client


class QuietRunner:
    """A runner that measures nothing about itself — a third-party tier, or
    any runner written before evidence existed."""

    def run(self, payload, workdir, inputs):
        outdir = Path(workdir) / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "metrics.json").write_text("{}")
        return outdir


def _loop(client, runner, cpu=None, gpu=None):
    from flashnode.executor.evidence import ResourceSampler

    return ExecutorLoop(
        client, "n1", runner=runner,
        sampler_factory=lambda: ResourceSampler(
            cpu_probe=(lambda: cpu), gpu_probe=(lambda: gpu),
            cpu_interval_s=3600.0, gpu_interval_s=3600.0,
        ),
    )


def test_the_commit_carries_the_evidence_the_agent_measured(stub, fake_module):
    coordinator, client = stub
    coordinator.make_lease("t1", payload={"module": fake_module, "task_id": "t1"})
    runner = SubprocessRunner(allowed_modules=frozenset({fake_module}))

    assert _loop(client, runner, cpu=64.0, gpu=91.0).run(max_tasks=1, idle_exit=True) == 1

    body = coordinator.complete_bodies[0]
    assert set(body) == {"output_sha256", "evidence"}
    evidence = body["evidence"]
    assert evidence["wall_seconds"] > 0.0
    assert evidence["cpu_percent_mean"] == 64.0
    assert evidence["gpu_util_percent_mean"] == 91.0
    assert evidence["exit_code"] == 0
    assert evidence["image_digest"] == ""


def test_a_host_that_can_measure_nothing_still_commits_and_reports_nulls(stub):
    """Absence must travel as null. A verifier reading these as zeros would
    see a machine that burned no CPU on a task it completed."""
    coordinator, client = stub
    coordinator.make_lease("t1", payload={"task_id": "t1"})

    assert _loop(client, QuietRunner()).run(max_tasks=1, idle_exit=True) == 1

    evidence = coordinator.complete_bodies[0]["evidence"]
    assert evidence["cpu_percent_mean"] is None
    assert evidence["gpu_util_percent_mean"] is None
    assert evidence["exit_code"] is None       # this runner measured none
    assert evidence["image_digest"] == ""
    assert evidence["wall_seconds"] > 0.0      # the one thing always knowable


def test_wall_seconds_times_the_run_and_not_the_upload(stub):
    """The agent's clock covers the runner and nothing else. Folding the
    artifact upload in would inflate every reading by however slow the
    coordinator was that minute, and inflation is the direction a liar
    wants."""
    import time

    coordinator, client = stub
    coordinator.make_lease("t1", payload={"task_id": "t1"})

    class SlowRunner(QuietRunner):
        def run(self, payload, workdir, inputs):
            time.sleep(0.2)
            return super().run(payload, workdir, inputs)

    assert _loop(client, SlowRunner()).run(max_tasks=1, idle_exit=True) == 1
    assert coordinator.complete_bodies[0]["evidence"]["wall_seconds"] >= 0.2


def test_a_client_asked_for_no_evidence_sends_the_body_it_always_sent(stub):
    """Compatibility in the other direction: this agent may be talking to a
    coordinator that predates the field. With nothing to say, the request is
    byte-for-byte the one every previous release sent."""
    coordinator, client = stub

    assert client.complete("ls-1", "a" * 64) is True
    assert coordinator.complete_bodies[0] == {"output_sha256": "a" * 64}
