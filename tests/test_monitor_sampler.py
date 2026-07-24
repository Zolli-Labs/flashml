# tests/test_monitor_sampler.py
"""ResourceSampler: telemetry.jsonl appends, optional-psutil degradation,
and totality (a raising psutil can never crash the sampler thread)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from flashruntime.monitor import ResourceSampler
from flashruntime.monitor import sampler as sampler_mod


# ---- a minimal fake psutil ------------------------------------------------


class _FakeVm:
    total = 16_000_000_000
    available = 8_000_000_000


class _FakeMemInfo:
    rss = 123_000_000


class _FakeProc:
    """One fake process node; children() fans out one child under the root."""

    def __init__(self, pid, ppid=None, children=()):
        self.pid = pid
        self._ppid = ppid
        self._children = list(children)

    # psutil.Process API used by the sampler
    def children(self, recursive=False):
        return self._children

    def oneshot(self):
        import contextlib

        return contextlib.nullcontext()

    def ppid(self):
        return self._ppid

    def cmdline(self):
        return ["python", "train.py", "--x", "1"]

    def name(self):
        return "python"

    def cpu_percent(self, interval=None):
        return 12.5

    def memory_info(self):
        return _FakeMemInfo()

    def create_time(self):
        return 1000.0

    def status(self):
        return "running"


class _FakePsutil:
    def cpu_percent(self, interval=None):
        return 42.0

    def virtual_memory(self):
        return _FakeVm()

    def Process(self, pid):
        return _FakeProc(pid, ppid=1, children=[_FakeProc(pid + 1, ppid=pid)])


class _RaisingPsutil:
    """Every call explodes — the sampler must degrade, not propagate."""

    def cpu_percent(self, interval=None):
        raise RuntimeError("boom")

    def virtual_memory(self):
        raise RuntimeError("boom")

    def Process(self, pid):
        raise RuntimeError("boom")


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---- machine sample -------------------------------------------------------


def test_machine_sample_with_psutil(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_psutil", lambda: _FakePsutil())
    m = sampler_mod._machine_sample()
    assert m["limited"] is False
    assert m["cpu_percent"] == 42.0
    assert m["mem_total"] == 16_000_000_000
    assert m["mem_used"] == 8_000_000_000
    assert isinstance(m["hostname"], str) and m["hostname"]
    assert isinstance(m["gpus"], list)


def test_machine_sample_without_psutil(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_psutil", lambda: None)
    m = sampler_mod._machine_sample()
    assert m["limited"] is True
    assert m["cpu_percent"] is None and m["mem_total"] is None and m["mem_used"] is None
    assert m["cpu_count"] is not None  # stdlib facts still present


def test_machine_sample_survives_raising_psutil(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_psutil", lambda: _RaisingPsutil())
    m = sampler_mod._machine_sample()  # must not raise
    assert m["limited"] is True
    assert m["cpu_percent"] is None


# ---- process tree -----------------------------------------------------------


def test_process_tree_with_psutil(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_psutil", lambda: _FakePsutil())
    procs = sampler_mod._process_tree(4000)
    assert [p["pid"] for p in procs] == [4000, 4001]
    assert procs[0]["cpu_percent"] == 12.5
    assert procs[0]["rss_bytes"] == 123_000_000
    assert procs[1]["ppid"] == 4000
    assert procs[0]["cmd"] == "python train.py --x"


def test_process_tree_without_psutil(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_psutil", lambda: None)
    procs = sampler_mod._process_tree(4000)
    assert procs == [
        {
            "pid": 4000,
            "ppid": None,
            "cmd": None,
            "cpu_percent": None,
            "rss_bytes": None,
            "create_time": None,
            "status": None,
        }
    ]


def test_process_tree_survives_raising_psutil(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_psutil", lambda: _RaisingPsutil())
    assert sampler_mod._process_tree(4000) == []  # degrade, never raise


# ---- the sampler thread -----------------------------------------------------


def test_sampler_writes_at_least_one_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(sampler_mod, "_psutil", lambda: _FakePsutil())
    s = ResourceSampler(tmp_path, root_pid=4000, period_s=30.0)  # long period:
    s.start()  # the first tick must happen immediately, not after period_s
    deadline = time.time() + 5
    path = tmp_path / "telemetry.jsonl"
    while time.time() < deadline and not path.exists():
        time.sleep(0.02)
    s.stop()
    lines = _read_lines(path)
    assert len(lines) >= 1
    sample = lines[0]
    assert set(sample) == {"ts", "machine", "processes"}
    assert sample["machine"]["cpu_percent"] == 42.0
    assert [p["pid"] for p in sample["processes"]] == [4000, 4001]


def test_sampler_stop_is_idempotent_and_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(sampler_mod, "_psutil", lambda: None)
    s = ResourceSampler(tmp_path, root_pid=1, period_s=30.0)
    s.start()
    t0 = time.time()
    s.stop()
    s.stop()  # second stop must be a no-op, not an error
    assert time.time() - t0 < 5  # never waits out the period
