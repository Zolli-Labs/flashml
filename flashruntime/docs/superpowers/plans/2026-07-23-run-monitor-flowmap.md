# Run Monitor Flow Map + Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the live run viewer into a real-time process flow map (machine → workers → ranks with PIDs) plus a KPI monitoring dashboard, backed by new telemetry (optional-psutil resource sampler + per-rank heartbeats).

**Architecture:** Three layers — (1) `flashruntime/monitor/` samples machine + process-tree stats to `telemetry.jsonl` per attempt; the `flashruntime.torch` helper writes per-rank heartbeat JSON files; (2) `viewer/state.py` enriches the `/api/state` snapshot with both (all reads total/degrading); (3) `viewer/flowmap.py` is a new shared UI component (CSS/JS strings) that `viewer/page.py` assembles into layout A (KPI strip → flow map with slide-in detail panel → charts → existing sections). Spec: `docs/superpowers/specs/2026-07-23-run-monitor-flowmap-design.md`.

**Tech Stack:** Python ≥3.10 stdlib (threading, json), optional psutil (`[monitor]` extra), plain inline JS/CSS (no framework, no CDN, no build step), pytest.

## Global Constraints

- **Core stays pydantic-only:** `import flashruntime` must never require psutil/numpy/torch. psutil is imported lazily inside functions only.
- **Viewer reads are TOTAL:** every disk hazard (torn JSON, missing file, unreadable dir) degrades to a partial snapshot, NEVER an exception (doctrine in `flashruntime/viewer/state.py` module docstring).
- **The sampler and heartbeats must never affect the run:** swallow all exceptions; a watched/instrumented run can never be crashed by observability.
- **Zero external assets in pages:** no CDN, no remote font/script/image. `tests/test_viewer_page.py::test_render_has_no_offhost_asset_references` enforces this.
- **Colors only from `viewer.page.TOKENS`:** CSS uses `%%token%%` placeholders; JS reads the injected global `T` object. Never hardcode a hex color in new CSS/JS.
- **Honest numbers:** a KPI with no data source renders `—` (and, where psutil is the missing source, the hint `flashruntime[monitor]`). Never invent a number.
- **ADR-0003 guardrail:** the torch helper's public surface does not grow. The heartbeat is a private function (`_write_heartbeat`), not a new verb; `__all__` is unchanged.
- **Code style:** match existing modules — explanatory docstrings that state WHY, comments only for constraints code can't show. JS is plain, commented, organized by section.
- Run tests from the repo root: `/Users/phongcao/Work/Zolli-Labs/flashruntime` with its `.venv` active (`source .venv/bin/activate` or use `.venv/bin/pytest`).

---

### Task 1: `flashruntime/monitor/` — ResourceSampler

**Files:**
- Create: `flashruntime/monitor/__init__.py`
- Create: `flashruntime/monitor/sampler.py`
- Create: `tests/test_monitor_sampler.py`
- Modify: `pyproject.toml` (add `monitor` extra in `[project.optional-dependencies]`, after the `oss` line)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `flashruntime.monitor.ResourceSampler(output_dir: Path|str, root_pid: int, period_s: float = 2.0)` with `.start() -> None` and `.stop() -> None`. Appends one JSON line per tick to `<output_dir>/telemetry.jsonl` shaped:
  `{"ts": float, "machine": {"hostname": str, "cpu_count": int|null, "load_avg": [f,f,f]|null, "cpu_percent": f|null, "mem_total": int|null, "mem_used": int|null, "gpus": [{"name","util_percent","mem_used_mb","mem_total_mb"}], "limited": bool}, "processes": [{"pid": int, "ppid": int|null, "cmd": str|null, "cpu_percent": f|null, "rss_bytes": int|null, "create_time": f|null, "status": str|null}]}`
  Module-level seams for tests: `_psutil()` (returns the psutil module or None) and `_gpu_sample()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monitor_sampler.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_monitor_sampler.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'flashruntime.monitor'`

- [ ] **Step 3: Write the implementation**

Create `flashruntime/monitor/sampler.py`:

```python
"""ResourceSampler — per-attempt machine + process-tree telemetry.

A daemon thread `flash.submit` starts next to each launched attempt. Every
`period_s` it appends one JSON line to `<output_dir>/telemetry.jsonl`:

    {"ts": ..., "machine": {...}, "processes": [...]}

The viewer tail-reads this file exactly like metrics.jsonl (bounded window,
torn last line skipped), so plain appends are the right write discipline.

psutil is OPTIONAL (the `monitor` extra): with it, cpu/mem and per-process
stats are real; without it the machine sample carries `"limited": true` and
only stdlib facts (hostname, cpu_count, load_avg) — the dashboard renders
the gaps honestly instead of faking numbers. GPU facts come from nvidia-smi
when present (macOS/CPU boxes simply get `[]`).

TOTAL by contract, like everything that watches a run: a tick swallows every
exception, and the thread can never affect the training process it observes.
psutil is imported lazily inside `_psutil()` so `import flashruntime` stays
pydantic-only (the clean-venv core smoke pins this).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def _psutil():
    """The psutil module, or None when the `monitor` extra is not installed.
    A function (not a module-level try/import) so tests can monkeypatch the
    seam and so the import cost is paid only inside the sampler thread."""
    try:
        import psutil

        return psutil
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return None


def _machine_sample() -> dict:
    """One machine-level sample. stdlib facts always; psutil facts when
    available; `limited` is True whenever the psutil numbers are absent
    (missing OR failing) so the UI can say 'install flashruntime[monitor]'."""
    sample: dict = {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cpu_percent": None,
        "mem_total": None,
        "mem_used": None,
        "gpus": _gpu_sample(),
        "limited": True,
    }
    ps = _psutil()
    if ps is not None:
        try:
            sample["cpu_percent"] = ps.cpu_percent(interval=None)
            vm = ps.virtual_memory()
            sample["mem_total"] = vm.total
            sample["mem_used"] = vm.total - vm.available
            sample["limited"] = False
        except Exception:  # noqa: BLE001 — degrade to the stdlib-only sample
            pass
    return sample


def _gpu_sample() -> list[dict]:
    """GPU name/util/memory rows via nvidia-smi, or [] when there is no
    nvidia-smi (macOS, CPU boxes) or it misbehaves. A subprocess (not torch)
    so the sampler works for non-torch workloads and never imports a
    framework."""
    try:
        out = subprocess.run(_GPU_QUERY, capture_output=True, text=True, timeout=2)
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            name, util, used, total = [p.strip() for p in line.split(",")]
            gpus.append(
                {
                    "name": name,
                    "util_percent": float(util),
                    "mem_used_mb": float(used),
                    "mem_total_mb": float(total),
                }
            )
        return gpus
    except Exception:  # noqa: BLE001 — no GPU story is ever worth an exception
        return []


def _process_tree(root_pid: int) -> list[dict]:
    """The launched process and all its descendants (torchrun's ranks, a
    sweep's children), one dict per process. Without psutil we can still name
    the root pid (the launcher knows it); with a vanished/denied root we
    return [] — the attempt likely just finished between ticks."""
    ps = _psutil()
    if ps is None:
        return [
            {
                "pid": root_pid,
                "ppid": None,
                "cmd": None,
                "cpu_percent": None,
                "rss_bytes": None,
                "create_time": None,
                "status": None,
            }
        ]
    try:
        root = ps.Process(root_pid)
        procs = [root] + root.children(recursive=True)
    except Exception:  # noqa: BLE001 — process gone / access denied
        return []
    out: list[dict] = []
    for p in procs:
        try:
            with p.oneshot():
                out.append(
                    {
                        "pid": p.pid,
                        "ppid": p.ppid(),
                        "cmd": " ".join(p.cmdline()[:3]) or p.name(),
                        "cpu_percent": p.cpu_percent(interval=None),
                        "rss_bytes": p.memory_info().rss,
                        "create_time": p.create_time(),
                        "status": p.status(),
                    }
                )
        except Exception:  # noqa: BLE001 — a process may exit mid-inspection
            continue
    return out


class ResourceSampler:
    """Sample machine + process-tree telemetry for one launched attempt.

    `start()` spawns a daemon thread that ticks immediately (a sub-period
    command must still leave one sample) and then every `period_s`;
    `stop()` wakes and joins it. Both are idempotent-enough for the SDK's
    best-effort use: stop on a never-started sampler is a no-op.
    """

    def __init__(self, output_dir: Path | str, root_pid: int, period_s: float = 2.0):
        self.output_dir = Path(output_dir)
        self.root_pid = int(root_pid)
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.period_s + 1)
            self._thread = None

    def _run(self) -> None:
        # tick FIRST, then wait: a command that finishes inside one period
        # still gets a sample, and stop() during the wait exits promptly.
        while True:
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — total by contract (module docstring)
                pass
            if self._stop.wait(self.period_s):
                return

    def _tick(self) -> None:
        sample = {
            "ts": time.time(),
            "machine": _machine_sample(),
            "processes": _process_tree(self.root_pid),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "telemetry.jsonl", "a") as f:
            f.write(json.dumps(sample) + "\n")
```

Create `flashruntime/monitor/__init__.py`:

```python
"""Run telemetry: the optional-psutil resource sampler the SDK starts per
launched attempt. Read side: `viewer.state.collect()` tails telemetry.jsonl.
"""

from flashruntime.monitor.sampler import ResourceSampler

__all__ = ["ResourceSampler"]
```

In `pyproject.toml`, add after the `oss = [...]` line in `[project.optional-dependencies]`:

```toml
monitor = ["psutil>=5.9"]                        # machine + process-tree telemetry for the run viewer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_monitor_sampler.py -v`
Expected: 8 passed

- [ ] **Step 5: Verify the core-import rule still holds**

Run: `.venv/bin/python -c "import sys; import flashruntime; assert 'psutil' not in sys.modules; import flashruntime.monitor; assert 'psutil' not in sys.modules; print('core import clean')"`
Expected: `core import clean`

- [ ] **Step 6: Commit**

```bash
git add flashruntime/monitor tests/test_monitor_sampler.py pyproject.toml
git commit -m "feat(monitor): ResourceSampler — optional-psutil machine + process-tree telemetry"
```

---

### Task 2: Per-rank heartbeats in `flashruntime.torch`

**Files:**
- Modify: `flashruntime/torch/__init__.py`
- Test: `tests/test_torch_helper.py` (extend)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `<FLASHML_OUTPUT_DIR>/ranks/rank-<N>.json` written atomically by every rank, shaped:
  `{"rank": int, "local_rank": int, "pid": int, "device": str, "backend": str|null, "world_size": int, "step": int|null, "ts": float}`
  Written on `prepare()` (forced), then refreshed by `checkpoint()` and `log_metrics()` calls, throttled to ≥1 s between writes per process. Private function `_write_heartbeat(step=None, force=False)`; module globals `_last_beat: float`, `_last_step: int | None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_torch_helper.py`, first extend the `ft` fixture — add these two lines before `return ft_mod`:

```python
    monkeypatch.setattr(ft_mod, "_last_beat", 0.0)
    monkeypatch.setattr(ft_mod, "_last_step", None)
```

Then append at the end of the file:

```python
# ---- per-rank heartbeats (run-monitor telemetry) ---------------------------
# Every rank mirrors its identity + progress to ranks/rank-N.json so the run
# viewer can draw machine → worker → rank with live PIDs and steps. The write
# is best-effort and throttled; it must never be able to crash training.


def _beat_path(tmp_path):
    return tmp_path / "out" / "ranks" / "rank-0.json"


def test_heartbeat_written_and_shaped(ft, tmp_path):
    import os

    ft._write_heartbeat(step=7, force=True)
    beat = json.loads(_beat_path(tmp_path).read_text())
    assert beat["rank"] == 0 and beat["local_rank"] == 0
    assert beat["pid"] == os.getpid()
    assert beat["world_size"] == 1
    assert beat["step"] == 7
    assert beat["device"] == "cpu"  # single-process default before prepare()
    assert isinstance(beat["ts"], float)


def test_heartbeat_throttles_but_force_overrides(ft, tmp_path):
    ft._write_heartbeat(step=1, force=True)
    ft._write_heartbeat(step=2)  # inside the 1 s window — must be skipped
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 1
    ft._write_heartbeat(step=3, force=True)  # force bypasses the throttle
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 3


def test_heartbeat_remembers_last_step_when_not_given(ft, tmp_path):
    ft._write_heartbeat(step=42, force=True)
    ft._write_heartbeat(force=True)  # no step: reuse the last known one
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 42


def test_heartbeat_never_raises_on_unwritable_dir(ft, tmp_path, monkeypatch):
    # FLASHML_OUTPUT_DIR pointing at a *file* makes ranks/ uncreatable
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("FLASHML_OUTPUT_DIR", str(blocker))
    ft._write_heartbeat(step=1, force=True)  # must not raise


def test_log_metrics_refreshes_heartbeat(ft, tmp_path):
    ft.log_metrics({"loss": 0.5, "step": 9})
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 9


def test_gated_checkpoint_still_beats(ft, tmp_path):
    # every-gated checkpoint calls return before touching torch, but the
    # heartbeat (progress signal) must still fire — the loop calls this
    # every iteration and the viewer wants the live step.
    ft.checkpoint(None, step=7, every=5)  # gated: no manifest written
    assert not list((tmp_path / "ckpt").glob("step-*"))
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 7


def test_prepare_writes_initial_heartbeat(ft, tmp_path):
    model = _model()
    ft.prepare(model, None, None)
    beat = json.loads(_beat_path(tmp_path).read_text())
    assert beat["step"] == 0 and beat["device"] == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_torch_helper.py -v -k heartbeat`
Expected: FAIL with `AttributeError: ... has no attribute '_last_beat'` (fixture) / `_write_heartbeat`

- [ ] **Step 3: Implement**

In `flashruntime/torch/__init__.py`:

(a) Add `time` to the imports:

```python
import json
import os
import time
from pathlib import Path
```

(b) After the `_backend: str | None = None` line, add:

```python
# Heartbeat state: the run viewer draws machine → worker → rank from a small
# per-rank JSON each process mirrors to ranks/rank-N.json. Throttled so a
# tight training loop calling log_metrics()/checkpoint() every step costs at
# most one small atomic write per second per rank.
_last_beat = 0.0
_last_step: int | None = None
_BEAT_MIN_INTERVAL_S = 1.0


def _write_heartbeat(step: int | None = None, force: bool = False) -> None:
    """Mirror this rank's identity + progress to ranks/rank-<N>.json.

    Atomic (tmp + os.replace, the run.json idiom) so the viewer never reads
    a torn file. Best-effort by contract: observability must never be able
    to crash training, so every failure is swallowed. `step=None` reuses the
    last step this process reported (a refresh must not erase progress)."""
    global _last_beat, _last_step
    if step is not None:
        _last_step = step
    now = time.time()
    if not force and now - _last_beat < _BEAT_MIN_INTERVAL_S:
        return
    try:
        beat = {
            "rank": rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "pid": os.getpid(),
            "device": _device,
            "backend": _backend,
            "world_size": world_size(),
            "step": _last_step,
            "ts": now,
        }
        ranks_dir = _output_dir() / "ranks"
        ranks_dir.mkdir(parents=True, exist_ok=True)
        tmp = ranks_dir / f".rank-{rank()}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(beat))
        os.replace(tmp, ranks_dir / f"rank-{rank()}.json")
        _last_beat = now
    except Exception:  # noqa: BLE001 — by contract, swallow everything
        pass
```

(c) At the end of `prepare()`, immediately before `return model, optimizer, dataloader`, add:

```python
    # announce this rank to the run viewer the moment it is wired
    _write_heartbeat(step=_restored_step, force=True)
```

(d) At the top of `checkpoint()`, immediately before the `if every is not None ...` gate, add:

```python
    _write_heartbeat(step=step)  # progress signal — fires even on gated calls
```

(e) At the top of `log_metrics()`, as the first statement of the function body, add:

```python
    # every rank beats (identity/progress); only rank 0 appends metrics below
    step = metrics.get("step") if isinstance(metrics, dict) else None
    _write_heartbeat(step=step if isinstance(step, int) else None)
```

(f) In the module docstring, extend the sentence about REPORTS: change
`torch's own primitives and REPORTS launch facts — nothing more.` to
`torch's own primitives and REPORTS launch facts — nothing more (that
reporting includes a per-rank heartbeat file the run viewer reads).`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_torch_helper.py -v`
Expected: all pass (7 new + all pre-existing).

- [ ] **Step 5: Commit**

```bash
git add flashruntime/torch/__init__.py tests/test_torch_helper.py
git commit -m "feat(torch): per-rank heartbeat files for the run viewer (throttled, best-effort)"
```

---

### Task 3: Snapshot enrichment in `viewer/state.py`

**Files:**
- Modify: `flashruntime/viewer/state.py`
- Create: `tests/test_viewer_state.py`

**Interfaces:**
- Consumes: Task 1's `telemetry.jsonl` line shape; Task 2's `ranks/rank-N.json` shape (reads both from each attempt's `output_dir`).
- Produces: on the `/api/state` snapshot — each attempt row gains `telemetry: list[dict]` (last ≤150 samples) and `ranks: list[dict]` (parsed heartbeat files, sorted by rank); the document gains top-level `monitor: dict | None` (the newest telemetry sample across attempts, i.e. `{"ts", "machine", "processes"}`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_viewer_state.py`:

```python
# tests/test_viewer_state.py
"""state.collect() enrichment for the flow map: per-attempt telemetry tails
and rank heartbeats, plus the top-level `monitor` sample. All reads must be
TOTAL — torn/invalid files degrade to partial data, never exceptions."""

from __future__ import annotations

import json

from flashruntime.viewer.state import collect


def _run_dir(tmp_path, attempts):
    doc = {
        "contract": "viewer_v1",
        "workload": {"command": ["python", "train.py"], "mode": "single", "source": "."},
        "state": "RUNNING",
        "started_at": 0.0,
        "finished_at": None,
        "max_restarts": 0,
        "attempts": attempts,
        "events": [],
        "trials": [],
    }
    (tmp_path / "run.json").write_text(json.dumps(doc))
    return tmp_path


def _attempt(tmp_path, name):
    d = tmp_path / "job" / name
    d.mkdir(parents=True)
    return {
        "attempt_id": name,
        "job_id": "local",
        "state": "RUNNING",
        "pid": "4000",
        "started_at": 0.0,
        "finished_at": None,
        "output_dir": str(d),
    }


def _sample(ts, cpu):
    return {
        "ts": ts,
        "machine": {"hostname": "box", "cpu_percent": cpu, "limited": False},
        "processes": [{"pid": 4000}],
    }


def test_attempts_enriched_with_telemetry_and_ranks(tmp_path):
    a = _attempt(tmp_path, "task-000")
    out = tmp_path / "job" / "task-000"
    with open(out / "telemetry.jsonl", "w") as f:
        f.write(json.dumps(_sample(1.0, 10.0)) + "\n")
        f.write(json.dumps(_sample(2.0, 20.0)) + "\n")
        f.write('{"torn')  # unterminated last line — writer mid-append
    ranks = out / "ranks"
    ranks.mkdir()
    (ranks / "rank-1.json").write_text(json.dumps({"rank": 1, "pid": 4002, "step": 5}))
    (ranks / "rank-0.json").write_text(json.dumps({"rank": 0, "pid": 4001, "step": 5}))
    (ranks / "rank-2.json").write_text("{torn")  # must be skipped, not fatal

    snap = collect(_run_dir(tmp_path, [a]))
    row = snap["attempts"][0]
    assert [t["ts"] for t in row["telemetry"]] == [1.0, 2.0]  # torn line skipped
    assert [r["rank"] for r in row["ranks"]] == [0, 1]  # sorted, torn skipped
    assert snap["monitor"]["ts"] == 2.0  # newest sample becomes the monitor


def test_monitor_picks_newest_across_attempts(tmp_path):
    a0 = _attempt(tmp_path, "task-000")
    a1 = _attempt(tmp_path, "task-001")
    with open(tmp_path / "job" / "task-000" / "telemetry.jsonl", "w") as f:
        f.write(json.dumps(_sample(5.0, 50.0)) + "\n")
    with open(tmp_path / "job" / "task-001" / "telemetry.jsonl", "w") as f:
        f.write(json.dumps(_sample(9.0, 90.0)) + "\n")
    snap = collect(_run_dir(tmp_path, [a0, a1]))
    assert snap["monitor"]["machine"]["cpu_percent"] == 90.0


def test_absent_telemetry_degrades_to_empty(tmp_path):
    a = _attempt(tmp_path, "task-000")
    snap = collect(_run_dir(tmp_path, [a]))
    row = snap["attempts"][0]
    assert row["telemetry"] == [] and row["ranks"] == []
    assert snap["monitor"] is None


def test_missing_output_dir_still_total(tmp_path):
    a = _attempt(tmp_path, "task-000")
    a["output_dir"] = str(tmp_path / "job" / "vanished")
    snap = collect(_run_dir(tmp_path, [a]))
    assert "error" not in snap
    assert snap["attempts"][0]["telemetry"] == []
    assert snap["attempts"][0]["ranks"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_viewer_state.py -v`
Expected: FAIL with `KeyError: 'telemetry'` (and `'monitor'`)

- [ ] **Step 3: Implement**

In `flashruntime/viewer/state.py`:

(a) After the `_LOG_TAIL = 100` line, add:

```python
_TELEMETRY_TAIL = 150  # last N sampler ticks — a ~5 min usage chart at 2 s period
```

(b) In `_collect()`, inside the `for attempt in ...` loop, after the `row["log_tail"] = ...` line, add:

```python
        row["telemetry"] = _telemetry_tail(attempt_dir) if attempt_dir else []
        row["ranks"] = _ranks(attempt_dir) if attempt_dir else []
```

(c) In `_collect()`, after the `doc["checkpoints"] = manifests` line and before `return doc`, add:

```python
    # The newest telemetry sample anywhere becomes the machine-level `monitor`
    # object (KPI tiles + machine node read it). Newest-by-ts, not "the
    # running attempt's", so a just-finished run still shows its last stats.
    monitor = None
    for row in enriched:
        tel = row.get("telemetry") or []
        cand = tel[-1] if tel else None
        if isinstance(cand, dict) and (monitor is None or cand.get("ts", 0) > monitor.get("ts", 0)):
            monitor = cand
    doc["monitor"] = monitor
```

(d) After `_metrics_tail()`, add two readers (same totality doctrine):

```python
def _telemetry_tail(attempt_dir: Path, limit: int = _TELEMETRY_TAIL) -> list[dict]:
    """Last `limit` sampler ticks from `telemetry.jsonl` — the same bounded
    tail-window + skip-torn-lines discipline as `_metrics_tail` (the sampler
    appends live while we read)."""
    records: list[dict] = []
    for line in _tail_window_lines(attempt_dir / "telemetry.jsonl"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # torn/partial line — skip it, keep the rest
        if isinstance(rec, dict):
            records.append(rec)
    return records[-limit:]


def _ranks(attempt_dir: Path) -> list[dict]:
    """Every parseable `ranks/rank-*.json` heartbeat, sorted by rank. A torn
    file (a rank mid-os.replace cannot tear, but a foreign writer could) is
    skipped; a missing dir is simply 'not instrumented' — []."""
    try:
        paths = sorted((attempt_dir / "ranks").glob("rank-*.json"))
    except OSError:
        return []
    out: list[dict] = []
    for p in paths:
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    out.sort(key=lambda r: r.get("rank", 0) if isinstance(r.get("rank", 0), int) else 0)
    return out
```

(e) In the module docstring, extend the enrichment sentence: after `and `log_tail` (launcher.log tail)` insert `, `telemetry` (sampler tail) and `ranks` (per-rank heartbeats),` and after the sentence about the top-level `checkpoints` list append ` A top-level `monitor` object carries the newest telemetry sample (machine stats + process tree).`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_viewer_state.py tests/test_viewer_server.py -v`
Expected: all pass (server tests prove nothing regressed).

- [ ] **Step 5: Commit**

```bash
git add flashruntime/viewer/state.py tests/test_viewer_state.py
git commit -m "feat(viewer): snapshot enrichment — telemetry tails, rank heartbeats, monitor sample"
```

---

### Task 4: SDK wiring — sampler per launched attempt

**Files:**
- Modify: `flashruntime/sdk.py`
- Test: `tests/test_sdk_submit.py` (extend)

**Interfaces:**
- Consumes: Task 1's `ResourceSampler(output_dir, root_pid, period_s)` / `.start()` / `.stop()`.
- Produces: `<attempt_output_dir>/telemetry.jsonl` exists after every `flash.submit` attempt (≥1 sample even for sub-second commands).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sdk_submit.py`:

```python
def test_submit_writes_attempt_telemetry(tmp_path):
    # the run-monitor sampler runs beside every launched attempt and must
    # leave at least one sample even for a sub-second command
    import flashruntime as flash

    source = _write_script(tmp_path, "print('quick')")
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": source}),
        output_dir=tmp_path / "out",
    )
    assert run.state.value == "SUCCEEDED"
    from pathlib import Path

    tel = Path(run.attempts[0]["output_dir"]) / "telemetry.jsonl"
    assert tel.is_file()
    lines = [json.loads(l) for l in tel.read_text().splitlines() if l.strip()]
    assert lines and set(lines[0]) == {"ts", "machine", "processes"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_sdk_submit.py::test_submit_writes_attempt_telemetry -v`
Expected: FAIL at `assert tel.is_file()`

- [ ] **Step 3: Implement**

In `flashruntime/sdk.py`:

(a) Add to the imports (after `from flashruntime.launchers.local import LocalProcessLauncher`):

```python
from flashruntime.monitor import ResourceSampler
```

(b) In `_drive()`, replace the launch/wait block

```python
            handle = launcher.launch(spec, job_id, attempt_id)
            run.record_event("LAUNCH_STARTED", f"{attempt_id} launched (pid {handle.execution_id})")
            run._add_attempt(attempt_id, job_id, handle, started_at)
            final_state = handle.wait()
```

with:

```python
            handle = launcher.launch(spec, job_id, attempt_id)
            run.record_event("LAUNCH_STARTED", f"{attempt_id} launched (pid {handle.execution_id})")
            run._add_attempt(attempt_id, job_id, handle, started_at)
            # Telemetry beside the attempt (machine + process tree → the run
            # viewer's flow map). Best-effort like the viewer itself: a
            # sampler that cannot start must never fail the run.
            sampler = None
            try:
                sampler = ResourceSampler(handle.output_dir, int(handle.execution_id))
                sampler.start()
            except Exception:  # noqa: BLE001 — observability never fails a run
                sampler = None
            try:
                final_state = handle.wait()
            finally:
                if sampler is not None:
                    sampler.stop()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sdk_submit.py tests/test_auto_recovery.py -v`
Expected: all pass (auto-recovery proves the restart loop still works around the new block).

- [ ] **Step 5: Commit**

```bash
git add flashruntime/sdk.py tests/test_sdk_submit.py
git commit -m "feat(sdk): start a ResourceSampler beside every launched attempt"
```

---

### Task 5: `viewer/flowmap.py` — the shared flow-map + KPI component

**Files:**
- Create: `flashruntime/viewer/flowmap.py`
- Create: `tests/test_viewer_flowmap.py`

**Interfaces:**
- Consumes: the snapshot shape from Task 3 (`attempts[].telemetry/ranks`, `monitor`), color tokens via a page-provided global JS object `T` and CSS `%%token%%` placeholders (`viewer.page.render()` substitutes them).
- Produces: two Python string constants:
  - `FLOWMAP_CSS` — styles for `#kpis`, `#flowmap`, `#detail` and `.fm-*` classes (contains `%%token%%` placeholders).
  - `FLOWMAP_JS` — defines JS functions `renderKpiTiles(tiles)` (tiles: `[{label, value, hint?, color?}]`), `renderFlowmap(snapshot)`, `renderDetail(snapshot)`, `fmStateColor(state)`, `fmFmtBytes(n)`, and internals (`fmEsc`, `fmDrawEdges`, selection global `fmSelected`). Requires elements `#kpis`, `#flowmap`, `#detail` and the global `T`.
  Task 6 (and later the coordinator dashboard) embed these strings.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_viewer_flowmap.py`:

```python
# tests/test_viewer_flowmap.py
"""The shared flow-map/KPI component: its JS API surface, its token
discipline (colors only via %%token%%/T — the dashboard reuses these strings
verbatim), and self-containment (no external references)."""

from __future__ import annotations

import re

from flashruntime.viewer.flowmap import FLOWMAP_CSS, FLOWMAP_JS


def test_js_defines_the_component_api():
    for name in ("renderKpiTiles", "renderFlowmap", "renderDetail", "fmStateColor", "fmFmtBytes"):
        assert f"function {name}" in FLOWMAP_JS, f"missing {name}"


def test_css_targets_the_contract_elements():
    for sel in ("#kpis", "#flowmap", "#detail", ".fm-node", ".fm-selected"):
        assert sel in FLOWMAP_CSS, f"missing selector {sel}"


def test_css_uses_token_placeholders():
    assert "%%panel%%" in FLOWMAP_CSS and "%%border%%" in FLOWMAP_CSS
    # no hardcoded hex colors — the single-source-of-truth palette rule
    stripped = re.sub(r"%%[a-z_]+%%", "", FLOWMAP_CSS)
    assert re.search(r"#[0-9a-fA-F]{6}\b", stripped) is None


def test_component_is_self_contained():
    blob = FLOWMAP_CSS + FLOWMAP_JS
    assert "://" not in blob  # no external URLs, ever
    assert "fetch(" not in FLOWMAP_JS  # polling belongs to the page, not the component
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_viewer_flowmap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flashruntime.viewer.flowmap'`

- [ ] **Step 3: Implement**

Create `flashruntime/viewer/flowmap.py` with exactly this content:

````python
"""The flow-map + KPI strip component, shared between the run viewer page
(`viewer/page.py`) and — in a later phase — the coordinator dashboard
(`service/dashboard.py`). Divergence between the two surfaces is a bug, so
the CSS and JS live here once, as strings the host page embeds inline.

Contract with the host page:
  * CSS colors are `%%token%%` placeholders resolved by the host's render()
    from `viewer.page.TOKENS` (the single source of truth).
  * The JS expects a global `T` (the injected token object) and three
    elements: `#kpis`, `#flowmap`, `#detail`.
  * The host polls its own data and calls `renderKpiTiles(tiles)`,
    `renderFlowmap(snapshot)`, `renderDetail(snapshot)` on each tick. The
    component never fetches — data acquisition is the host's business.

The map draws machine → workers (attempts) → ranks as DOM nodes (free click
targets + text layout) with one SVG layer behind them for the edges. It is
rebuilt idempotently from each snapshot; `fmSelected` (the clicked node)
survives rebuilds and feeds the slide-in detail panel with live values.
"""

from __future__ import annotations

FLOWMAP_CSS = r"""
  /* KPI strip -------------------------------------------------------------- */
  #kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; }
  .kpi { background: %%panel%%; border: 1px solid %%border%%; border-radius: 8px; padding: 8px 12px; }
  .kpi-label { color: %%muted%%; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  .kpi-value { color: %%text_bright%%; font-size: 18px; font-weight: 600; margin-top: 2px; }
  .kpi-hint { color: %%muted%%; font-size: 10px; margin-top: 2px; }

  /* flow map ---------------------------------------------------------------- */
  #flowmap { position: relative; display: flex; gap: 48px; padding: 14px;
             min-height: 180px; background: %%bg_inset%%; border-radius: 6px;
             overflow-x: auto; }
  #fm-edges { position: absolute; inset: 0; pointer-events: none; }
  .fm-col { display: flex; flex-direction: column; gap: 10px; justify-content: center;
            position: relative; z-index: 1; min-width: 180px; }
  .fm-node { background: %%panel%%; border: 1px solid %%border%%; border-radius: 8px;
             padding: 8px 10px; cursor: pointer; }
  .fm-node:hover { filter: brightness(1.25); }
  .fm-selected { outline: 2px solid %%running%%; outline-offset: 1px; }
  .fm-machine { border-color: %%running%%; }
  .fm-done { opacity: .55; }
  .fm-live .fm-title { animation: fm-pulse 2s ease-in-out infinite; }
  @keyframes fm-pulse { 50% { opacity: .55; } }
  .fm-title { color: %%text_bright%%; }
  .fm-sub { color: %%muted%%; font-size: 11px; margin-top: 2px; }
  .fm-badge { color: %%warn%%; border: 1px solid %%warn%%; border-radius: 4px;
              font-size: 10px; padding: 0 4px; }
  .fm-more { color: %%muted%%; font-size: 11px; }

  /* detail panel (slides in over the map's right edge) ---------------------- */
  #detail { position: absolute; top: 8px; right: 8px; bottom: 8px; width: min(360px, 60%);
            background: %%panel%%; border: 1px solid %%running%%; border-radius: 8px;
            padding: 14px; overflow-y: auto; z-index: 2; }
  #detail h3 { color: %%text_bright%%; font-size: 13px; margin-bottom: 10px;
               text-transform: none; letter-spacing: 0; }
  .fm-kv { display: flex; justify-content: space-between; gap: 12px; padding: 3px 0;
           border-bottom: 1px solid %%border%%; }
  .fm-kv span { color: %%muted%%; }
  .fm-kv b { color: %%text%%; font-weight: 600; text-align: right; word-break: break-all; }
  .fm-close { position: absolute; top: 8px; right: 8px; background: none;
              border: 1px solid %%border%%; color: %%muted%%; border-radius: 6px;
              cursor: pointer; padding: 2px 8px; font: inherit; }
  .fm-close:hover { color: %%text_bright%%; }
  .fm-log { margin-top: 10px; padding: 8px; background: %%bg_inset%%; border-radius: 6px;
            white-space: pre-wrap; word-break: break-word; color: %%text%%;
            font-size: 11px; max-height: 200px; overflow-y: auto; }
  .fm-hint { margin-top: 10px; color: %%warn%%; font-size: 11px; }
"""

FLOWMAP_JS = r"""
// ==== flow map + KPI components (viewer/flowmap.py — shared surface) =======
// Requires: global `T` (color tokens) and elements #kpis, #flowmap, #detail.
// The host page owns polling; these functions only render a given snapshot.

const fmEsc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Lifecycle state → accent. The shared vocabulary both surfaces use.
function fmStateColor(s) {
  switch (s) {
    case "RUNNING": return T.running;                  // cyan
    case "LEASED": case "RECOVERING": return T.warn;   // amber
    case "SUCCEEDED": case "COMPLETED": return T.ok;   // green
    case "FAILED": return T.fail;                      // red
    default: return T.muted;                           // PENDING / CANCELLED
  }
}

function fmFmtBytes(n) {
  if (typeof n !== "number" || !isFinite(n)) return "—";
  const gb = n / (1024 ** 3);
  return gb >= 1 ? gb.toFixed(1) + " GB" : Math.round(n / (1024 ** 2)) + " MB";
}

// ---- KPI strip -------------------------------------------------------------
// tiles: [{label, value, hint?, color?}]. A tile with no data source shows
// "—" (the host builds tiles that way) — the strip never invents a number.
function renderKpiTiles(tiles) {
  const el = document.getElementById("kpis");
  el.innerHTML = tiles.map((t) =>
    '<div class="kpi"><div class="kpi-label">' + fmEsc(t.label) + "</div>" +
    '<div class="kpi-value"' + (t.color ? ' style="color:' + t.color + '"' : "") + ">" +
    fmEsc(t.value) + "</div>" +
    (t.hint ? '<div class="kpi-hint">' + fmEsc(t.hint) + "</div>" : "") +
    "</div>"
  ).join("");
}

// ---- flow map ---------------------------------------------------------------
// Selection survives re-renders: nodes are keyed {kind, id} and the map is
// rebuilt from scratch on every snapshot (idempotent — no DOM bookkeeping).
let fmSelected = null; // {kind: "machine"|"worker"|"rank", id: string} | null

// newest telemetry sample carried by one attempt row (or null)
function fmLastSample(a) {
  const tel = a.telemetry || [];
  return tel.length ? tel[tel.length - 1] : null;
}

function renderFlowmap(s) {
  const el = document.getElementById("flowmap");
  const attempts = s.attempts || [];
  const machine = (s.monitor && s.monitor.machine) || null;

  // machine column — one node (multi-machine arrives with the coordinator)
  const mSel = fmSelected && fmSelected.kind === "machine";
  const host = machine ? machine.hostname : "localhost";
  const mSub = machine
    ? (machine.cpu_count || "?") + " cores" +
      (machine.cpu_percent != null ? " · cpu " + Math.round(machine.cpu_percent) + "%" : "") +
      (machine.gpus && machine.gpus.length ? " · " + machine.gpus.length + " gpu" : "")
    : "no telemetry yet";
  let html = '<div class="fm-col">' +
    '<div class="fm-node fm-machine' + (mSel ? " fm-selected" : "") +
    '" data-kind="machine" data-id="machine">' +
    '<div class="fm-title">▣ ' + fmEsc(host) + "</div>" +
    '<div class="fm-sub">' + fmEsc(mSub) + "</div></div></div>";

  // worker column — one node per attempt, newest kept (cap so a 100-trial
  // sweep stays readable; the count of hidden ones is stated, not silent)
  const shown = attempts.slice(-16);
  html += '<div class="fm-col">' + (attempts.length > shown.length
    ? '<div class="fm-more">… ' + (attempts.length - shown.length) + " earlier attempts</div>" : "");
  for (const a of shown) {
    const sel = fmSelected && fmSelected.kind === "worker" && fmSelected.id === a.attempt_id;
    const color = fmStateColor(a.state);
    const respawn = /-r\d+$/.test(a.attempt_id || "");
    const mark = a.state === "RUNNING" ? "●" : a.state === "SUCCEEDED" ? "✓"
      : a.state === "FAILED" ? "✗" : "○";
    html += '<div class="fm-node fm-worker' + (sel ? " fm-selected" : "") +
      (a.state === "RUNNING" ? " fm-live" : " fm-done") + '"' +
      ' data-kind="worker" data-id="' + fmEsc(a.attempt_id) + '" style="border-color:' + color + '">' +
      '<div class="fm-title" style="color:' + color + '">' + mark + " " + fmEsc(a.attempt_id) +
      (respawn ? ' <span class="fm-badge">⟳ respawn</span>' : "") + "</div>" +
      '<div class="fm-sub">pid ' + fmEsc(a.pid) + " · " + fmEsc(a.state) + "</div></div>";
  }
  html += "</div>";

  // rank column — heartbeat ranks per attempt; for a live uninstrumented
  // attempt, fall back to its process tree's children (pid-only nodes)
  html += '<div class="fm-col">';
  for (const a of shown) {
    const ranks = a.ranks || [];
    const sample = fmLastSample(a);
    const procs = (sample && sample.processes) || [];
    const byPid = {};
    for (const p of procs) byPid[p.pid] = p;
    if (ranks.length) {
      for (const r of ranks) {
        const rid = a.attempt_id + "/" + r.rank;
        const sel = fmSelected && fmSelected.kind === "rank" && fmSelected.id === rid;
        const proc = byPid[r.pid];
        html += '<div class="fm-node fm-rank' + (sel ? " fm-selected" : "") +
          (a.state === "RUNNING" ? " fm-live" : " fm-done") + '"' +
          ' data-kind="rank" data-id="' + fmEsc(rid) + '" data-worker="' + fmEsc(a.attempt_id) + '">' +
          '<div class="fm-title">rank ' + fmEsc(r.rank) + " · pid " + fmEsc(r.pid) + "</div>" +
          '<div class="fm-sub">' + fmEsc(r.device || "?") +
          (r.step != null ? " · step " + fmEsc(r.step) : "") +
          (proc && proc.cpu_percent != null ? " · cpu " + Math.round(proc.cpu_percent) + "%" : "") +
          "</div></div>";
      }
    } else if (a.state === "RUNNING" && procs.length > 1) {
      for (const p of procs.slice(1, 9)) { // [0] is the launched root itself
        const rid = a.attempt_id + "/pid-" + p.pid;
        const sel = fmSelected && fmSelected.kind === "rank" && fmSelected.id === rid;
        html += '<div class="fm-node fm-rank fm-live' + (sel ? " fm-selected" : "") + '"' +
          ' data-kind="rank" data-id="' + fmEsc(rid) + '" data-worker="' + fmEsc(a.attempt_id) + '">' +
          '<div class="fm-title">pid ' + fmEsc(p.pid) + "</div>" +
          '<div class="fm-sub">' + fmEsc(p.cmd || "") +
          (p.cpu_percent != null ? " · cpu " + Math.round(p.cpu_percent) + "%" : "") +
          "</div></div>";
      }
    }
  }
  html += "</div>";

  el.innerHTML = html + '<svg id="fm-edges"></svg>';

  el.querySelectorAll(".fm-node").forEach((node) => {
    node.addEventListener("click", () => {
      fmSelected = { kind: node.dataset.kind, id: node.dataset.id };
      renderFlowmap(s); // re-render for the selection outline
      renderDetail(s);
    });
  });

  fmDrawEdges(el);
}

// Bézier connectors machine→worker and worker→its ranks, drawn into the SVG
// layer AFTER the DOM has laid the nodes out (positions read back from
// getBoundingClientRect, so the lines are correct at any width).
function fmDrawEdges(el) {
  const svg = el.querySelector("#fm-edges");
  const box = el.getBoundingClientRect();
  svg.setAttribute("width", box.width);
  svg.setAttribute("height", box.height);
  const anchor = (n, side) => {
    const r = n.getBoundingClientRect();
    return { x: (side === "r" ? r.right : r.left) - box.left,
             y: r.top + r.height / 2 - box.top };
  };
  const curve = (a, b) =>
    '<path d="M' + a.x + " " + a.y +
    " C" + ((a.x + b.x) / 2) + " " + a.y + "," +
    ((a.x + b.x) / 2) + " " + b.y + "," + b.x + " " + b.y +
    '" stroke="' + T.border + '" fill="none" stroke-width="1"/>';
  const machine = el.querySelector(".fm-machine");
  let lines = "";
  el.querySelectorAll(".fm-worker").forEach((w) => {
    if (machine) lines += curve(anchor(machine, "r"), anchor(w, "l"));
    el.querySelectorAll('.fm-rank[data-worker="' + CSS.escape(w.dataset.id) + '"]')
      .forEach((r) => { lines += curve(anchor(w, "r"), anchor(r, "l")); });
  });
  svg.innerHTML = lines;
}

// ---- detail panel -----------------------------------------------------------
// Re-rendered on every snapshot while open, so its numbers stay live.
function renderDetail(s) {
  const el = document.getElementById("detail");
  if (!fmSelected) { el.hidden = true; return; }
  const kv = (k, v) => '<div class="fm-kv"><span>' + fmEsc(k) + "</span><b>" + fmEsc(v) + "</b></div>";
  let body = "";
  if (fmSelected.kind === "machine") {
    const m = (s.monitor && s.monitor.machine) || {};
    body = "<h3>machine · " + fmEsc(m.hostname || "localhost") + "</h3>" +
      kv("cores", m.cpu_count != null ? m.cpu_count : "—") +
      kv("cpu", m.cpu_percent != null ? Math.round(m.cpu_percent) + "%" : "—") +
      kv("memory", m.mem_used != null ? fmFmtBytes(m.mem_used) + " / " + fmFmtBytes(m.mem_total) : "—") +
      kv("load avg", m.load_avg ? m.load_avg.map((x) => x.toFixed(2)).join("  ") : "—") +
      ((m.gpus || []).map((g, i) => kv("gpu " + i, g.name + " · " + Math.round(g.util_percent) +
        "% · " + Math.round(g.mem_used_mb) + "/" + Math.round(g.mem_total_mb) + " MB")).join("")) +
      (m.limited ? '<div class="fm-hint">full stats: pip install "flashruntime[monitor]"</div>' : "");
  } else {
    const attemptId = fmSelected.kind === "worker" ? fmSelected.id : fmSelected.id.split("/")[0];
    const a = (s.attempts || []).find((x) => x.attempt_id === attemptId);
    if (!a) { el.hidden = true; return; }
    if (fmSelected.kind === "worker") {
      const dur = a.finished_at
        ? (a.finished_at - a.started_at).toFixed(1) + "s"
        : ((Date.now() / 1000) - a.started_at).toFixed(0) + "s so far";
      body = "<h3>" + fmEsc(a.attempt_id) + "</h3>" +
        kv("state", a.state) + kv("pid", a.pid) + kv("job", a.job_id) + kv("runtime", dur) +
        '<div class="fm-log">' +
        fmEsc((a.log_tail || "").split("\n").slice(-12).join("\n") || "no log yet") + "</div>";
    } else {
      const key = fmSelected.id.split("/")[1];
      const sample = fmLastSample(a);
      const procs = (sample && sample.processes) || [];
      const isPid = key.startsWith("pid-");
      const r = isPid ? null : (a.ranks || []).find((x) => String(x.rank) === key);
      const pid = isPid ? Number(key.slice(4)) : r && r.pid;
      const proc = procs.find((p) => p.pid === pid);
      body = "<h3>" + (r ? "rank " + fmEsc(r.rank) : "pid " + fmEsc(pid)) + "</h3>" +
        (r ? kv("device", r.device || "—") + kv("backend", r.backend || "—") +
             kv("world size", r.world_size) + kv("step", r.step != null ? r.step : "—") : "") +
        kv("pid", pid != null ? pid : "—") +
        kv("cpu", proc && proc.cpu_percent != null ? Math.round(proc.cpu_percent) + "%" : "—") +
        kv("memory", proc ? fmFmtBytes(proc.rss_bytes) : "—") +
        kv("status", proc ? (proc.status || "—") : a.state);
    }
  }
  el.innerHTML = '<button id="fm-close" class="fm-close">✕</button>' + body;
  el.hidden = false;
  el.querySelector("#fm-close").addEventListener("click", () => {
    fmSelected = null;
    el.hidden = true;
    renderFlowmap(s);
  });
}
"""
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_viewer_flowmap.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add flashruntime/viewer/flowmap.py tests/test_viewer_flowmap.py
git commit -m "feat(viewer): flowmap.py — shared flow-map + KPI component (CSS/JS strings)"
```

---

### Task 6: Page assembly — layout A in `viewer/page.py`

**Files:**
- Modify: `flashruntime/viewer/page.py`
- Modify: `tests/test_viewer_page.py`

**Interfaces:**
- Consumes: Task 5's `FLOWMAP_CSS`/`FLOWMAP_JS` (embedded via `%%FLOWMAP_CSS%%`/`%%FLOWMAP_JS%%` placeholders); Task 3's snapshot fields.
- Produces: the rebuilt live run page. New element ids: `#kpis`, `#flowmap`, `#detail`, `#resources`. Removed: `#topology` (canvas + drawTopology/animate/roundRect/withAlpha/stateColor — fmStateColor replaces the latter).

- [ ] **Step 1: Update the page tests (they will fail until Step 3)**

In `tests/test_viewer_page.py`:

(a) Replace the body of `test_render_contains_required_section_markers` with:

```python
def test_render_contains_required_section_markers():
    html = render()
    for marker in (
        'id="kpis"',       # KPI dashboard strip
        'id="flowmap"',    # machine → workers → ranks map
        'id="detail"',     # slide-in node detail panel
        'id="resources"',  # cpu/mem usage chart
        'id="loss"',
        'id="checkpoints"',
        'id="events"',
    ):
        assert marker in html, f"missing section marker {marker}"
    # the page must poll the live snapshot endpoint
    assert "/api/state" in html
```

(b) Add two new tests after it:

```python
def test_render_resolves_every_placeholder():
    # %%token%% and %%FLOWMAP_*%% placeholders must all be substituted —
    # a leaked %% means a broken style or a missing component
    assert "%%" not in render()


def test_render_embeds_the_flowmap_component():
    from flashruntime.viewer.flowmap import FLOWMAP_JS

    html = render()
    # the component JS is embedded verbatim (tokens are substituted in CSS,
    # not JS, so the JS string survives render() unchanged)
    assert "function renderFlowmap" in html
    assert "function renderKpiTiles" in html
    assert FLOWMAP_JS[:200].strip().splitlines()[0] in html
```

(c) Run `grep -n "topology" tests/test_viewer_page.py` — if any other test still asserts on `topology`, update it to `flowmap` equivalents the same way.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_viewer_page.py -v`
Expected: the three tests above FAIL (missing markers/placeholders); the rest pass.

- [ ] **Step 3: Rebuild the page**

All edits in `flashruntime/viewer/page.py`:

(a) Add the import after `import json`:

```python
from flashruntime.viewer.flowmap import FLOWMAP_CSS, FLOWMAP_JS
```

(b) In the module docstring, replace the sentence naming the sections
(`Sections, top to bottom: header → topology canvas → loss canvas → checkpoint timeline → events feed → collapsible logs.`) with:

```
Sections, top to bottom (layout A, spec 2026-07-23): header → KPI strip →
process flow map (machine → workers → ranks; click a node for the slide-in
detail panel) → loss + resource charts → checkpoint timeline → events feed
→ collapsible logs. The flow-map/KPI component itself lives in
`viewer/flowmap.py` — shared with the coordinator dashboard.
```

(c) In the CSS block, replace the line:

```css
  #topology { height: 220px; }
  #loss { height: 200px; }
```

with:

```css
  #loss { height: 200px; }
  #resources { height: 160px; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
%%FLOWMAP_CSS%%
```

(d) In the body, replace the Topology and Loss sections:

```html
<section>
  <h2>Topology</h2>
  <div class="panel"><canvas id="topology"></canvas></div>
</section>

<section>
  <h2>Loss</h2>
  <div class="panel"><canvas id="loss"></canvas></div>
</section>
```

with:

```html
<section>
  <h2>Run</h2>
  <div id="kpis"></div>
</section>

<section>
  <h2>Process map</h2>
  <div class="panel" style="position: relative;">
    <div id="flowmap"></div>
    <div id="detail" hidden></div>
  </div>
</section>

<section class="charts">
  <div>
    <h2>Loss</h2>
    <div class="panel"><canvas id="loss"></canvas></div>
  </div>
  <div>
    <h2>Resources</h2>
    <div class="panel"><canvas id="resources"></canvas></div>
  </div>
</section>
```

(e) In the `<script>`, right after the `const $ = ...`/`esc`/`clockTime` helper block, insert the component:

```
%%FLOWMAP_JS%%
```

(f) Delete these now-dead pieces of the page JS entirely:
- `function stateColor(...)` (replaced by the component's `fmStateColor`)
- `function withAlpha(...)`
- `function roundRect(...)`
- the whole `// ---- topology canvas (animated) ---` block (`function drawTopology(...)`)
- `function animate(now) {...}` and the `requestAnimationFrame(animate);` line at the bottom.

(g) In `renderHeader`, change both `stateColor(s.state)` calls to `fmStateColor(s.state)`.

(h) After the `// ---- logs tail ---` section (after `renderLogs`), add the page-side KPI/rate/resource logic:

```js
// ---- KPI tiles + steps/s derivation ---------------------------------------
// steps/s comes from consecutive rank heartbeats (ts + step) observed across
// polls — the only honest rate source; without heartbeats the tile shows "—".
let prevBeat = null;      // {ts, step} from the previous poll
let stepsPerSec = null;   // last derived rate

function fmtDur(sec) {
  if (typeof sec !== "number" || !isFinite(sec) || sec < 0) return "—";
  const m = Math.floor(sec / 60), r = Math.floor(sec % 60), h = Math.floor(m / 60);
  return h ? h + "h " + (m % 60) + "m" : m + ":" + String(r).padStart(2, "0");
}

function newestBeat(s) {
  let best = null;
  (s.attempts || []).forEach((a) => (a.ranks || []).forEach((r) => {
    if (typeof r.step === "number" && typeof r.ts === "number" &&
        (best == null || r.ts > best.ts)) best = r;
  }));
  return best;
}

function updateRate(s) {
  const beat = newestBeat(s);
  if (!beat) return;
  if (prevBeat && beat.ts > prevBeat.ts && beat.step >= prevBeat.step) {
    const rate = (beat.step - prevBeat.step) / (beat.ts - prevBeat.ts);
    if (isFinite(rate) && rate > 0) stepsPerSec = rate;
  }
  prevBeat = { ts: beat.ts, step: beat.step };
}

function buildKpiTiles(s) {
  const m = (s.monitor && s.monitor.machine) || null;
  const beat = newestBeat(s);
  const attempts = s.attempts || [];
  const used = attempts.filter((a) => /-r\d+$/.test(a.attempt_id || "")).length;
  const verified = (s.checkpoints || []).filter((c) => c.validation === "hash_verified").length;
  const end = s.finished_at || Date.now() / 1000;
  const hint = m && m.limited ? "flashruntime[monitor]" : undefined;
  const tiles = [
    { label: "state", value: s.state || "—", color: fmStateColor(s.state) },
    { label: "elapsed", value: fmtDur(end - s.started_at) },
    { label: "step", value: beat && beat.step != null ? String(beat.step) : "—" },
    { label: "steps/s", value: stepsPerSec != null ? stepsPerSec.toFixed(2) : "—" },
    { label: "step latency", value: stepsPerSec ? Math.round(1000 / stepsPerSec) + " ms" : "—" },
    { label: "cpu", value: m && m.cpu_percent != null ? Math.round(m.cpu_percent) + "%" : "—", hint: hint },
    { label: "memory", value: m && m.mem_used != null ? fmFmtBytes(m.mem_used) : "—", hint: hint },
    { label: "restarts", value: used + " / " + (s.max_restarts == null ? "?" : s.max_restarts) },
    { label: "ckpts ✓", value: String(verified) },
  ];
  if (m && m.gpus && m.gpus.length) {
    tiles.splice(7, 0, { label: "gpu", value: Math.round(m.gpus[0].util_percent) + "%" });
  }
  return tiles;
}

// ---- resource chart ---------------------------------------------------------
// cpu% (cyan) + memory% (violet) over the telemetry tail, 0–100 fixed axis.
function collectResources(s) {
  const samples = [];
  (s.attempts || []).forEach((a) => (a.telemetry || []).forEach((t) => {
    if (t && t.machine && typeof t.ts === "number") samples.push(t);
  }));
  samples.sort((p, q) => p.ts - q.ts);
  return samples.slice(-150);
}

function drawResources() {
  const canvas = $("resources");
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!STATE || STATE.error) return;
  const samples = collectResources(STATE);
  const usable = samples.filter((t) => t.machine.cpu_percent != null);
  if (usable.length < 2) {
    ctx.fillStyle = T.muted;
    ctx.font = "11px " + fontFamily();
    ctx.textAlign = "center";
    const msg = samples.length
      ? 'cpu/mem need psutil — pip install "flashruntime[monitor]"'
      : "no telemetry yet";
    ctx.fillText(msg, w / 2, h / 2);
    return;
  }
  const padL = 8, padR = 56, padT = 10, padB = 14;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const t0 = usable[0].ts, t1 = usable[usable.length - 1].ts;
  const sx = (ts) => padL + ((ts - t0) / (t1 - t0 || 1)) * plotW;
  const sy = (pct) => padT + (1 - Math.min(100, Math.max(0, pct)) / 100) * plotH;
  ctx.strokeStyle = T.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  const series = (pick, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    usable.forEach((t) => {
      const v = pick(t);
      if (typeof v !== "number") return;
      started ? ctx.lineTo(sx(t.ts), sy(v)) : ctx.moveTo(sx(t.ts), sy(v));
      started = true;
    });
    ctx.stroke();
  };
  const memPct = (t) => t.machine.mem_total
    ? 100 * t.machine.mem_used / t.machine.mem_total : null;
  series((t) => t.machine.cpu_percent, T.running);
  series(memPct, T.ckpt);
  const last = usable[usable.length - 1];
  ctx.font = "11px " + fontFamily();
  ctx.textAlign = "left";
  ctx.fillStyle = T.running;
  ctx.fillText("cpu " + Math.round(last.machine.cpu_percent) + "%",
    padL + plotW + 4, sy(last.machine.cpu_percent));
  const mp = memPct(last);
  if (mp != null) {
    ctx.fillStyle = T.ckpt;
    ctx.fillText("mem " + Math.round(mp) + "%", padL + plotW + 4, sy(mp));
  }
}
```

(i) Replace the body of `renderStatic` after the `err.style.display = "none";` line with:

```js
  renderHeader(s);
  updateRate(s);
  renderKpiTiles(buildKpiTiles(s));
  renderFlowmap(s);
  renderDetail(s);
  drawLoss();
  drawResources();
  renderCheckpoints(s);
  renderEvents(s);
  renderLogs(s);
```

(j) Update `render()` to inject the component before token substitution:

```python
def render() -> str:
    """Return the live run page as one self-contained HTML string.

    The flow-map component (viewer/flowmap.py) is spliced in first, THEN the
    `%%token%%` placeholders are substituted from `TOKENS` — so the component's
    CSS resolves against the same single-source-of-truth palette. No other
    templating — the page is otherwise a static document.
    """
    html = _TEMPLATE.replace("%%FLOWMAP_CSS%%", FLOWMAP_CSS).replace("%%FLOWMAP_JS%%", FLOWMAP_JS)
    subs = {f"%%{name}%%": value for name, value in TOKENS.items()}
    subs["%%tokens_json%%"] = json.dumps({k: TOKENS[k] for k in _JS_TOKEN_KEYS})
    for placeholder, value in subs.items():
        html = html.replace(placeholder, value)
    return html
```

- [ ] **Step 4: Run the viewer tests, then the whole suite**

Run: `.venv/bin/pytest tests/test_viewer_page.py tests/test_viewer_flowmap.py tests/test_viewer_server.py tests/test_viewer_state.py -v`
Expected: all pass.

Run: `.venv/bin/pytest`
Expected: no failures (count grows from 237 passed by the new tests; 1 skipped, 9 deselected unchanged).

- [ ] **Step 5: Eyeball the real page**

```bash
.venv/bin/python examples/demo_pytorch_ddp.py 2>/dev/null || true
```

If the demo needs arguments or a different invocation, check `examples/README.md`. Confirm in the printed `viewer: http://127.0.0.1:...` page (or ask the human partner to): KPI tiles fill in, the flow map shows machine → torchrun worker → 2 rank nodes with PIDs, clicking a rank opens the detail panel, the resources chart draws (or shows the psutil hint honestly).

- [ ] **Step 6: Commit**

```bash
git add flashruntime/viewer/page.py tests/test_viewer_page.py
git commit -m "feat(viewer): layout-A run page — KPI strip, process flow map, detail panel, resource chart"
```

---

### Task 7: Docs + status log

**Files:**
- Modify: `CLAUDE.md` (the "Live run viewer" bullet in *Current state*)
- Modify: `/Users/phongcao/Work/Zolli-Labs/PROGRESS.md` (dated work-log entry per its LOGGING PROTOCOL)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update `CLAUDE.md`**

In the `## Current state` bullet that begins `- **Live run viewer** (`viewer/`, July 2026):`, extend the parenthetical describing the page: replace `(topology / loss / verified checkpoints / recovery feed)` with `(KPI dashboard strip, machine → worker → rank process flow map with click-in detail panel, loss + resource charts, verified checkpoints, recovery feed; telemetry from `monitor/` ResourceSampler — optional psutil via the `[monitor]` extra — and per-rank heartbeat files from `flashruntime.torch`)`.

Also update the test-count line(s) `**237 passed, 1 skipped**` to the actual new totals from Task 6 Step 4's full run (both occurrences: *Current state* and *Dev workflow*).

- [ ] **Step 2: Update the workspace `PROGRESS.md`**

Read the LOGGING PROTOCOL section at the top of `/Users/phongcao/Work/Zolli-Labs/PROGRESS.md` first, then append a dated (2026-07-23) work-log entry in its established format summarizing: run-monitor flow map + KPI dashboard shipped in the viewer (spec + plan paths), new `flashruntime/monitor/` sampler with `[monitor]` extra, per-rank heartbeats, and the new test totals.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: run-monitor flow map + dashboard in Current state"
cd /Users/phongcao/Work/Zolli-Labs && git add PROGRESS.md && git commit -m "log: flashruntime run-monitor flow map + dashboard (2026-07-23)" 2>/dev/null; cd /Users/phongcao/Work/Zolli-Labs/flashruntime
```

(If the workspace root is not itself a git repo, skip its commit — only save the file.)
