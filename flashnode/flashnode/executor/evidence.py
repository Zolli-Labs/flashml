"""What this agent can honestly say about a run it just finished.

Fills `flashruntime.protocol.v1alpha1.ExecutionEvidence`, the optional block
the executor attaches to `complete`. Three sources, all best-effort:

- **wall clock** — measured by the loop around the runner, always available.
- **CPU / GPU utilisation** — sampled on a background thread while the task
  runs (`ResourceSampler`), averaged at the end.
- **image digest** — asked of the local docker daemon after the container
  exits (`image_digest`).

THE ONE RULE. A value that could not be measured is reported as `None` (or
`""` for the digest), never as `0`. The temptation runs entirely one way:
`sum([]) / max(len([]), 1)` is 0.0, `psutil.cpu_percent()`'s first call in a
process is 0.0, an unparsed `docker` error string is truthy. Each of those is
a fabricated measurement that a verifier downstream cannot tell from a real
one — and "0% CPU on a completed task" is not a missing field, it is an
accusation. Every path below that could produce a zero by accident is written
so it produces absence instead, and each has a test.

NO NEW DEPENDENCIES. `psutil` is already an agent dependency (inventory/);
GPU utilisation goes through the same `nvidia-smi` subprocess as the
registration probe. This code runs on strangers' machines and every
dependency is attack surface (AGENTS.md).

Every probe takes its subprocess/sampler as a PARAMETER, resolved at call
time, so the whole test suite runs with no NVIDIA driver and no docker
daemon — the shape `inventory/gpu.py` and `doctor.py` established.
"""

from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from typing import Callable

from flashnode.inventory.gpu import probe_gpu_utilisation

__all__ = ["ResourceSampler", "default_cpu_probe", "image_digest"]

#: A probe answers "what is it now?" — a percentage, or None for "could not
#: read it". Never raises for the caller; the sampler defends anyway.
Probe = Callable[[], "float | None"]

CommandRunner = Callable[..., subprocess.CompletedProcess]

#: `docker image inspect` reads metadata the daemon already holds. Short on
#: purpose: this runs on the commit path, and a wedged daemon must cost the
#: field, not the task's result.
DIGEST_TIMEOUT_S = 10

#: What a content-addressable image identity looks like. Anything else the
#: daemon prints — a prose error, `<no value>` from a template miss, an empty
#: line — is NOT a digest and must not be forwarded as one.
_DIGEST_RE = re.compile(r"^[a-z0-9]+:[0-9a-f]{32,}$")


def _text(raw: object) -> str:
    """Subprocess output as a string, or `""` for anything that is not one.

    Total on purpose, and not quite the shape `inventory/gpu.py` uses: the
    runner is a parameter, so `stdout` is whatever a caller's stub put there.
    A duck-typed object with a `.decode` that returns another object would
    otherwise walk a non-string all the way to the regex below and raise
    TypeError from inside a best-effort probe.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode(errors="replace")
    return ""


def _reading(value: object) -> float | None:
    """One probe answer as a number the mean can survive, or None.

    NaN is the reason this exists: a single NaN sample turns a whole run's
    mean into NaN, which serialises to something no verifier can read. `bool`
    is excluded explicitly — it is a subclass of `int`, so `True` would
    otherwise average in as 1.0%.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    """The average, or None for no samples at all.

    NOT 0.0. This one line is the whole honesty contract of this module: an
    empty list means nothing was measured, and a verifier reading 0.0 would
    see a machine that did nothing on a task it completed.
    """
    if not values:
        return None
    return sum(values) / len(values)


def default_cpu_probe() -> Probe:
    """A host-wide CPU probe, with psutil's meaningless first reading burned.

    `psutil.cpu_percent(interval=None)` reports usage since the LAST call, so
    its first answer in a process is 0.0 by construction — a fabricated zero
    that looks exactly like an idle machine. That call happens here, at
    construction, before the task starts; every reading the sampler takes
    afterwards covers a real window.

    Host-wide, not task-scoped: a volunteer's machine has other things
    running on it. That makes a HIGH reading weak evidence and a LOW one the
    interesting direction — a completed training task on a machine that never
    got busy. `ExecutionEvidence` says so in the field's own docstring.

    No psutil (a stripped install) ⇒ a probe that always says None.
    """
    try:
        import psutil
    except Exception:  # noqa: BLE001 - an optional field must never break the run
        return lambda: None

    try:
        psutil.cpu_percent(interval=None)  # prime; the answer is meaningless
    except Exception:  # noqa: BLE001
        return lambda: None

    def probe() -> float | None:
        return psutil.cpu_percent(interval=None)

    return probe


class ResourceSampler(threading.Thread):
    """Averages utilisation across one task's run.

    Daemon thread, like `_AttemptHeartbeat` and `_CheckpointRelay`: telemetry
    must never be the reason an agent will not exit. Started before the
    runner and stopped after it; `stop()` returns
    `(cpu_percent_mean, gpu_util_percent_mean)`, either of which may be None.

    Two cadences, because the two probes cost wildly different amounts.
    Reading CPU is a memory access; reading GPU utilisation spawns
    `nvidia-smi`. Sampling both at one interval means either a useless CPU
    trace or thousands of subprocesses over a long job on a machine somebody
    lent us.
    """

    def __init__(
        self,
        cpu_probe: Probe | None = None,
        gpu_probe: Probe | None = None,
        cpu_interval_s: float = 1.0,
        gpu_interval_s: float = 5.0,
    ):
        super().__init__(daemon=True)
        # Built now, not on the thread: `default_cpu_probe` burns psutil's
        # priming reading, and that has to happen before the run starts.
        self._cpu_probe = cpu_probe if cpu_probe is not None else default_cpu_probe()
        self._gpu_probe = gpu_probe if gpu_probe is not None else probe_gpu_utilisation
        self._cpu_interval_s = cpu_interval_s
        self._gpu_interval_s = gpu_interval_s
        self._halt = threading.Event()
        self._cpu: list[float] = []
        self._gpu: list[float] = []
        #: Set False the first time the GPU probe answers "nothing here", so
        #: a host with no driver — the overwhelmingly common case — is asked
        #: once instead of every few seconds for the length of the job.
        self._gpu_worth_asking = True

    def run(self) -> None:
        next_gpu = 0.0  # sample the GPU on the first tick
        while not self._halt.wait(self._cpu_interval_s):
            self._sample_cpu()
            now = time.monotonic()
            if now >= next_gpu:
                self._sample_gpu()
                next_gpu = now + self._gpu_interval_s

    def stop(self) -> tuple[float | None, float | None]:
        """Halt, take one last reading, and report the means.

        The final reading is not a courtesy: a task shorter than one tick
        would otherwise produce zero samples, and "no samples" on every fast
        task is how a fabricated default gets added later. It measures the
        window that genuinely just elapsed.
        """
        self._halt.set()
        if self.is_alive():
            self.join(timeout=self._cpu_interval_s + 5.0)
        self._sample_cpu()
        if not self._gpu:  # only to rescue a run too short for a single tick
            self._sample_gpu()
        return _mean(self._cpu), _mean(self._gpu)

    # -- sampling ------------------------------------------------------------

    def _ask(self, probe: Probe) -> float | None:
        """One probe call that cannot take the task down with it."""
        try:
            return _reading(probe())
        except Exception:  # noqa: BLE001 - a field is never worth a run
            return None

    def _sample_cpu(self) -> None:
        value = self._ask(self._cpu_probe)
        if value is not None:
            self._cpu.append(value)

    def _sample_gpu(self) -> None:
        if not self._gpu_worth_asking:
            return
        value = self._ask(self._gpu_probe)
        if value is None:
            # Only give up when nothing has EVER been read. Once a reading has
            # succeeded the driver demonstrably exists, and one blip must not
            # cost the rest of the run's GPU evidence.
            if not self._gpu:
                self._gpu_worth_asking = False
            return
        self._gpu.append(value)


def image_digest(reference: str, run: CommandRunner | None = None) -> str:
    """Which image bytes actually executed, as this host resolved them.

    `docker image inspect --format {{.Id}}` — the sha256 of the image config,
    i.e. the local content-addressable identity of the thing that ran.

    NOT `RepoDigests[0]`, the registry manifest digest, though that reads like
    the more obvious answer. A locally built or `docker load`-ed image has no
    `RepoDigests` entry at all, so that field is empty on exactly the hosts
    most worth checking. `.Id` is always present and is what two honest nodes
    running the same pinned tag on the same architecture agree on, which is
    the comparison the verifier actually wants to make.

    `""` means unknown: no daemon, an image it will not describe, or output
    that is not a digest. Never the payload's image reference — echoing back
    what the coordinator already told us is not evidence, it is a rumour with
    a hash-shaped name.
    """
    if not reference or not isinstance(reference, str):
        return ""
    run = run or subprocess.run
    try:
        proc = run(
            ["docker", "image", "inspect", reference, "--format", "{{.Id}}"],
            capture_output=True,
            timeout=DIGEST_TIMEOUT_S,
            check=False,
        )
        if getattr(proc, "returncode", 1) != 0:
            return ""
        value = _text(getattr(proc, "stdout", None)).strip()
    except Exception:  # noqa: BLE001 - best effort, like every field here
        return ""
    return value if _DIGEST_RE.match(value) else ""
