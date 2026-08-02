# Benchmarks

Every number on this page is **measured**, never asserted. The tables below are
rendered at docs-build time straight from the committed baseline JSON
(`benchmarks/results/baseline-<host>.json`) — the docs cannot show a figure the
suite did not produce. Each scenario states its hypothesis and its measurement
method in its own source file (`benchmarks/scenarios/`), so the methodology is
auditable from the code alone, and every caveat and skip is printed verbatim in
the notes under each table.

Where a comparator (ray, accelerate) is not installed on the baseline machine,
its row says so and its setup code is *counted*, not run, from the cited
fixtures in `benchmarks/scenarios/snippets/` — an honest line count, never a
fabricated timing. Some figures here are small, zero, or negative: that is the
suite working as intended. On a tiny CPU model, process-startup dominates
wall-clock and a checkpoint write or a 40-step recompute falls below the
run-to-run noise floor — the notes say so, and the size-independent guarantees
(e.g. `steps_not_recomputed`) are reported alongside. The value shows up at real
model scale; the honesty shows up here.

The suite is split into two sections, each rendered as its own table below.
**Performance** measures the overheads a user pays for adopting flashruntime —
launch overhead, per-checkpoint cost, submit latency, fan-out throughput,
adoption line count — as wall-clock medians against a bare-`torchrun` or plain
sequential baseline. **Resilience** measures the fault-tolerance guarantees
themselves, under real failure injection rather than modelling:

- **Correct classification** — `fault_recovery_matrix` runs five distinct fault
  types (import error, mid-run `SystemExit`, a worker killed mid-run by an
  external `SIGKILL`, a `SIGKILL` inside the checkpoint-write window, and a corrupted
  newest part) and *counts* — from terminal run state, never asserts — how many
  the typed recovery taxonomy routes to the right action.
- **Integrity under `kill -9`** — `checkpoint_integrity` fires repeated
  `SIGKILL`s that land inside the checkpoint write window; the parts-first /
  manifest-last commit means a torn write is never a *valid* checkpoint, so
  resume falls back to the last verified step. The naive `torch.save` comparator
  is killed in the same window and its corruption rate is reported beside ours.
- **Goodput under a storm** — `crash_storm` fans out a batch of trials with half
  of them armed to crash on their first attempt, then measures the fraction of
  useful work retained (and the tail actually recomputed) once every crashed
  trial auto-resumes from its checkpoint, with zero manual intervention.
- **Measured MTTD/MTTR** — `lease_recovery_latency` boots the real FastAPI
  coordinator over real loopback sockets, kills a worker mid-lease, and times
  detection (lease-expiry sweep) and recovery (re-claim → artifact commit) end
  to end, alongside steady-state claim/heartbeat round-trip latency.

Each resilience number is COUNTED or timed from observable run state, never
baked into an assertion. The failure injection lives in `benchmarks/faults.py`
(crashy-trainer generation, `kill -9` timing, part corruption), each scenario's
measurement method is stated in its own source file under
`benchmarks/scenarios/`, and the long chaos loops carry a `bench_stress` marker
so `pytest -m bench_stress` re-runs them.

Reproduce the whole baseline yourself:

```bash
python -m benchmarks run --all --repeats 5
```

Run a single scenario, or a fast labelled smoke:

```bash
python -m benchmarks run --scenario recovery_economics
python -m benchmarks run --all --smoke
```

<!-- BENCH_TABLES -->
