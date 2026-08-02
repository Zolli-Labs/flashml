# Design: Resilience & performance showcase suite

Date: 2026-07-23 · Status: approved (user: "go with all six") · Extends:
`benchmarks/` (spec `2026-07-22`, §5b honesty rules — all inherited verbatim)

## 1. Goal

Six new measured scenarios that showcase what flashruntime does *under
fire* — error handling, recovery correctness, latency distributions,
throughput — each with an honest comparator and a specific, reproducible
claim. They double as the functional stress suite ("later testing the full
functionalities"): every scenario exercises real product paths (SDK submit,
recovery loop, manifest contract, lease HTTP machinery), not mocks.

## 2. Inherited rules (unchanged, binding)

Spec §5b honesty (measured only; skips/unfavorables documented, never
deleted; medians of N≥5 where timing, exact counts where correctness);
§2b readability (one scenario file ≤120 lines, hypothesis at top,
auditable method in the docstring, registry line to add); `bench_v1` row
schema; docs render only from committed baseline JSON; `bench_smoke` CI
tier. TDD for every task: failing test first, then implementation.

## 3. The six scenarios

### S1 `fault_recovery_matrix` — error-handling headline
Inject 5 failure types into real `flash.submit(..., max_restarts=1)` runs:
(a) deterministic bug (ImportError at startup), (b) transient crash
(SystemExit mid-training), (c) SIGKILL'd worker mid-training, (d) kill -9
DURING a checkpoint write, (e) corrupted newest checkpoint on disk before
resume. Measure per type: classification correctness (fail-fast for (a),
retry for (b)-(d), valid-manifest fallback for (e)), time-to-recovery
(seconds), steps preserved. Comparator: bare torchrun — the same
injections end the run; recovery = manual restart from 0 (modelled as
t_crash + t_full, labeled as modelled, same convention as
recovery_economics). Headline row: `correct_handling` = N/5 (must be
measured, target 5/5) + `manual_interventions` 0 vs 5.

### S2 `checkpoint_integrity` — never resume from garbage
Loop ≥20 iterations (stress tier): run a fast-checkpointing trainer, kill
-9 at a random offset inside the checkpoint-write window, then resume and
record: resumed step is a hash-verified manifest step (never a torn one),
torn/partial part files quarantined not loaded. `integrity_rate` must be
measured over all iterations (claim: 100%). Comparator: naive
`torch.save`→newest-file→`torch.load` under identical kills — report its
real observed failure modes/rate (load error or partial state), whatever
they are.

### S3 `crash_storm` — goodput under chaos
16-trial fan-out; 50% of trials crash on first attempt (deterministic
selection by trial index; crash-on-fresh-only marker like kill_at_step).
Measure: completion 16/16 via retries, goodput = useful steps /
total-steps-executed, wall-clock penalty vs a crash-free identical sweep,
manual interventions = 0.

### S4 `submit_latency` — distributions, not medians
Cold + warm submit→first-training-step latency, ≥20 runs, report
p50/p95 and a phase split (compile+launch vs child import+first step —
child stamps a first-step timestamp into metrics; parent computes deltas).
Honest framing note: constant overhead, invisible at training scale.

### S5 `fanout_throughput` — scheduling speed
16 short trials: tasks/min via SDK fan-out vs a plain sequential
subprocess loop; plus the amortization curve — per-task overhead at task
durations {~0.5 s, ~2 s, ~5 s} (overhead fraction shrinks as tasks grow);
curve points land in `comparators`/`notes`.

### S6 `lease_recovery_latency` — Mode A MTTD/MTTR, measured
Boot the REAL coordinator app locally (uvicorn subprocess on a loopback
port — real HTTP sockets, not TestClient; the scenario notes state this
choice and why: latency claims require real round-trips). Two synthetic
workers register; submit a lease job; worker A claims and dies (stops
heartbeating); measure: claim/heartbeat/complete round-trip p50/p95 under
normal operation, and dead-worker MTTD (last-heartbeat→lease-expiry
requeue, bounded by lease_seconds + 2 s sweeper — measure actual, report
the config bound beside it) and MTTR (requeue→worker B completes). This
is the first measured slice of the Stage-8 metrics debt (AGENTS Missing
list) — note that in the scenario docstring; the coordinator-side ledger
metrics remain future work.

## 4. Mechanics

- New shared fault-injection helpers in `benchmarks/faults.py` (kill-child
  -by-pid via Run attempts' `pid`, corrupt-newest-part, crash-marker
  scripts) — one small module, reused by S1/S2/S3, unit-TESTED (TDD).
- New `bench_stress` pytest marker (registered + deselected by default,
  like bench_smoke): S2's ≥20-kill loop and any scenario leg >60 s runs
  under stress tier; every scenario still has a bench_smoke variant
  (1 repeat / reduced loop, assert-it-runs).
- CI: bench-smoke job unchanged (new scenarios' smoke variants join it);
  stress tier is local/on-demand (`pytest -m bench_stress` documented).
- S6 needs the `[service]` extra — scenario skips gracefully with a note
  when fastapi/uvicorn are absent (they are installed in this venv).
- Baseline: after all six land, run
  `python -m benchmarks run --all --repeats 5` (stress loops at their
  full N) on this Mac and commit the refreshed
  `benchmarks/results/baseline-<host>.json`.

## 5. Docs & claims

- `docs/site/benchmarks.md` gains a **Resilience** section: the S1 matrix
  table, S2 integrity rate vs comparator, S3 goodput, S6 MTTD/MTTR —
  rendered from the baseline JSON only (extend the existing BENCH_TABLES
  hook if section grouping is needed; a `section` field on rows is the
  additive path).
- README benchmarks teaser gains ONE resilience line (the S1 matrix
  result) once measured — exact wording written from the JSON, never
  before it.

## 6. Acceptance criteria

1. Six scenario files + faults.py, each ≤120 lines, registry-listed,
   hypothesis+method auditable, TDD evidence per task.
2. `python -m benchmarks run --all` executes all scenarios (S2/S6 full
   loops under stress flags/tier as designed) and the refreshed baseline
   JSON is committed; docs page renders the Resilience section from it;
   README teaser updated from measured numbers.
3. Smoke tier: `pytest -m bench_smoke` covers all six (fast); stress
   tier: `pytest -m bench_stress` runs S2's full loop; default suite
   count unchanged except new deselected markers.
4. Honesty spot-checkable: S1 correctness is counted not asserted; S2
   comparator's real failure rate reported; S6 states the sockets-vs-
   TestClient choice; modelled baselines labeled modelled.
5. Full suite green; docs --check green; audit clean.

## 7. Out of scope

GPU variants (ride a future RunPod run); coordinator-side ledger
MTTD/MTTR events (S6 measures externally; Stage-8 proper stays on the
missing list); any new runtime dependency.
