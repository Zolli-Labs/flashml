# Resilience Showcase Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Six new measured benchmark scenarios (fault_recovery_matrix, checkpoint_integrity, crash_storm, submit_latency, fanout_throughput, lease_recovery_latency) + shared fault-injection helpers, a `bench_stress` tier, a docs Resilience section, and a refreshed measured baseline.

**Architecture:** Extends `benchmarks/` exactly as built in the deploy-ready plan: one file per scenario in `benchmarks/scenarios/`, registry dict entry, `bench_v1` rows, docs rendered from committed baseline JSON only. Spec: `docs/superpowers/specs/2026-07-23-resilience-showcase-design.md`.

**Tech Stack:** Python ≥3.10; scenario code uses only stdlib + the installed flashruntime + torch (already in venv) + fastapi/uvicorn for S6 (service extra, installed; graceful skip if absent).

## Global Constraints

- **Honesty (spec §5b, user-mandated):** measured only; correctness counted not asserted; comparator failures reported as observed; modelled baselines labeled "modelled"; unfavorable results shipped with framing. Fabrication is a review failure.
- **Readability (§2b):** scenario files ≤120 lines, hypothesis at top, measurement method auditable from the file alone; `benchmarks/faults.py` helpers have one clear job each with a why-docstring.
- **TDD is mandatory per task (user-mandated this run):** failing test first for every helper and every scenario's smoke variant; RED evidence (command + failing output) and GREEN in each report.
- **Markers:** `bench_smoke` exists (deselected by default). This plan adds `bench_stress` — register in pyproject markers AND add to `addopts` deselection. Every scenario gets a smoke test; long loops (S2 full, anything >60 s) get stress tests.
- **Subprocess PATH discipline:** every scenario spawning children uses the existing `benchmarks` helpers (`bench_env()` / `ensure_venv_on_path()`) — never bare `os.environ`. (Recurring bug; reviewers check.)
- **Scenario module protocol (existing, do not drift):** module attrs `name: str`, `hypothesis: str`, `run(repeats: int) -> ResultRow` (see `benchmarks/schema.py` for `ResultRow`: scenario/unit/median/p10/p90/repeats/comparators/notes). Registry: one import + one entry in `benchmarks/registry.py` `SCENARIOS`/`_MODULES`. Read `benchmarks/scenarios/recovery_economics.py` first as the house pattern (it already does crash-injection + modelled-baseline labeling).
- **Additive `section` field:** rows gain optional `section: str = "performance"` (ResultRow, additive with default — old JSONs stay valid); the docs hook groups tables by section; new scenarios set `section="resilience"` where the spec says so.
- Suite baseline at plan start: **237 passed + 1 skipped + 9 deselected** (`PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest`). Full suite green before every commit. Branch `local-milestone-2026-07`; do not push.

## File Map (delta)

| Path | Status | Task |
|---|---|---|
| `benchmarks/faults.py` | create | T1 |
| `benchmarks/schema.py` (+`section`), `benchmarks/report.py` (section grouping), `scripts/build_docs.py` hook (grouped render) | modify | T1 (schema) / T6 (render) |
| `benchmarks/scenarios/fault_recovery_matrix.py` | create | T1 |
| `benchmarks/scenarios/checkpoint_integrity.py` + `bench_stress` marker in `pyproject.toml` | create/modify | T2 |
| `benchmarks/scenarios/crash_storm.py` | create | T3 |
| `benchmarks/scenarios/submit_latency.py`, `benchmarks/scenarios/fanout_throughput.py` | create | T4 |
| `benchmarks/scenarios/lease_recovery_latency.py` | create | T5 |
| `benchmarks/registry.py` | modify (one line per scenario, in its task) | T1–T5 |
| `tests/test_benchmarks.py` (extend per task) | modify | T1–T5 |
| `docs/site/benchmarks.md` (Resilience section), refreshed `benchmarks/results/baseline-*.json`, `README.md` teaser line, `.github/workflows/ci.yml` (smoke picks up new scenarios automatically — verify only), workspace `../PROGRESS.md` | modify | T6 |

Sequencing: T1 first (faults.py + the `section` schema field are shared infrastructure; fault_recovery_matrix proves them). T2–T5 independent after T1 (run sequentially per SDD rules). T6 last (measure baseline over all six, render, README, log).

---

### Task 1: `faults.py` + `section` field + S1 `fault_recovery_matrix`

**Files:** Create `benchmarks/faults.py`, `benchmarks/scenarios/fault_recovery_matrix.py`; modify `benchmarks/schema.py` (additive `section: str = "performance"`), `benchmarks/registry.py`, `tests/test_benchmarks.py`.

**Interfaces produced (T2/T3 consume — exact):**
- `faults.write_crashy_trainer(dir: Path, *, steps: int, checkpoint_every: int, crash: str | None) -> Path` — writes a self-contained stdlib+ft training script; `crash` ∈ {None, "import_error" (top-level `import definitely_not_a_module`), "systemexit_mid" (SystemExit(3) at midpoint, fresh-run-only marker via FLASHML_CKPT_DIR progress file), "hang_after_step" (sleeps forever after step k — enables external SIGKILL)}. Returns the script path.
- `faults.kill_child(run, *, sig=signal.SIGKILL, when: Callable[[], bool], timeout_s: float) -> bool` — polls a `wait=False` Run's newest attempt pid (from `run.attempts`) and signals it when `when()` first returns True; returns whether it fired.
- `faults.corrupt_newest_part(ckpt_root: Path) -> Path | None` — flips bytes mid-file in the newest manifest's largest part; returns the corrupted path.
- S1 emits ONE ResultRow: `scenario="fault_recovery_matrix"`, `section="resilience"`, `unit="correct/5"`, `median=<counted correct handling>`, `comparators={"manual_interventions_flash": 0.0-as-counted, "manual_interventions_torchrun_modelled": 5.0, "mean_recovery_s": ..., "mean_steps_preserved": ...}`, per-case verdicts in `notes` (5 entries, e.g. `"(a) import_error → FAIL_JOB fast-stop ✓ (1 attempt, 0 restarts burned)"`). Correctness is COUNTED from run.json/state (attempts count, events, resumed_from), never asserted-then-caught.

Case mechanics (each a small helper in the scenario file, ≤120-line budget rules the file — factor shared bits into faults.py): (a) import_error script + max_restarts=3 → expect FAILED with 1 attempt; (b) systemexit_mid + max_restarts=1 → expect SUCCEEDED, resumed_from>0; (c) hang_after_step + kill_child(SIGKILL when a checkpoint manifest exists) + max_restarts=1 → SUCCEEDED via resume; (d) fast-checkpointing script, kill_child timed inside the write window (poll for a `step-*` dir existing without `manifest.json` yet — that IS the write window) → resumed step must be a hash-verified earlier manifest; (e) run to completion once, `corrupt_newest_part`, rerun same output_dir → resumed_from == an EARLIER valid step (never the corrupted one). Torchrun comparator: same (b) injection under bare torchrun argv → process dies, no retry; recovery modelled `t_crash + t_full` (labeled modelled, house convention).

- [ ] **Step 1 (TDD):** failing unit tests for the three faults helpers (script materializes+compiles; kill_child fires on a live sleep-subprocess wrapped in a fake run-like object — test the polling seam with a stub exposing `.attempts`; corrupt_newest_part flips bytes and `latest_valid_manifest` then skips that manifest) + a `bench_smoke` test asserting `fault_recovery_matrix.run(repeats=1)` returns a ResultRow with `section=="resilience"` and 5 notes. Run → RED (paste).
- [ ] **Step 2:** implement `faults.py`, the `section` field (additive default — verify old baseline JSON still validates via existing tests), the scenario, registry line.
- [ ] **Step 3:** GREEN on the new tests; run the scenario for real once (`python -m benchmarks run --scenario fault_recovery_matrix --repeats 1 --smoke`) and paste its notes table in the report — sanity: 5/5 expected; if any case measures WRONG-handling, report it honestly and stop (that's a product bug find, not a test bug).
- [ ] **Step 4:** full suite green; commit `feat(benchmarks): fault-injection helpers + fault_recovery_matrix (S1)`.

### Task 2: S2 `checkpoint_integrity` + `bench_stress` tier

**Files:** Create `benchmarks/scenarios/checkpoint_integrity.py`; modify `pyproject.toml` (marker + addopts), `benchmarks/registry.py`, `tests/test_benchmarks.py`.

Row: `section="resilience"`, `unit="integrity_rate"`, `median` = flash integrity rate over the loop (target 1.0, MEASURED), `comparators={"naive_torch_save_failure_rate": <observed>, "iterations": N, "torn_writes_hit": <count of kills that landed inside a write window>}`, notes explain the kill-window technique + the comparator's observed failure modes (verbatim exception classes seen). Loop N: 20 full (stress), 3 smoke. Kill technique per iteration: fast checkpointer (checkpoint_every=1, tiny model), `kill_child` when a fresh `step-*` dir appears without its manifest (the write window from T1(d)) — iterations whose kill misses the window are counted separately (`torn_writes_hit`), not silently blended. Naive comparator: identical trainer shape but `torch.save(state, "latest.pt")` each step; after kill, `torch.load("latest.pt")` — record success/exception per iteration.

- [ ] TDD: failing tests — marker registered + deselected (assert via pytest config introspection like the existing bench_smoke test), smoke variant (N=3) returns a row with `median<=1.0` and `iterations==3`, stress test exists under `@pytest.mark.bench_stress` running N=20. RED → implement → GREEN → run smoke for real, paste → full suite → commit `feat(benchmarks): checkpoint_integrity chaos scenario + bench_stress tier (S2)`.

### Task 3: S3 `crash_storm`

**Files:** Create `benchmarks/scenarios/crash_storm.py`; modify registry, tests.

Row: `section="resilience"`, `unit="completed/16"`, `median`=completions counted from run.trials, `comparators={"goodput_fraction": useful_steps/total_steps_executed, "wallclock_penalty_fraction": (t_storm-t_clean)/t_clean, "manual_interventions": 0.0-as-counted, "crashed_first_attempt": 8.0-as-counted}`. Mechanics: 16-trial fan-out via task_params; trials with even index crash on fresh attempt (systemexit_mid script from faults.py — the fresh-only marker makes retries succeed); `max_restarts=1`; a second identical crash-free sweep gives t_clean. Steps counted from each trial's metrics (steps + resumed_from ⇒ executed = steps + resumed_from for crashed trials' two attempts — derive carefully and show the arithmetic in the docstring). Smoke: 4 trials.

- [ ] TDD: failing smoke test (4-trial variant: completed==4, goodput ≤1, crashed_first_attempt==2) → RED → implement → GREEN → real smoke run pasted → full suite → commit `feat(benchmarks): crash_storm goodput scenario (S3)`.

### Task 4: S4 `submit_latency` + S5 `fanout_throughput`

**Files:** Create both scenario files; modify registry, tests.

- S4 row: `section="performance"`, `unit="s (p50 cold)"`, `median`=cold p50; `comparators={"cold_p95":..., "warm_p50":..., "warm_p95":..., "phase_launch_s":..., "phase_child_s":...}`. Child stamps `first_step_ts` (time.time()) into metrics.json; parent computes submit-call→first-step deltas. Cold = fresh output_dir each run; warm = repeated into same process (2nd..Nth). N=20 full / 3 smoke. Note: constant overhead framing.
- S5 row: `section="performance"`, `unit="tasks/min"`, `median`=fan-out throughput at ~0.5 s tasks; `comparators={"sequential_tasks_min":..., "overhead_frac_0５s":..., "overhead_frac_2s":..., "overhead_frac_5s":...}` — amortization curve at three task durations (task sleeps to target duration), overhead_frac = (flash_wall - ideal_serial_work)/flash_wall. 16 tasks full / 4 smoke. Honest note that local fan-out is sequential by design (house convention from hpo_sweep).
- [ ] TDD: failing smoke tests for both (row shape + comparator keys present + sane bounds) → RED → implement → GREEN → real smoke runs pasted → full suite → commit `feat(benchmarks): submit_latency + fanout_throughput scenarios (S4,S5)`.

### Task 5: S6 `lease_recovery_latency`

**Files:** Create `benchmarks/scenarios/lease_recovery_latency.py`; modify registry, tests.

Row: `section="resilience"`, `unit="s (MTTR)"`, `median`=dead-worker requeue→recompletion; `comparators={"claim_rt_p50_ms":..., "claim_rt_p95_ms":..., "mttd_s": <last-heartbeat→requeue observed>, "mttd_bound_s": lease_seconds+2.0, "roundtrips": N}`. Mechanics: boot the REAL app via `uvicorn` **subprocess** on port 0-resolved loopback (find a free port; env: FLASHML_ENABLE_KUBERAY=0, tmp ledger + artifacts dirs; readiness-poll /healthz; ALWAYS terminate in finally). Real HTTP via `httpx`. Register workers A+B (`/v1alpha1/nodes/register`); submit a small `hyperparameter_search` job (`submit-spec`-shaped POST /v1alpha1/jobs with `trials` params and short `lease_seconds` e.g. 3.0); worker A claims, heartbeats twice, then goes silent; measure last-heartbeat→(task claimable by B) via polling claims; B "completes" by writing the expected artifact via PUT /v1alpha1/artifacts/{commit_key} then POST complete with its sha256 (read `service/modea.py` + `tests/test_service_modea.py` for the exact contract — the commit is sha256-validated). Round-trip timings over N≥20 claim/heartbeat cycles. Docstring: states real-sockets-vs-TestClient choice + that this is the first measured slice of the Stage-8 metrics debt. Graceful skip row (notes) when fastapi/uvicorn missing. Smoke: N=5 round-trips, one death cycle.

- [ ] TDD: failing smoke test (boots server, row has mttd_s ≤ mttd_bound_s + slack, claim_rt keys present; marks `bench_smoke`; skips without uvicorn) → RED → implement → GREEN → real smoke run pasted (include the measured MTTD vs bound) → full suite → commit `feat(benchmarks): lease_recovery_latency — measured Mode A MTTD/MTTR (S6)`.

### Task 6: Render, measure, publish

**Files:** Modify `benchmarks/report.py` + `scripts/build_docs.py` (group rows by `section` → "Performance" + "Resilience" tables; existing single-table render becomes the grouped render — keep the marker contract), `docs/site/benchmarks.md` (Resilience intro prose + methodology), refreshed `benchmarks/results/baseline-<host>.json` (RUN `python -m benchmarks run --all --repeats 5` — stress loops full; expect ~15-30 min wall; paste the table), `README.md` (ONE resilience teaser line written FROM the measured JSON — e.g. the 5/5 matrix + integrity rate; never write it before measuring), `tests/test_benchmarks.py` (grouped-render test: fixture JSON with two sections renders two tables), workspace `../PROGRESS.md` entry (its protocol; separate repo commit).

- [ ] TDD: failing grouped-render test → RED → implement render → GREEN → measure the full baseline → commit JSON + docs rebuild (`--check`) → README line → full suite (`237+1+9` + new deselected stress/smoke additions accounted — record exact) → audit script still CLEAN → both repos committed: `feat(benchmarks): resilience section — measured baseline, grouped docs render` + PROGRESS entry.

---

## Self-Review Notes

- Spec coverage: S1→T1, S2→T2, S3→T3, S4/S5→T4, S6→T5, docs/baseline/README→T6, faults+section→T1, stress tier→T2, acceptance 1-5 → T1-T6 collectively (4's spot-checks are review lens items; reviewers get them).
- Type consistency: `section` lands in schema at T1 with a default so T1-T5 rows all carry it before T6 renders it; faults.py signatures stated once in T1 and consumed by T2/T3 as written.
- The S6 commit contract (PUT artifact then complete with sha256) is deliberately delegated to reading `service/modea.py` — it is the live wire contract and the implementer must code against the source, not this plan.
