# Design: Deploy-ready flashruntime — gaps, live viewer, docs

Date: 2026-07-22 · Status: approved (design) · Builds on:
`2026-07-21-command-workloads-design.md` (shipped)

## 1. Goal

Make flashruntime publishable (`pip install flashruntime`) with every README
claim true, watchable (a live run viewer that opens on submit), and
learnable (PyTorch-style docs) — while cutting the codebase down to the one
story it now tells. Three workstreams, one release: **0.1.0**.

## 2. Non-goals (deferred to 0.2.0/0.3.0)

- Multi-node DDP rendezvous (nnodes>1 stays `NotImplementedError`).
- flashnode argv runner / sandbox tiers (cross-repo).
- `flash.run(StrategyPlan)` wiring; server-side manifest persistence;
  Stage-8 ledger metrics.
- A React/Node toolchain anywhere in this repo. The viewer and docs are
  no-build, self-contained HTML (the `service/dashboard.py` philosophy).
- Copying anything from flashml-cloud (private forever). Its viz *style*
  (dark surface, oklch accents, mono type, canvas drawing) is reimplemented
  from scratch.

## 2b. Code clarity principles (bind every workstream)

Stated by the user as a release requirement, not a preference:

- **Human-readable, reasoned code.** Every module/function exists for a
  reason a newcomer can read (docstrings explain *why*, code shows *how*).
  Prefer boring, explicit code over clever code. If a reviewer needs the
  author to explain it, it fails review.
- **Horizontal extensibility is a design test.** Adding support for
  another library must stay "one small adapter file in `integrations/`
  (+ optionally one recipe)" — never a core change. New workstream code
  must preserve this: the viewer reads a versioned `run.json` contract
  (any future launcher can write it); benchmarks add scenarios via a
  registry, not edits to a monolith.
- **Small files, one responsibility each.** The viewer page is the one
  sanctioned large file (self-contained HTML is the point); everything
  else stays modular.

## 3. Workstream A — close the gaps

### A1. Remove the prototype engine (approved by user)

Delete: `flashruntime/engine/`, `flashruntime/algorithms/`,
`flashruntime/adapters/`, `flashruntime/storage/`,
`examples/local_kmeans_and_linear_regression.py`,
`tests/test_local_provider.py`, the `[prototype]` extra, the lazy
`_PROTOTYPE_EXPORTS` block in `flashruntime/__init__.py`, and the
`docs/archive-prototype/` tree (historical; lives in git history).
Keep: `flashml_workloads/` (live Mode A task modules), `backends/`
(KubeRay), everything else. Update README/AGENTS.md accordingly; the
"built-in algorithms" section of the bring-your-code guide is rewritten to
point at `flashml_workloads/` + Mode A instead of `Cluster.train`.
Sweep for dangling references (docs, tests, `sync-docs` sources).

### A2. Release engineering

- `.github/workflows/ci.yml`: pytest matrix — Python 3.10/3.11/3.12/3.13 ×
  ubuntu-latest/macos-latest; core-import smoke (pydantic-only check) as its
  own job; e2e job installs `[sklearn,dev]` + CPU torch and runs the full
  suite including `test_examples_e2e.py`.
- `.github/workflows/release.yml`: on tag `v*` — build sdist+wheel
  (`python -m build`), publish via PyPI **trusted publishing** (OIDC, no
  token in repo), then deploy the docs site (C) to GitHub Pages.
- Secrets/history audit: `git log -p` scan for key patterns (incl.
  `rpa_`/RunPod, AWS, tokens), `.env` never tracked, no flashml-cloud
  references in public files. Findings block release.
- Community files: `CONTRIBUTING.md` (dev setup, test tiers, protocol
  versioning rules), `SECURITY.md`, `CHANGELOG.md` (0.1.0 entry),
  `flashruntime/py.typed` + `package_data`.
- `pyproject.toml`: project URLs (docs site), classifiers, keywords;
  version stays `0.1.0` until tag.

### A3. GPU wiring in `flashruntime.torch`

`prepare()` on CUDA machines: `local_rank = LOCAL_RANK`, `torch.cuda.
set_device(local_rank)`, move model to `cuda:{local_rank}` **before** DDP
wrap, `DistributedDataParallel(model, device_ids=[local_rank])`; single-
process CUDA (WORLD_SIZE==1, cuda available) also moves the model.
Checkpoint restore maps to the local device (`map_location`), save still
writes CPU tensors (topology-agnostic manifests, unchanged contract).
`checkpoint()` unwraps and `.cpu()`s state dicts before `torch.save`.
Unit tests run on CPU by factoring device selection into a pure helper
(`_resolve_device(world_size, cuda_available, local_rank)`) tested
directly; the CUDA path is exercised for real by A6. The guide's GPU
calibration sentence is updated to "validated on RunPod".

### A4. Automatic fault tolerance (`max_restarts`)

`flash.submit(workload, output_dir=None, max_restarts=0)`:
- After a FAILED coordinated/local launch, build `FailureSignals` from what
  the SDK can observe (exit code, log tail patterns, heartbeat n/a locally)
  → `recovery.classify()` → `recovery.decide(failure, mode="coordinated_
  training" | local)`.
- Action `FAIL_JOB` (deterministic app error) → stop immediately, surface
  the decision. Action `RESTART_GROUP`/`RETRY_TASK` → relaunch the same
  compiled spec; `ft.prepare` in the child restores from the newest valid
  manifest (existing behavior — this is why it works with zero user code).
  Bounded by `max_restarts`; each decision is recorded on the run
  (`run.events`: FAILURE_CLASSIFIED, RECOVERY_ACTION_SELECTED, restart
  count) — the viewer renders these.
- Mode A fan-out keeps its existing per-task retry semantics (lease
  machinery); `max_restarts` applies per-trial subprocess relaunch only
  when the trial wrote no accepted metrics (keep simple: relaunch failed
  trials up to max_restarts).
- e2e: the kill-at-40 example now completes in ONE submit with
  `max_restarts=1` (asserting resumed_from==40 and bit-identical loss);
  a deterministic-failure script (bad import) asserts FAIL_JOB fast-stop
  (no restart burn).

### A5. Async submit + CLI

- `submit(..., wait=True)` default unchanged; `wait=False` returns
  immediately with a background thread driving the launch loop; `run.wait
  (timeout=None)`, `run.state` live, `run.events` append-only list
  (thread-safe append; reads are snapshots).
- CLI: `flashruntime submit CMD [--source DIR] [--task-params JSON]
  [--max-restarts N] [--output-dir DIR] [--watch/--no-watch]` in
  `service/cli.py` (same argparse style as existing commands). `--watch`
  opens the viewer (B); default on when stdout is a TTY.

### A6. RunPod real-GPU validation harness (dev-only, not packaged)

`scripts/runpod_gpu_e2e.py` — stdlib + `httpx` (dev extra) against the
RunPod REST API (`api.runpod.io/graphql` or v2 REST; key from
`RUNPOD_API_KEY` in `.env`, loaded via os.environ — the script refuses to
run if the key looks committed anywhere):
1. Provision cheapest available 2-GPU pod (target RTX 4090 community,
   fallback A5000/3090), CUDA-enabled PyTorch base image.
2. Wait ready → SSH/exec: pip install the locally-built wheel (uploaded via
   the pod's HTTP proxy or pulled from a temp URL — simplest: build wheel,
   `runpodctl`-free scp over exposed SSH), install CPU-free torch already in
   image; copy `examples/user_pytorch/` + the GPU e2e test file.
3. Run: real 2-GPU nccl DDP train-to-completion; kill-and-resume with
   `max_restarts=1` on GPU; capture metrics + `nvidia-smi` proof.
4. Fetch results JSON + logs into `scripts/runpod_results/` (gitignored).
5. **Always terminate the pod** (finally-block + orphan sweep listing pods
   tagged `flashruntime-e2e`). Hard caps: 30 min wall clock, one pod at a
   time, abort+terminate on any step failure. Estimated cost ≤ $1/run.
Success criterion: the results JSON shows nccl backend, 2 CUDA devices,
resumed_from>0, loss finite — recorded into `docs/` as the "validated on"
note and into PROGRESS.md.

## 4. Workstream B — live run viewer (open on submit)

New `flashruntime/viewer/` (core-safe: stdlib only — `http.server`,
`json`, `threading`, `webbrowser`):
- `viewer/server.py` — `RunViewerServer(run_dir, port=0)`: background
  `ThreadingHTTPServer` serving (a) `GET /` the page, (b) `GET /api/state`
  — a JSON snapshot assembled from the run directory: run/attempt states +
  restart decisions (from a `run.json` the SDK now writes per attempt),
  workers/ranks (from launcher pids + heartbeat mtimes), `metrics.jsonl`
  tail, checkpoint manifests (via `latest`/all with validity), events,
  `launcher.log` tail; (c) `GET /docs/...` the built docs (C). Read-only:
  the server never mutates the run.
- `viewer/page.py` — `_PAGE`: ONE self-contained HTML page, no external
  assets, reimplementing the cloud viz language: dark `#0d1117`-family
  surface, oklch accent palette, ui-monospace, uppercase tracked section
  labels; canvas topology drawing (machine box → rank/worker nodes with
  heartbeat pulse animation; Mode A task grid with lease-state colors),
  canvas loss curve from metrics.jsonl, checkpoint timeline with
  hash-verified badges, restart/event feed. Polls `/api/state` every 2 s.
- SDK integration: `submit(..., watch=None)` — None = auto (open when
  `sys.stdout.isatty()` and not CI), True/False explicit. On start:
  `webbrowser.open(f"http://127.0.0.1:{port}")`. Works for sync and async
  submit (sync runs server on a thread; page keeps history after
  completion until process exit; `run.viewer_url` exposed).
- The SDK writes `run.json` (workload summary, mode, trials, per-attempt
  status+timestamps, restart decisions) into the output dir — the single
  data contract between SDK and viewer, versioned `viewer_v1`.
- Coordinator dashboard (`service/dashboard.py`) is restyled to match
  (shared palette constants duplicated deliberately — no cross-import from
  viewer to service).

## 5. Workstream C — documentation site

- Sources: `docs/site/` markdown — structure mirroring PyTorch:
  `index.md` (landing: what/why, 60-second demo GIF-less code), `get-
  started.md` (install, first run, first DDP), `tutorials/convnet.md`
  (**centerpiece: the user's OurConvNet class verbatim** — plain script →
  `ft.prepare` one-import → `flash.submit(pytorch.ddp(...))` → kill-and-
  resume with `max_restarts=1` → watch it in the viewer),
  `tutorials/sklearn-sweeps.md`, `tutorials/fault-tolerance.md`,
  `guides/` (adapted from existing bring-your-code.md, split per
  framework + isolation/jobspec), `concepts/architecture.md` (four axes,
  leases, manifests), `reference/` (SDK, integrations, flashruntime.torch,
  CLI — hand-written, accurate against code, doc-tested).
- Build: `scripts/build_docs.py` (dev-only; `markdown` lib in `[dev]`)
  renders markdown → static HTML with the same visual language (shared
  template: sidebar nav like PyTorch docs, client-side search over a
  generated JSON index, code tabs, copy buttons). Output →
  `flashruntime/viewer/_docs/` (packaged, served by the viewer at /docs)
  and `site/` (gitignored; CI deploys it to GitHub Pages).
- CI: docs build + link check in ci.yml; deploy on release and on main.
- `tests/test_documentation.py` extended: docs build succeeds, internal
  links resolve, every ```python block in reference/tutorial pages
  compiles (`compile()` smoke — not executed).

## 5b. Workstream D — evaluation & benchmarks

Purpose: back the pitch with measurements — "equal-or-better performance
than the alternatives, with far less adoption effort." **Honesty rule:
results are measured, never asserted.** The benchmark suite is the source
of truth; the docs page renders whatever it measured, favorable or not,
with pinned versions, fixed seeds, N≥5 repeats, medians + spread, and the
exact hardware recorded. If a number is unfavorable, it ships with a note
explaining the trade-off — credibility is the product here.

- `benchmarks/` (dev-only, not packaged): a small registry of scenarios,
  each a module with `run(repeats) -> ResultRow` (adding a scenario = one
  file — the horizontal-extensibility rule applied to benchmarks):
  1. **Launch overhead** — `flash.submit(ddp(...))` vs bare `torchrun`
     on the identical script: wall-clock delta (hypothesis: small constant
     seconds, ~0% of any real training run).
  2. **Training-loop overhead** — steps/sec of the `ft.prepare`-based
     script vs the vanilla-DDP script between checkpoints (hypothesis:
     ≈0%), plus the measured cost of one `ft.checkpoint` (state_dict save
     + sha256 manifest) vs plain `torch.save`.
  3. **Recovery economics** — the headline: crash at step k; time-to-
     trained-model with `max_restarts=1` (auto-resume from step k) vs
     restart-from-scratch (what raw torchrun gives you). Report wall-clock
     saved and "lost work" steps.
  4. **HPO sweep** — N sklearn trials: flashruntime fan-out vs sequential
     loop vs (optional job, heavy dep) Ray Tune on the same grid: wall
     clock, peak RSS, and **setup lines-of-code**.
  5. **Adoption cost & footprint** — measured, not vibes: LOC diff to make
     the ConvNet script distributed+fault-tolerant with flashruntime
     (import + 2 calls) vs Accelerate vs Ray Train (their own quickstart
     patterns, cited); `pip install` dependency count and cold
     `import` time (pydantic-only core vs ray/accelerate imports).
- Output: `benchmarks/results/*.json` (schema `bench_v1`: scenario, hw
  fingerprint, versions, repeats, median, p10/p90) — gitignored; a
  checked-in `benchmarks/results/baseline-<host>.json` snapshot powers the
  docs page so the site never shows unmeasured numbers.
- Docs: `docs/site/benchmarks.md` auto-includes tables (and simple bars)
  generated from the results JSON at docs-build time, with the methodology
  and reproduction command (`python -m benchmarks run --all`) printed
  beside every table.
- CPU scenarios run in CI as a smoke (1 repeat, assert-they-run not
  assert-numbers); full runs are local/RunPod. GPU variants of #2/#3 ride
  the A6 RunPod harness.
- Comparators are optional extras (`benchmarks/requirements.txt`), never
  runtime deps of flashruntime.

## 6. Acceptance criteria (0.1.0 gate)

1. `pip install dist/flashruntime-*.whl` in a clean venv → `import
   flashruntime` (pydantic-only) → the 60-second quickstart from the docs
   runs.
2. Prototype engine gone; full suite green; no dangling references
   (doc-link test + grep sweep clean).
3. CI matrix green on GitHub; release workflow dry-runs (build job)
   locally via `python -m build`.
4. `flash.submit(ddp(...), max_restarts=1)` survives the kill-at-40 crash
   in ONE call with bit-identical final loss; deterministic app error
   fail-fasts without burning restarts; both asserted in e2e.
5. `submit(wait=False)` + `run.wait()` works; `flashruntime submit` CLI
   runs the sklearn example from the shell.
6. Submitting with `--watch` opens the viewer; `/api/state` shows live
   rank states, loss points, manifests, and the restart decision during
   the kill-and-resume e2e (asserted via HTTP in a test — no browser
   needed).
7. Docs site builds; viewer serves it at /docs; ConvNet tutorial's code
   blocks compile; Pages deploy job configured.
8. RunPod harness executes end-to-end: real 2-GPU nccl DDP + GPU
   kill-and-resume pass; pod provably terminated; cost report printed;
   results recorded in PROGRESS.md. (Requires RUNPOD_API_KEY in .env —
   present.)
9. Secrets audit clean (incl. the RunPod key pattern) across history and
   worktree-minus-.env.
10. Benchmark suite runs (`python -m benchmarks run --all` locally; CI
    smoke green); a measured baseline JSON is checked in; the docs
    benchmarks page renders only measured numbers with methodology +
    repro command; recovery-economics and adoption-cost scenarios show
    the differentiator honestly.
11. Readability gate: every task review explicitly checks "could a
    newcomer follow this, and does every piece state its reason?" —
    clever-but-opaque code is a review failure by definition (§2b).

## 7. Open follow-ups (created by this work)

- Rotate the RunPod key after validation (it transited chat).
- Multi-node + flashnode runner (0.2.0/0.3.0) — viewer already renders
  multi-worker topology, so remote workers plug into the same page later.
- Docs versioning (mike or manual) once 0.2.0 exists.
