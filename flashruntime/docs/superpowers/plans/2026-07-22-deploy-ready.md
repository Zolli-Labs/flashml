# Deploy-Ready 0.1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pip-publishable flashruntime 0.1.0: prototype engine removed, GPU wiring + automatic crash recovery + async submit + CLI, a live run viewer that opens on submit, a PyTorch-style docs site (served in the viewer and on GitHub Pages), an honest benchmark suite, CI/release pipelines, and real-GPU validation on RunPod.

**Architecture:** Extends the shipped command-workloads feature (spec `2026-07-21`). New: `flashruntime/viewer/` (stdlib-only server + one self-contained page), `docs/site/` + dev-time builder, `benchmarks/` registry, `.github/workflows/`, `scripts/runpod_gpu_e2e.py`. Spec: `docs/superpowers/specs/2026-07-22-deploy-ready-design.md`.

**Tech Stack:** Python ≥3.10, pydantic v2, stdlib `http.server`/`threading`/`webbrowser` for the viewer, `markdown` (dev-only) for docs, `httpx` (dev-only) for RunPod, GitHub Actions.

## Global Constraints

- **Readability gate (spec §2b, user-mandated):** every piece of code must be understandable by a newcomer and state its reason (docstrings explain *why*). Clever-but-opaque code is a review failure. Reviewers check this explicitly on every task.
- **Horizontal extensibility (spec §2b):** adding a framework stays one adapter file; the viewer consumes only the versioned `run.json` contract; benchmarks add scenarios via one-file registry entries.
- **Pydantic-only core:** `import flashruntime` (and now `flashruntime.viewer`) needs stdlib+pydantic only. `markdown`, `httpx`, comparator libs are dev/benchmark-only. torch only inside `flashruntime/torch/` function bodies.
- **No Node/build step anywhere.** Viewer page and docs pages are self-contained HTML; docs are rendered by a dev-time Python script.
- **flashml-cloud is style reference only — never copy its code** (private repo). The oklch/dark/mono/canvas language is reimplemented.
- **Secrets:** `.env` never tracked; the RunPod key never appears in code, briefs, reports, logs, or commits — read from `os.environ` only. The audit (T11) greps history for `rpa_` among other patterns.
- **Honest benchmarks (spec §5b):** measured numbers only; medians of N≥5 repeats locally (1-repeat smoke in CI); pinned versions + hardware fingerprint recorded; unfavorable results ship with a trade-off note, never deleted.
- **Cost caps (T12):** one RunPod pod at a time, ≤30 min, always-terminate in `finally`, abort on any step failure. Target ≤$1/run.
- **Branch:** `local-milestone-2026-07`. Full suite green before every commit (`PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest`); baseline at plan start: 183 passed + 4 deselected. Do not push.
- Match existing code style; `from __future__ import annotations`; commit after every task.

## File Map (delta)

| Path | Status | Responsibility |
|---|---|---|
| `flashruntime/{engine,algorithms,adapters,storage}/`, `docs/archive-prototype/`, `examples/local_kmeans_and_linear_regression.py`, `tests/test_local_provider.py` | **delete** | prototype engine (T1) |
| `flashruntime/torch/__init__.py` | modify | GPU device wiring (T2) |
| `flashruntime/sdk.py` | modify | `run.json`, async submit, `max_restarts`, `watch` (T3, T4, T7) |
| `flashruntime/recovery/signals.py` | create | exit-code/log → `FailureSignals` translation (T4) |
| `flashruntime/service/cli.py` | modify | `flashruntime submit` (T5) |
| `flashruntime/viewer/{__init__,server,page,state}.py` | create | live run viewer (T6, T7) |
| `flashruntime/viewer/_docs/` | generated+packaged | built docs served at /docs (T8) |
| `docs/site/**/*.md`, `scripts/build_docs.py` | create | docs sources + builder (T8, T9) |
| `benchmarks/` (`__main__.py`, `registry.py`, `scenarios/*.py`, `report.py`, `requirements.txt`, `results/baseline-*.json`) | create | evaluation suite (T10) |
| `.github/workflows/{ci,release}.yml`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `flashruntime/py.typed`, `scripts/audit_secrets.sh` | create | release engineering (T11) |
| `scripts/runpod_gpu_e2e.py` | create | real-GPU validation (T12) |

Sequencing rationale: T1 first so all content targets the clean API; T3 (run.json/async) before T4 (recovery writes into it) and T6 (viewer reads it); T8 before T9 (builder before content); T10 after T4 (benchmarks measure auto-recovery); T11 after T8/T10 (CI covers docs+bench); T12 last (needs the wheel + everything).

---

### Task 1: Remove the prototype engine

**Files:**
- Delete: `flashruntime/engine/`, `flashruntime/algorithms/`, `flashruntime/adapters/`, `flashruntime/storage/`, `examples/local_kmeans_and_linear_regression.py`, `tests/test_local_provider.py`, `docs/archive-prototype/` (entire tree)
- Modify: `flashruntime/__init__.py` (remove `_PROTOTYPE_EXPORTS` block + its `__all__` entries + module docstring mention), `pyproject.toml` (drop `[prototype]` extra; `sklearn` extra keeps numpy+scikit-learn — still used by `flashml_workloads/sklearn_trial.py`), `README.md` (remove Quickstart 3 + `Cluster.train` example + prototype layout lines), `AGENTS.md` (current-state + hard-rule mentions), `docs/guides/bring-your-code.md` (rewrite the "Built-in algorithms" section: built-in *task modules* in `flashml_workloads/` + Mode A are the examples now; `Cluster.train` is gone)
- Test: existing suite (no new file)

**Interfaces:**
- Consumes: nothing new.
- Produces: `flashruntime.Cluster/Job/JobEvent/algorithms/registered_providers` **no longer exist** — later tasks and docs must never reference them.

- [ ] **Step 1: Inventory references before deleting** — `grep -rn "engine\.\|algorithms\.\|Cluster\|adapters\.\|flashruntime\.storage\|prototype" flashruntime/ tests/ examples/ docs/ README.md AGENTS.md pyproject.toml --include="*.py" --include="*.md" --include="*.toml" | grep -v archive-prototype` and list every hit that will dangle. (Expect: `__init__.py`, README, AGENTS, guide, the four packages themselves, `test_local_provider.py`. `flashml_workloads/sharded_kmeans.py` mentions nothing from these packages — verify.)
- [ ] **Step 2: Delete + prune** per the file list. In `flashruntime/__init__.py` remove the lazy prototype block wholesale; keep `_SDK_EXPORTS` untouched. In the module docstring, drop item 4's prototype mention and the `[prototype]`/`[sklearn]` engine line (sklearn extra stays, reworded to "task-module examples").
- [ ] **Step 3: Sweep check** — rerun the Step-1 grep; only historical mentions in `docs/superpowers/` (spec/plan archives) may remain. Run `pytest tests/test_documentation.py -q` (link checker must pass with `docs/archive-prototype/` gone — fix any doc that linked into it).
- [ ] **Step 4: Full suite + core smoke**

Run: `PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest -q` → expect **182 passed** (183 − 1 removed test file's test... verify actual count and record it) and `.venv/bin/python -c "import flashruntime; print(flashruntime.PLANNER_VERSION)"` → prints version.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "refactor!: remove the pre-K8s prototype engine — one story, bring your code"`

---

### Task 2: GPU device wiring in `flashruntime.torch`

**Files:**
- Modify: `flashruntime/torch/__init__.py`
- Test: `tests/test_torch_helper.py` (extend)

**Interfaces:**
- Produces: pure helper `_resolve_device(world_size: int, cuda_available: bool, local_rank: int) -> str` returning `"cpu"` or `f"cuda:{local_rank}"`; `prepare()` moves the model to that device *before* DDP wrap and passes `device_ids=[local_rank]` + `output_device=local_rank` when on CUDA; restore uses `map_location=device`; `checkpoint()` moves state dicts to CPU before save (unchanged manifests contract). Surface stays exactly the 7 functions (guardrail). *(Amended in T12, reviewer-blessed: +`device()`/`backend()` read-only accessors — the guardrail is capability, not count.)*

- [ ] **Step 1: Write failing tests** (CPU-runnable — the device *decision* is pure):

```python
def test_resolve_device_is_pure_and_explicit():
    from flashruntime.torch import _resolve_device

    assert _resolve_device(world_size=1, cuda_available=False, local_rank=0) == "cpu"
    assert _resolve_device(world_size=2, cuda_available=False, local_rank=1) == "cpu"
    assert _resolve_device(world_size=1, cuda_available=True, local_rank=0) == "cuda:0"
    assert _resolve_device(world_size=4, cuda_available=True, local_rank=3) == "cuda:3"


def test_checkpoint_state_dicts_are_cpu(ft, tmp_path):
    # saved tensors must be CPU regardless of training device, so manifests
    # stay topology- and device-agnostic (restore maps them wherever needed)
    model = _model()
    ft.checkpoint(model, step=5)
    state = torch.load(tmp_path / "ckpt" / "step-000005" / "model.pt", map_location=None)
    assert all(t.device.type == "cpu" for t in state.values())
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_torch_helper.py -k "resolve_device or cpu" -v` → ImportError/AttributeError for `_resolve_device`.
- [ ] **Step 3: Implement.** `_resolve_device` as a module-level pure function with a docstring stating the reason (device choice must be testable without CUDA). In `prepare()`: compute `local_rank`/`device = _resolve_device(...)` once; if device is CUDA: `torch.cuda.set_device(local_rank)`, `model = model.to(device)`; DDP wrap gains `device_ids=[local_rank], output_device=local_rank` **only when CUDA** (gloo/CPU keeps the no-args wrap — DDP requires that); restore `torch.load(..., map_location=device)` and `model/optimizer` load onto the live device. In `checkpoint()`: `{k: v.detach().cpu() for k, v in target.state_dict().items()}` before save (reason in comment: device-agnostic manifests). Single-process CUDA (world_size==1, cuda available) also moves the model — the "just works on a GPU box" path.
- [ ] **Step 4: Full suite** → green (existing gloo e2e unaffected: CPU path unchanged).
- [ ] **Step 5: Update the guide's GPU sentence** in `docs/guides/bring-your-code.md`: from "later slice" to "CUDA device placement is wired; validated on RunPod (see PROGRESS)" — but keep "multi-node is a later slice".
- [ ] **Step 6: Commit** — `feat(torch): CUDA device placement in prepare/checkpoint (validated on RunPod in T12)`

---

### Task 3: `run.json` contract + async submit

**Files:**
- Modify: `flashruntime/sdk.py`
- Test: `tests/test_sdk_submit.py` (extend), `tests/test_run_json.py` (create)

**Interfaces:**
- Produces (the SDK↔viewer contract, `viewer_v1` — T6 consumes it verbatim):

```python
# written to <output_dir>/run.json after every state change, atomically
# (tmp + os.replace — a reader must never see a torn file)
{
  "contract": "viewer_v1",
  "workload": {"command": [...], "mode": "coordinated", "source": "..."},
  "state": "PENDING|RUNNING|SUCCEEDED|FAILED|CANCELLED",
  "started_at": 1753...,          # unix seconds
  "finished_at": null,
  "max_restarts": 1,
  "attempts": [                    # one entry per launch (restarts append)
    {"attempt_id": "task-000", "job_id": "local", "state": "RUNNING",
     "pid": "12345", "started_at": ..., "finished_at": null,
     "output_dir": "..."}
  ],
  "events": [                      # append-only; recovery decisions land here (T4)
    {"ts": ..., "type": "LAUNCH_STARTED", "message": "..."}
  ],
  "trials": [ ... ]                # metrics dicts, as today
}
```

- `submit(workload, output_dir=None, wait=True)`; `wait=False` returns a `Run` immediately with a daemon thread driving the loop; `Run.wait(timeout=None) -> LaunchState`; `Run.state`/`Run.events` readable live (guarded by one `threading.Lock`; readers get copies). `Run.run_json_path` property.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_run_json.py
import json, time


def test_run_json_written_and_versioned(tmp_path):
    import flashruntime as flash

    src = _write_script(tmp_path, "import json; json.dump({}, open('metrics.json','w'))")
    run = flash.submit(flash.CommandWorkload(command=f"{sys.executable} train.py",
                                             source={"path": src}), output_dir=tmp_path / "o")
    doc = json.loads((tmp_path / "o" / "run.json").read_text())
    assert doc["contract"] == "viewer_v1"
    assert doc["state"] == "SUCCEEDED"
    assert doc["attempts"][0]["state"] == "SUCCEEDED"
    assert any(e["type"] == "LAUNCH_STARTED" for e in doc["events"])


def test_async_submit_wait(tmp_path):
    import flashruntime as flash

    src = _write_script(tmp_path, "import time; time.sleep(1)")
    run = flash.submit(flash.CommandWorkload(command=f"{sys.executable} train.py",
                                             source={"path": src}),
                       output_dir=tmp_path / "o", wait=False)
    assert run.state.value in ("PENDING", "RUNNING")  # returned before finish
    assert run.wait(timeout=30).value == "SUCCEEDED"
    assert run.state.value == "SUCCEEDED"
```

(reuse `_write_script` from test_sdk_submit; import it or duplicate the 5-line helper — duplication is fine here, note why.)
- [ ] **Step 2: RED** — attributes/file missing.
- [ ] **Step 3: Implement.** Refactor `submit()` minimally: extract the existing loop body into `_drive(run, workload, launcher, param_sets)` (same code, one indentation move — reviewers must see this is a move, not a rewrite); `wait=True` calls `_drive` inline (behavior identical to today), `wait=False` runs it in `threading.Thread(daemon=True)`. `Run` gains `_lock`, `_write_run_json()` (tmp+`os.replace`, called on every transition), `record_event(type, message)`, `wait(timeout)` (Event-based, not polling). Docstrings state reasons (atomic write: a reader must never parse a torn file; lock: viewer thread reads while driver writes).
- [ ] **Step 4: Full suite** → green.
- [ ] **Step 5: Commit** — `feat(sdk): versioned run.json (viewer_v1) + async submit/wait`

---

### Task 4: Automatic fault tolerance (`max_restarts`)

**Files:**
- Create: `flashruntime/recovery/signals.py`
- Modify: `flashruntime/sdk.py`
- Test: `tests/test_auto_recovery.py` (create), `tests/test_examples_e2e.py` (extend)

**Interfaces:**
- Consumes: `recovery.classify(FailureSignals)`, `recovery.decide(failure, mode)` (existing, tested, never-before-wired), `latest_valid_manifest`.
- Produces: `signals.from_local_launch(exit_code: int | None, log_tail: str) -> FailureSignals` — a small, transparent translation table (exit 0→n/a; deterministic Python tracebacks (`SyntaxError`, `ImportError`, `ModuleNotFoundError`, an explicit `Traceback` + nonzero) → application-error signals; signal deaths/OOM patterns → worker-crash signals; docstring lists every rule and why). `submit(..., max_restarts: int = 0)`; on FAILED with restarts remaining: classify → decide → if action is `FAIL_JOB` stop and record; else record `FAILURE_CLASSIFIED` + `RECOVERY_ACTION_SELECTED` events in run.json and relaunch the same spec (`ft.prepare` in the child resumes from the newest valid manifest — that's the whole trick and the docstring says so). Applies to non-fanout launches and per-trial in fan-out.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_auto_recovery.py
def test_signals_deterministic_error_maps_to_application_error():
    from flashruntime.protocol.v1alpha1 import FailureClass
    from flashruntime.recovery import classify
    from flashruntime.recovery.signals import from_local_launch

    sig = from_local_launch(1, "Traceback (most recent call last):\nModuleNotFoundError: No module named 'x'")
    assert classify(sig) is FailureClass.APPLICATION_ERROR


def test_crash_then_auto_resume_single_call(tmp_path):
    """A script that dies at step 3 on fresh runs (marker file) and counts
    steps in a checkpoint-like file: with max_restarts=1 the SECOND attempt
    resumes and finishes — one submit() call, no human."""
    import flashruntime as flash

    src = _write_script(tmp_path, '''
        import json, os, pathlib
        ck = pathlib.Path(os.environ["FLASHML_CKPT_DIR"]); ck.mkdir(parents=True, exist_ok=True)
        state = ck / "progress.txt"
        start = int(state.read_text()) if state.exists() else 0
        for step in range(start + 1, 7):
            state.write_text(str(step))
            if step == 3 and start == 0:
                raise SystemExit(9)   # simulated crash, fresh run only
        json.dump({"steps": 6, "resumed_from": start}, open("metrics.json", "w"))
    ''')
    run = flash.submit(flash.CommandWorkload(command=f"{sys.executable} train.py",
                                             source={"path": src}),
                       output_dir=tmp_path / "o", max_restarts=1)
    assert run.state.value == "SUCCEEDED"
    assert run.trials[0]["resumed_from"] == 3
    types = [e["type"] for e in run.events]
    assert "FAILURE_CLASSIFIED" in types and "RECOVERY_ACTION_SELECTED" in types


def test_deterministic_failure_fails_fast_without_burning_restarts(tmp_path):
    import flashruntime as flash

    src = _write_script(tmp_path, "import definitely_not_a_module")
    run = flash.submit(flash.CommandWorkload(command=f"{sys.executable} train.py",
                                             source={"path": src}),
                       output_dir=tmp_path / "o", max_restarts=3)
    assert run.state.value == "FAILED"
    doc = json.loads((tmp_path / "o" / "run.json").read_text())
    assert len(doc["attempts"]) == 1          # FAIL_JOB: no retry storm
    assert any(e["type"] == "FAILURE_CLASSIFIED" for e in doc["events"])
```

- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** `signals.py` (transparent mapping, every rule one `if` with a comment naming its reason) and the submit retry loop: `for restart in range(max_restarts + 1): launch; if SUCCEEDED: break; signals→classify→decide(mode=...); record both events (with the decision's reason string); if decision.action is FAIL_JOB or restarts exhausted: break`. Fan-out: same per-trial loop. Attempt ids stay deterministic (`task-000` … reused; restart attempts append `-r1` suffix: `task-000-r1` — the launcher's job-scoped ckpt dir is what makes resume work across differently-named attempts; state that in a comment).
- [ ] **Step 4: e2e upgrade** — in `tests/test_examples_e2e.py::test_kill_and_resume...`, replace the two-submit choreography with ONE `submit(..., max_restarts=1)` asserting `resumed_from == 40`, bit-identical loss vs baseline, and both recovery events present. Keep the old two-submit path as a second, smaller assertion? No — delete it; the single-call form supersedes it (note in the test docstring).
- [ ] **Step 5: Full suite (incl. e2e)** → green.
- [ ] **Step 6: Commit** — `feat(sdk): max_restarts — automatic classify/decide/resume recovery`

---

### Task 5: `flashruntime submit` CLI

**Files:**
- Modify: `flashruntime/service/cli.py`
- Test: `tests/test_cli_submit.py` (create)

**Interfaces:**
- Produces: `flashruntime submit CMD [--source DIR] [--task-params JSON] [--max-restarts N] [--output-dir DIR] [--watch | --no-watch]` → builds `CommandWorkload`, calls `flash.submit`, prints a human summary (state, trials count, best trial if primary metric, output dir, viewer URL when watching), exit code 0/1 by state. `--watch` default: on iff `sys.stdout.isatty()` (reason: never block/open browsers in CI). Viewer wiring lands in T7 — until then `--watch` prints "viewer not yet available" (honest placeholder, removed in T7).

- [ ] **Step 1: Failing test** — invoke `main(["submit", f"{sys.executable} train.py", "--source", src, "--output-dir", str(out), "--no-watch"])` (import `main` from `flashruntime.service.cli`); assert exit 0, `run.json` exists, stdout mentions SUCCEEDED. A failing script → exit 1.
- [ ] **Step 2: RED** (unknown command).
- [ ] **Step 3: Implement** following the existing cli.py subcommand pattern exactly (read it first; match its argparse style and how it prints). Keep the handler ≤50 lines; parsing `--task-params` is `json.loads` with a clear error message.
- [ ] **Step 4: Full suite → green. Commit** — `feat(cli): flashruntime submit — shell front door for command workloads`

---

### Task 6: Viewer server + state API

**Files:**
- Create: `flashruntime/viewer/__init__.py`, `flashruntime/viewer/state.py`, `flashruntime/viewer/server.py`
- Test: `tests/test_viewer_server.py`

**Interfaces:**
- `state.collect(run_dir: Path) -> dict` — pure function assembling the `/api/state` JSON from disk: parsed `run.json` (contract-checked: unknown contract → `{"error": ...}`, never a crash), per-attempt `metrics.jsonl` tail (last 500 points: `[{"step":..,"loss":..}, ...]` — pass through whatever keys exist), checkpoint manifests under the job ckpt roots via `latest_valid_manifest`-adjacent listing (each: step, validation, part count, age), `launcher.log` tail (last 100 lines), events. Read-only, exception-safe (a half-written file yields a partial snapshot, never a 500 — reason in docstring: the viewer must never interfere with or die because of a live run).
- `server.RunViewerServer(run_dir: Path, port: int = 0)` — `.start() -> str` (returns `http://127.0.0.1:{port}`), `.stop()`; stdlib `ThreadingHTTPServer` + `BaseHTTPRequestHandler` routing exactly: `/` → page (T7; until then a minimal placeholder page), `/api/state` → `state.collect` JSON, `/docs` + `/docs/...` → files under `flashruntime/viewer/_docs/` if present else 404 with a one-line "docs not built" body. Nothing else; unknown paths 404. No logging spam (override `log_message` to silence — reason: the user's terminal belongs to their training output).

- [ ] **Step 1: Failing tests** — build a fake run dir (write a `viewer_v1` run.json + metrics.jsonl + a manifest via `write_manifest`); `collect()` returns the assembled dict (assert keys + metrics tail + manifest step); torn `run.json` (write half a JSON) → `collect` returns dict with `"error"` and no exception; server test: start on port 0, `urllib.request.urlopen(url + "/api/state")` → 200 + JSON matches `collect`; `/nope` → 404; stop() frees the port.
- [ ] **Step 2: RED. Step 3: Implement** (~120 lines total across the two modules; every route and every guard commented with its reason). **Step 4: full suite. Step 5: Commit** — `feat(viewer): stdlib run-state server (/api/state, /docs)`

---

### Task 7: Viewer page + open-on-submit

**Files:**
- Create: `flashruntime/viewer/page.py` (the one sanctioned large file)
- Modify: `flashruntime/sdk.py` (`watch` param), `flashruntime/service/cli.py` (wire `--watch`), `flashruntime/service/dashboard.py` (palette alignment only — swap its color values to the shared tokens; no structural change)
- Test: `tests/test_viewer_page.py`

**Interfaces:**
- `page.render() -> str` — one self-contained HTML document (no external assets/CDNs). **Authored task** — implementer designs within this contract:
  - Visual tokens (top of file as a Python dict, single source of truth, also used by dashboard.py): background `#0d1117` family, panel `#161b22`, border `#21262d`, text `#c9d1d9`/muted `#8b949e`, accents as oklch strings — cyan `oklch(0.80 0.16 200)` (running), green `oklch(0.76 0.18 145)` (succeeded/verified), amber `oklch(0.80 0.18 60)` (leased/recovering), red `oklch(0.70 0.20 20)` (failed), violet `oklch(0.65 0.20 290)` (checkpoints). ui-monospace stack; uppercase letter-spaced section labels — the established house style (see `service/dashboard.py`).
  - Sections, in order: **header** (workload command, mode, state badge, restart counter); **topology canvas** — machine box containing one node per attempt/rank, colored by state, soft radial pulse animation while RUNNING (reimplementation of the cloud viz *language*, not its code); Mode A fan-out renders the task grid with lease-state colors; **loss curve canvas** — device-pixel-ratio-aware, from `/api/state` metrics tail, autoscaled with padded bounds, last value labeled; **checkpoint timeline** — manifests as violet markers with step + a `hash-verified` badge; **events feed** — recovery decisions highlighted (FAILURE_CLASSIFIED amber, RECOVERY_ACTION_SELECTED cyan, with the reason text); **logs tail** collapsible; **Docs** link to `/docs/` top-right.
  - Behavior: poll `/api/state` every 2 s (pause on `document.hidden` — reason: don't burn laptop batteries); keep rendering after run completion; all drawing in plain JS functions named after their section (`drawTopology`, `drawLoss`) — readable, commented, no minification, no framework.
- SDK: `submit(..., watch: bool | None = None)` — None = auto (`sys.stdout.isatty()` and env `CI` unset); when on: start `RunViewerServer(output_dir)`, `webbrowser.open(url)`, print the URL; server lifetime = process (daemon thread); `Run.viewer_url: str | None`. CLI `--watch/--no-watch` maps to True/False and prints the URL.
- Test contract (headless — no browser): `render()` returns HTML containing the literal markers `id="topology"`, `id="loss"`, `id="checkpoints"`, `id="events"`, `"/api/state"`, and NO `http(s)://` external asset references (regex for `src=`/`href=` pointing off-host, excluding the `/docs` link); server `/` returns it; `submit(..., watch=True)` with `webbrowser.open` monkeypatched records a `127.0.0.1` URL and the server actually serves `/api/state` for that run while it executes.

- [ ] **Step 1: failing tests per the contract above. Step 2: RED. Step 3: implement (author the page). Step 4: manual look** — run `PATH=... python examples/bring_your_code_demo.py` with watch forced on once, screenshot-level sanity by eye (implementer describes what renders in the report). **Step 5: full suite. Step 6: Commit** — `feat(viewer): live run page — topology, loss, checkpoints, recovery feed — opens on submit`

---

### Task 8: Docs builder + site skeleton

**Files:**
- Create: `scripts/build_docs.py`, `docs/site/index.md`, `docs/site/get-started.md`, `docs/site/_nav.yml`
- Modify: `pyproject.toml` (`dev` extra += `markdown`, `pyyaml` already present; `[tool.setuptools.package-data] flashruntime = ["py.typed", "viewer/_docs/**"]` — py.typed itself lands in T11), `.gitignore` (+ `site/`, `flashruntime/viewer/_docs/`)
- Test: `tests/test_documentation.py` (extend)

**Interfaces:**
- `build_docs.py`: reads `docs/site/_nav.yml` (ordered `{section: [file, ...]}` — the nav is data, not code, so adding a page = adding a line); renders each md via `markdown` (extensions: `fenced_code`, `tables`, `toc`) into a shared HTML template (same visual tokens as the viewer — import them from `flashruntime.viewer.page`); sidebar nav from `_nav.yml` with the current page highlighted; client-side search: builder emits `search-index.json` (page → title + plain text) + a ≤60-line vanilla JS filter box; code blocks get a copy button. Outputs BOTH to `flashruntime/viewer/_docs/` and `site/`. `--check` mode: build to temp, verify every internal link + nav entry resolves, exit 1 on failure.
- Doc tests: build succeeds in-process; every internal href resolves; every ```python block in `docs/site/**` passes `compile(src, name, "exec")` (not executed — reason: docs must at least be syntactically true).

- [ ] Steps: failing tests (build + linkcheck + compile-blocks against a stub page with a deliberate bad link fixture) → RED → implement builder (≤250 lines, sectioned with why-comments) + `index.md` (what/why + the 60-second demo) + `get-started.md` (install, first run, first DDP — accurate against the T1-cleaned API) → GREEN → full suite → **Commit** — `feat(docs): PyTorch-style site builder — served in the viewer, deployable to Pages`

---

### Task 9: Docs content (authored)

**Files:**
- Create: `docs/site/tutorials/convnet.md`, `docs/site/tutorials/sklearn-sweeps.md`, `docs/site/tutorials/fault-tolerance.md`, `docs/site/guides/pytorch.md`, `docs/site/guides/sklearn.md`, `docs/site/guides/huggingface.md`, `docs/site/guides/jobspec-and-isolation.md`, `docs/site/concepts/architecture.md`, `docs/site/reference/{sdk,integrations,torch-helper,cli}.md`; update `_nav.yml`
- Modify: `docs/guides/bring-your-code.md` (slim to a pointer at the site sections; keep it valid for the repo-browser reader)
- Test: existing doc tests (build/linkcheck/compile) now cover all pages

**Interfaces / requirements (authored like the shipped bring-your-code guide — verify every API against code, not memory):**
- **`tutorials/convnet.md` is the centerpiece.** It uses the user's pictured model VERBATIM as the starting point:

```python
class OurConvNet(torch.nn.Module):
    def __init__(self, num_outputs=20):
        super(OurConvNet, self).__init__()
        self.Conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.Sigma = nn.Sigmoid()
        self.Avg = nn.AvgPool2d(2, stride=2, padding=0)
        self.Conv2 = nn.Conv2d(64, 128, 5, 1)
        self.Fl = nn.Flatten()
        self.Linear = nn.Linear(128*5*5, num_outputs)

    def forward(self, x):
        out = self.Sigma(x)
        out = self.Conv1(out)
        out = self.Avg(out)
        out = self.Conv2(out)
        out = self.Avg(out)
        out = self.Fl(out)
        out = self.Linear(out)
        return out
```

  Arc: (1) the plain single-process training script around this class (synthetic 3×24×24 data so shapes work: 24→pool 12→conv 8→pool... **verify the arithmetic actually feeds `128*5*5`** — pick the input size that makes it true (input 3×27×27 → conv1 27 → avg 13 → conv2(k5) 9 → avg 4 → 128*4*4 mismatch… the implementer must solve the shape math and state it in a comment; if `128*5*5` needs input 3×26×26 via conv2 no-pad arithmetic, document the chosen size); (2) add `import flashruntime.torch as ft` + `prepare` + `checkpoint` + `log_metrics` — the diff shown as a diff; (3) `flash.submit(fr_torch.ddp("train.py", nproc_per_node=2))`; (4) crash it with `--kill-at-step` and rerun with `max_restarts=1` — one call, auto-resume; (5) "watch it live" — the viewer section with what each panel shows. The full final script is downloadable/copyable and must be runnable on CPU.
- Guides: split/adapt from the existing bring-your-code.md, per framework, each ending with "adding another framework" pointing at the adapter pattern (extensibility as documentation).
- Reference pages: exact signatures with one-line "why it exists" per function; CLI page mirrors `--help`.
- `concepts/architecture.md`: the four axes, leases, manifests, recovery — with one ASCII diagram each, sourced from AGENTS/ADR-0003 (public content only).

- [ ] Steps: write pages → `python scripts/build_docs.py --check` green → doc tests green → full suite → **Commit** — `docs(site): tutorials (ConvNet centerpiece), guides, concepts, reference`

---

### Task 10: Benchmark suite (Workstream D)

**Files:**
- Create: `benchmarks/__init__.py`, `benchmarks/__main__.py`, `benchmarks/registry.py`, `benchmarks/report.py`, `benchmarks/scenarios/{launch_overhead,loop_overhead,recovery_economics,hpo_sweep,adoption_cost}.py`, `benchmarks/requirements.txt` (ray[tune] pinned, accelerate pinned — optional comparators), `benchmarks/results/baseline-<hostname>.json` (measured in Step 4)
- Modify: `docs/site/benchmarks.md` (created here, added to `_nav.yml`), `scripts/build_docs.py` (tiny hook: render bench tables from the baseline JSON), `.github/workflows/ci.yml` is T11 (smoke marker only here: a pytest `-m bench_smoke` test)
- Test: `tests/test_benchmarks.py`

**Interfaces:**
- `registry.SCENARIOS: dict[str, Scenario]`; `Scenario` protocol: `name`, `hypothesis` (one sentence shown in docs), `run(repeats: int) -> ResultRow`. `ResultRow` (pydantic): `scenario`, `unit`, `median`, `p10`, `p90`, `repeats`, `comparators: dict[str, float]`, `notes: list[str]`. `python -m benchmarks run --all|--scenario NAME [--repeats N]` → writes `benchmarks/results/<timestamp>.json` with schema `bench_v1`: `{schema, host: {os, cpu, cores, ram_gb, python, torch, flashruntime}, rows: [...]}` and prints a table.
- Scenario definitions (exact measurements, spec §5b):
  1. `launch_overhead`: median wall-clock of `flash.submit(ddp(user_pytorch, steps=1))` minus median of the identical bare `torchrun` invocation (same argv, `subprocess.run`). Unit: seconds.
  2. `loop_overhead`: steps/sec of `examples/user_pytorch` (200 steps, checkpoint_every=0 disabled vs =50) vs `examples/user_pytorch_vanilla` — reports ft-vs-vanilla ratio and per-checkpoint cost (unit: ms/checkpoint, measured as the delta divided by checkpoint count).
  3. `recovery_economics`: total wall-clock to a finished 80-step run when crashed at 40: (a) `max_restarts=1` auto-resume vs (b) simulated raw-torchrun behavior = rerun from step 0 after the crash. Unit: seconds saved + steps not recomputed.
  4. `hpo_sweep`: 8 sklearn trials — flashruntime sweep vs a plain `for` loop of `subprocess.run` vs (if `ray` importable) `ray.tune` same grid. Wall clock + peak RSS (`resource.getrusage(RUSAGE_CHILDREN)`), + `setup_loc` counted from committed snippet files in `benchmarks/scenarios/snippets/` (each comparator's minimal working setup, honestly written from its own docs, cited in a comment).
  5. `adoption_cost`: no timing — LOC diff (vanilla script → framework-ready) computed with difflib against the snippets; import-time (`python -c "import X"` timed, median of 5) and dependency count (`pip show`-walk or `importlib.metadata.requires`) for flashruntime vs ray vs accelerate (skip gracefully with a note when a comparator isn't installed).
- Honesty mechanics: every row carries `notes` (e.g. "ray not installed — comparator skipped"); the docs table renders exactly the JSON; `report.py` refuses to render a row whose repeats < 3 except in `--smoke` mode which labels tables "smoke run — not representative".
- Tests: registry lists 5 scenarios; each scenario's `run(repeats=1)` completes on CPU (mark the two torch ones `@pytest.mark.bench_smoke` and skip without torch); ResultRow schema round-trips; report renders a table from a fixture JSON; smoke rows are labeled.

- [ ] Steps: failing tests → RED → implement scenarios (each file ≤120 lines, hypothesis at top, measurement method in the docstring — a reader must be able to audit the methodology from the file alone) → GREEN → **Step 4: run the real baseline on this Mac** (`python -m benchmarks run --all --repeats 5`, comparators installed in a THROWAWAY venv copy if heavy — implementer's judgment; record honestly what ran and what was skipped) → commit the baseline JSON + regenerate docs → full suite → **Commit** — `feat(benchmarks): honest evaluation suite + measured baseline + docs page`

---

### Task 11: CI, release pipeline, community files, audit

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `flashruntime/py.typed`, `scripts/audit_secrets.sh`
- Modify: `pyproject.toml` (URLs incl. docs site, classifiers, keywords, package-data for py.typed + viewer/_docs)
- Test: local dry-runs (no GitHub execution possible here)

**Interfaces / exact contents:**
- `ci.yml`: jobs — `core-smoke` (3.10, `pip install .`, the pydantic-only import assert incl. `flashruntime.viewer`); `test` (matrix 3.10–3.13 × ubuntu/macos, `pip install -e ".[sklearn,dev,service]"` + CPU torch index, full pytest incl. e2e); `docs` (build `--check`); `bench-smoke` (`pytest -m bench_smoke` + `python -m benchmarks run --all --repeats 1 --smoke`). Concurrency-cancel on same ref; pip cache.
- `release.yml`: on `push: tags: v*` — build (`python -m build`), `pypa/gh-action-pypi-publish` with trusted publishing (environment `pypi`, `id-token: write`, note in comments: user must register the publisher on PyPI once), then docs build → `actions/deploy-pages`. Manual `workflow_dispatch` for pages-only.
- `audit_secrets.sh`: greps full history (`git log -p`) and worktree for `rpa_[A-Za-z0-9]{20,}`, `AKIA[0-9A-Z]{16}`, `ghp_[A-Za-z0-9]{36}`, `-----BEGIN.*PRIVATE KEY`, `sk-[A-Za-z0-9]{20,}`; excludes `.env` (untracked anyway — assert it IS untracked and fail loudly if ever tracked); exit 1 on any hit. **Run it and paste the output in the report.**
- `CONTRIBUTING.md`: dev setup (uv), test tiers (unit / e2e / integration / bench), adapter-in-one-file how-to (the extensibility promise as contributor docs), protocol additive-only rule, readability rule (§2b) stated as a review criterion. `SECURITY.md`: private disclosure email + fail-closed philosophy. `CHANGELOG.md`: 0.1.0 — Keep-a-Changelog format, the honest feature list.
- Local verification: `python -m build` succeeds; `pip install dist/*.whl` into a scratch venv → core smoke + `flashruntime submit --help` work; `bash scripts/audit_secrets.sh` clean; YAML parses (`python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"`).

- [ ] Steps: write files → run all local verifications (wheel install test is the acceptance-criterion-1 proof — paste output) → full suite → **Commit** — `chore(release): CI matrix, trusted-publishing pipeline, community files, secrets audit`

---

### Task 12: RunPod real-GPU validation (spends money — caps below)

**Files:**
- Create: `scripts/runpod_gpu_e2e.py`, `tests/test_gpu_e2e.py` (the file the pod runs; skips without CUDA locally)
- Modify: workspace `../PROGRESS.md` + `docs/site/get-started.md` ("validated on" note) after success
- Test: the run itself is the test.

**Interfaces / hard rules:**
- `tests/test_gpu_e2e.py`: `pytest.importorskip("torch")` + skip when `not torch.cuda.is_available()`. Asserts: 2-process nccl DDP on `examples/user_pytorch` completes (`torch.distributed` backend reported as nccl in metrics — add an optional `backend` key to the example's metrics for this); GPU kill-and-resume via `max_restarts=1` succeeds with `resumed_from == 40`; `_resolve_device` path exercised (model parameters on cuda — assert via a metrics key `device: "cuda:0"`). Benchmarks #2/#3 run once each on GPU, results saved.
- `scripts/runpod_gpu_e2e.py` (dev-only; `httpx` from the service extra): reads `RUNPOD_API_KEY` from env/.env (refuse to start if missing; never print it). Flow, each step logged with cost-relevant timestamps:
  1. Query RunPod for the cheapest available 2-GPU offer among (RTX 4090, RTX A5000, RTX 3090), community cloud, CUDA≥12 PyTorch image (`runpod/pytorch` current tag).
  2. Create pod (name `flashruntime-e2e-<ts>`, label it), poll to RUNNING with SSH details; overall deadline 30 min enforced by the script.
  3. Over SSH (`subprocess` + `ssh -o StrictHostKeyChecking=accept-new`, key from RunPod's returned credentials): upload the locally-built wheel + `examples/` + the two test files via `scp`; `pip install <wheel> pytest`; run `pytest tests/test_gpu_e2e.py -v` and `python -m benchmarks run --scenario loop_overhead --scenario recovery_economics --repeats 3`; scp back a `results/` bundle.
  4. **`finally`: terminate the pod; then list pods filtered by the label and terminate any stragglers.** Print total runtime + estimated cost (runtime × offer price).
  5. Write `scripts/runpod_results/<ts>/` (gitignored) + a short human summary to stdout.
- **Caps (Global Constraints):** one pod, ≤30 min hard deadline, abort-and-terminate on any failed step, target ≤$1.
- On success: append the validation note (GPU model, torch/CUDA versions, date, test list, cost) to `../PROGRESS.md` per its protocol and the one-line "Validated on real GPUs (2×<model>, nccl) — <date>" to `docs/site/get-started.md`; rebuild docs; commit.

- [ ] Steps: write both files → **dry-run mode** (`--plan-only`: query offers + print the chosen one and est. cost, create nothing) and show it → **execute for real** → verify results bundle asserts all green → terminate confirmed (list-pods shows none) → PROGRESS/docs updates → full local suite still green → **Commit** — `feat(gpu): RunPod validation harness + real 2-GPU nccl evidence`

---

### Task 13: Final assembly

**Files:** `README.md` (rewrite front section around: install → 60-second run → viewer screenshot placeholder → docs-site link → benchmarks table teaser — honest numbers from the baseline), `AGENTS.md` (current-state refresh), workspace `../PROGRESS.md` (full work-log entry per protocol), ledger close.

- [ ] Steps: author README/AGENTS updates (verify every claim against code/tests) → `pytest` full suite green → doc build green → **Commit** — `docs: 0.1.0 README + status refresh` → then the controller runs the final whole-branch review per the SDD skill.

---

## Self-Review Notes

- **Spec coverage:** §2b→Global Constraints + every reviewer prompt; A1→T1; A2→T11; A3→T2; A4→T4; A5→T3+T5; A6→T12; B→T6+T7; C→T8+T9; D→T10; acceptance 1→T11 wheel test, 2→T1, 3→T11, 4→T4, 5→T3/T5, 6→T6/T7, 7→T8/T9, 8→T12, 9→T11 audit, 10→T10, 11→every review.
- **Known open decisions left to implementers (bounded):** ConvNet input-size arithmetic (T9, must be solved and stated); RunPod API surface details (GraphQL vs REST — T12 implementer reads current RunPod docs via WebFetch/WebSearch and records the choice).
- **Type consistency:** `run.json` schema defined once (T3) and consumed by T4 (events append), T6 (`state.collect`), T7 (page fields); visual tokens defined in `viewer/page.py` (T7) and imported by `build_docs.py` (T8) — T8 runs after T7. Wait — T8 imports tokens from `page.py`, which is created in T7: **T7 must precede T8** (it does).
- **Placeholder scan:** T5's `--watch` placeholder is explicitly temporary and removed in T7 (stated in both tasks). No TBDs remain.
