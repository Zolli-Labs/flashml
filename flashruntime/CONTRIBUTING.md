# Contributing to FlashRuntime

Thanks for helping build the open fault-tolerant distributed ML runtime.
This guide covers local setup, the test tiers, and the two rules every change
is reviewed against: **one-file extensibility** and **readable, reasoned code**.

FlashRuntime plans, launches, observes, and recovers ML jobs — it never
reimplements distributed training. The heavy lifting stays in established
libraries (PyTorch DDP/FSDP2/torchrun, Ray, Hugging Face); we integrate them.
Keep that boundary in mind when proposing changes.

## Dev setup

We use [uv](https://docs.astral.sh/uv/). From a clone:

```bash
uv venv && uv pip install -e ".[sklearn,dev,service]"
```

Extras are deliberately granular — install only what a change needs:

| extra      | pulls in                                | needed for                          |
| ---------- | --------------------------------------- | ----------------------------------- |
| (none)     | pydantic only                           | protocol, planner, leases, checkpoint, recovery, viewer |
| `sklearn`  | numpy, scikit-learn                     | the `flashml_workloads` task examples + sklearn e2e |
| `dev`      | pytest, pytest-asyncio, numpy, pyyaml, markdown | running the suite + building docs |
| `service`  | fastapi, uvicorn, httpx, kubernetes, minio, pyyaml | the HTTP coordinator + service tests |
| `k8s` / `artifacts` / `oss` | kubernetes / minio / oss2  | the respective backends             |

**The core stays pydantic-only.** `import flashruntime` and the planner /
leases / checkpoint / recovery / viewer subpackages must never require numpy,
torch, kubernetes, minio, or fastapi. CI's `core-smoke` job enforces this — if
you add a heavy import to a core module, that job fails.

## Running the tests

There are four tiers. The default `pytest` run is the fast one:

```bash
pytest                    # unit + e2e (integration + bench_smoke deselected)
pytest -m integration     # needs docker / kubernetes / minio (auto-skips without them)
pytest -m bench_smoke     # benchmark scenarios run at 1 repeat (asserts they RUN, not a number)
python scripts/build_docs.py --check   # docs build + every internal link resolves
```

- **unit** — no infrastructure, runs everywhere in seconds.
- **e2e** (`tests/test_examples_e2e.py`) — real user code operated end to end
  (the sklearn sweep, DDP on CPU via torchrun, kill-and-resume). Each test
  `importorskip`s its dependency, so they skip cleanly if an extra is missing.
- **integration** — opt-in; lives in `tests/integration/`, auto-skips without
  the external service.
- **bench** — `python -m benchmarks run --all --repeats 5` produces the real
  measured baseline; `--smoke` (1 repeat) just proves the harness runs.

New behavior needs a test. Write it first (red), then make it green — see
`superpowers:test-driven-development` if you use the skill suite.

## Adding support for another framework — one adapter file

This is the extensibility promise, and it is a **design test we enforce in
review**: teaching FlashRuntime to drive a new framework must stay *one small
file in `flashruntime/integrations/`* — never a core change.

An adapter is a pure function that returns a `CommandWorkload`. It imports no
framework code; the contract with the user's script is convention only (CLI
flags in, `metrics.json`/checkpoints out). The whole sklearn adapter is ~30
lines:

```python
# flashruntime/integrations/yourframework.py
from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source

def train(script: str, *, source: str = ".", python: str = "python", **params) -> CommandWorkload:
    """Return a workload that runs the user's script. No import of the
    framework here — the user's script owns the model."""
    command = [python, script]
    for key in sorted(params):
        command += [f"--{key}", "{" + key + "}"]
    return CommandWorkload(
        command=command,
        source=Source(path=source),
        task_params=[params],
        mode="independent_tasks",            # or "coordinated_training" for DDP/FSDP
        outputs=OutputSpec(collect=["metrics.json"], primary_metric="accuracy", maximize=True),
    )
```

Read `flashruntime/integrations/sklearn.py` (independent tasks) and
`flashruntime/integrations/pytorch.py` (coordinated training via torchrun) as
the two reference shapes, add a one-file test alongside the existing
`tests/test_integrations.py`, and you're done. If a new framework needs a
*core* change to work, that's a bug in the boundary — open an issue first so we
can fix the seam instead of special-casing.

The viewer and benchmarks follow the same rule: the viewer consumes only the
versioned `run.json` contract (any launcher that writes it renders), and a new
benchmark scenario is one registry entry in `benchmarks/scenarios/`, never an
edit to a monolith.

## Protocol changes are additive-only

The wire-visible protocol (`flashruntime/protocol/`) is versioned and imported
by the sibling repos (flashnode, flashml-cloud). Within a schema version:

- **Add** optional fields, new event types, new enum members — always fine.
- **Never** remove or repurpose a field, tighten a type, or change a field's
  meaning within a version. That's a breaking change → a new schema version.
- Security-relevant fields **fail closed**: a missing/invalid value denies, it
  does not default-allow.

If you're unsure whether a change is additive, it probably isn't — ask in the
issue.

## Code clarity is a review criterion (§2b)

Stated by the project as a release requirement, not a style preference — every
PR is reviewed against it explicitly:

- **Human-readable, reasoned code.** Every module and function exists for a
  reason a newcomer can read. Docstrings explain *why*; the code shows *how*.
  Prefer boring, explicit code over clever code. **If a reviewer needs the
  author to explain it, it fails review.**
- **Small files, one responsibility each.** The viewer page (self-contained
  HTML) is the single sanctioned large file; everything else stays modular.
- **Horizontal extensibility is preserved** (see the adapter rule above).

## Pull request checklist

- [ ] `pytest` is green (and `pytest -m bench_smoke` if you touched benchmarks).
- [ ] `python scripts/build_docs.py --check` passes if you touched docs.
- [ ] `bash scripts/audit_secrets.sh` is clean — never commit a key; secrets
      live only in the gitignored `.env`.
- [ ] New behavior has a test; new framework support is one adapter file.
- [ ] Public/wire schema changes are additive within their version.
- [ ] A newcomer could read your diff without you narrating it.

By contributing you agree your work is licensed under the repository's
[Apache-2.0](LICENSE) license.
