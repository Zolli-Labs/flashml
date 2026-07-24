"""fault_recovery_matrix — does automated recovery do the RIGHT typed thing
across a spread of failure modes, or only on the one crash we rehearsed?

HYPOTHESIS: flashruntime handles a diverse fault matrix — a deterministic bug,
a mid-run crash, a hang, a kill inside the checkpoint write window, and a
corrupt checkpoint — CORRECTLY and without a human: fail-fast on the bug,
auto-resume from the newest VALID checkpoint on the rest. A bare torchrun does
none of it (every fault needs a person).

MEASUREMENT METHOD (auditable from this file alone):
  Five injected faults, each a real local ``flash.submit`` run. Correctness is
  COUNTED from observable run state — the terminal state, the number of attempts,
  and ``trials[0].resumed_from`` (which manifest step the resume actually
  restored) — never a ``try/assert`` that predetermines the score. The headline
  ``median`` is that count out of five; a case that measures as mishandled lowers
  it and is a FINDING, not a test to fix. Per-case verdicts ship in ``notes``.

  Comparator: the same mid-run crash (case b) under BARE torchrun argv (from the
  pytorch adapter) — it dies with no retry, and its recovery cost is MODELLED as
  ``t_crash + t_full`` (no checkpoint ⇒ rerun from zero), labelled modelled, the
  same house convention as recovery_economics. A missing torchrun is a note, not
  a skip: this scenario measures flashruntime, which needs no torch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmarks import faults
from benchmarks._util import bench_env, ensure_venv_on_path, median, percentile, timed
from benchmarks.schema import ResultRow
from flashruntime.checkpoint.local import latest_valid_manifest

name = "fault_recovery_matrix"
hypothesis = "Automated recovery does the right typed thing across a fault matrix — fail-fast on a deterministic bug, resume from the newest valid checkpoint on transient crashes/hangs/mid-write kills/corruption — with zero human interventions where a bare torchrun needs one per fault."

STEPS, EVERY = 8, 2


def _wl(script_dir: Path, script: Path, *, steps: int = STEPS, every: int = EVERY):
    from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source

    return CommandWorkload(
        command=[sys.executable, str(script), "--steps", str(steps), "--checkpoint-every", str(every)],
        source=Source(path=str(script_dir)),
        outputs=OutputSpec(collect=["metrics.json"]),
    )


def _resumed(run) -> int | None:
    return run.trials[0].get("resumed_from") if run.trials else None


def _ckpt(out: Path) -> Path:
    return out / "local" / "ckpt"  # launcher exports FLASHML_CKPT_DIR = <out>/local/ckpt


# Each case returns (correct, verdict, recovery_s | None, steps_preserved | None).
def _case_import_error(tmp: Path):
    import flashruntime as flash

    script = faults.write_crashy_trainer(tmp, steps=STEPS, checkpoint_every=EVERY, crash="import_error")
    _, run = timed(lambda: flash.submit(_wl(tmp, script), output_dir=tmp / "run", max_restarts=3))
    n = len(run.attempts)
    ok = run.state.value == "FAILED" and n == 1
    return ok, f"(a) import_error → FAIL_JOB fast-stop {_m(ok)} ({n} attempt, 0 restarts burned)", None, None


def _case_systemexit_mid(tmp: Path):
    import flashruntime as flash

    script = faults.write_crashy_trainer(tmp, steps=STEPS, checkpoint_every=EVERY, crash="systemexit_mid")
    t, run = timed(lambda: flash.submit(_wl(tmp, script), output_dir=tmp / "run", max_restarts=1))
    r = _resumed(run)
    ok = run.state.value == "SUCCEEDED" and bool(r)
    return ok, f"(b) systemexit_mid → auto-resume {_m(ok)} (resumed_from={r}, {len(run.attempts)} attempts)", t, r


def _case_hang(tmp: Path):
    import flashruntime as flash

    script = faults.write_crashy_trainer(tmp, steps=STEPS, checkpoint_every=EVERY, crash="hang_after_step")
    ck = _ckpt(tmp / "run")
    t0 = time.perf_counter()
    run = flash.submit(_wl(tmp, script), output_dir=tmp / "run", max_restarts=1, wait=False)
    fired = faults.kill_child(run, when=lambda: any(ck.glob("step-*/manifest.json")), timeout_s=30)
    run.wait(timeout=90)
    t, r = time.perf_counter() - t0, _resumed(run)
    ok = fired and run.state.value == "SUCCEEDED" and bool(r)
    return ok, f"(c) hang+external SIGKILL → resume {_m(ok)} (fired={fired}, resumed_from={r})", t, r


def _case_write_window(tmp: Path):
    import flashruntime as flash

    script = faults.write_crashy_trainer(tmp, steps=24, checkpoint_every=1, crash=None)
    ck = _ckpt(tmp / "run")

    def in_write_window() -> bool:
        # the write window IS "a step-* dir exists but its manifest.json does
        # not yet" (parts are written first); require a complete manifest too so
        # the resume has a verified earlier target.
        partial = any(d.is_dir() and not (d / "manifest.json").exists() for d in ck.glob("step-*"))
        return partial and latest_valid_manifest(ck) is not None

    t0 = time.perf_counter()
    run = flash.submit(_wl(tmp, script, steps=24, every=1), output_dir=tmp / "run", max_restarts=1, wait=False)
    fired = faults.kill_child(run, when=in_write_window, timeout_s=30)
    run.wait(timeout=90)
    t, r = time.perf_counter() - t0, _resumed(run)
    ok = fired and run.state.value == "SUCCEEDED" and bool(r)
    return ok, f"(d) mid-write SIGKILL → resume from verified earlier step {_m(ok)} (fired={fired}, resumed_from={r})", t, r


def _case_corruption(tmp: Path):
    import flashruntime as flash

    script = faults.write_crashy_trainer(tmp, steps=STEPS, checkpoint_every=EVERY, crash=None)
    wl = _wl(tmp, script)
    run1 = flash.submit(wl, output_dir=tmp / "run")
    ck = _ckpt(tmp / "run")
    before = latest_valid_manifest(ck)
    corrupt_step = before.step if before else None
    hit = faults.corrupt_newest_part(ck)
    t, run2 = timed(lambda: flash.submit(wl, output_dir=tmp / "run"))
    r = _resumed(run2)
    ok = (run1.state.value == "SUCCEEDED" and run2.state.value == "SUCCEEDED" and hit is not None
          and corrupt_step is not None and bool(r) and r < corrupt_step)
    return ok, f"(e) corrupt newest part → resume from earlier valid {_m(ok)} (corrupt@{corrupt_step}, resumed_from={r})", t, r


def _m(ok: bool) -> str:
    return "✓" if ok else "✗"


def _torchrun_modelled(tmp: Path):
    """Case (b) under bare torchrun argv: it dies, no retry; recovery MODELLED
    as t_crash + t_full. Returns (modelled_seconds | None, note)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return None, "torchrun comparator skipped — torch not installed (this scenario measures flashruntime, which needs no torch)"
    if shutil.which("torchrun") is None:
        return None, "torchrun comparator skipped — torchrun not on PATH"
    from flashruntime.integrations import pytorch as fr_torch

    crash_s = faults.write_crashy_trainer(tmp, steps=STEPS, checkpoint_every=EVERY, crash="systemexit_mid")
    full_s = faults.write_crashy_trainer(tmp, steps=STEPS, checkpoint_every=EVERY, crash=None)
    env = bench_env()

    def argv(script: Path) -> list[str]:
        return fr_torch.ddp(str(script), nproc_per_node=1, script_args=f"--steps {STEPS} --checkpoint-every {EVERY}").argv()

    cd_crash, cd_full = tmp / "tr_crash", tmp / "tr_full"  # isolated cwds ⇒ isolated ckpt trees
    cd_crash.mkdir(parents=True, exist_ok=True)
    cd_full.mkdir(parents=True, exist_ok=True)
    c0 = time.perf_counter()
    crashed = subprocess.run(argv(crash_s), cwd=str(cd_crash), env=env, capture_output=True, text=True)
    t_crash = time.perf_counter() - c0
    f0 = time.perf_counter()
    full = subprocess.run(argv(full_s), cwd=str(cd_full), env=env, capture_output=True, text=True)
    t_full = time.perf_counter() - f0
    if full.returncode != 0:  # a broken comparator must not fabricate a number
        return None, f"torchrun comparator skipped — clean full run exited {full.returncode}:\n{full.stderr[-400:]}"
    modelled = t_crash + t_full
    note = (f"torchrun comparator (case b, bare torchrun argv, nproc=1): crash run exits {crashed.returncode} "
            f"with NO retry; recovery modelled as t_crash+t_full={modelled:.2f}s (no checkpoint ⇒ rerun from zero) "
            "— labelled modelled, house convention")
    return modelled, note


def run(repeats: int) -> ResultRow:
    ensure_venv_on_path()
    import tempfile

    counts: list[float] = []
    recov: list[float] = []
    preserved: list[float] = []
    modelled: list[float] = []
    verdicts: list[str] = []
    tr_note = "torchrun comparator not run"
    for rep in range(repeats):
        with tempfile.TemporaryDirectory(prefix="fault-matrix-") as td:
            root = Path(td)
            cases = [
                _case_import_error(root / "a"),
                _case_systemexit_mid(root / "b"),
                _case_hang(root / "c"),
                _case_write_window(root / "d"),
                _case_corruption(root / "e"),
            ]
            counts.append(float(sum(1 for ok, *_ in cases)))
            for _ok, _v, rs, sp in cases:
                if rs is not None:
                    recov.append(rs)
                if sp is not None:
                    preserved.append(float(sp))
            if rep == 0:
                verdicts = [v for _ok, v, _rs, _sp in cases]
            m, tr_note = _torchrun_modelled(root / "tr")
            if m is not None:
                modelled.append(m)

    correct = median(counts)
    comparators = {
        # cases the automation did NOT resolve correctly on its own (0 ideal)
        "manual_interventions_flash": round(5.0 - correct, 1),
        "manual_interventions_torchrun_modelled": 5.0,  # torchrun: no classify, no resume — all 5 need a human
        "mean_recovery_s": round(sum(recov) / len(recov), 3) if recov else 0.0,
        "mean_steps_preserved": round(sum(preserved) / len(preserved), 1) if preserved else 0.0,
    }
    if modelled:
        comparators["torchrun_recovery_modelled_s"] = round(sum(modelled) / len(modelled), 3)
    notes = verdicts + [
        "correctness is COUNTED from run state (terminal state, attempts count, trials[0].resumed_from), never asserted",
        "manual_interventions_flash = cases the automation did NOT resolve correctly; torchrun modelled at 5 (no "
        "typed classification, no auto-resume — every fault needs a human to notice and resubmit)",
        tr_note,
    ]
    return ResultRow(
        scenario=name,
        section="resilience",
        unit="correct/5",
        median=correct,
        p10=percentile(counts, 0.1),
        p90=percentile(counts, 0.9),
        repeats=repeats,
        comparators=comparators,
        notes=notes,
    )
