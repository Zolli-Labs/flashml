# tests/test_auto_recovery.py
"""Automatic fault tolerance (max_restarts): the first real caller of the
recovery package. A FAILED launch is translated to FailureSignals, classified,
and run against the versioned policy — deterministic bugs fail fast, transient
crashes resume from the job-scoped checkpoint without a human."""

from __future__ import annotations

import json
import sys
import textwrap


def _write_script(tmp_path, body: str) -> str:
    src = tmp_path / "userproj"
    src.mkdir(exist_ok=True)
    (src / "train.py").write_text(textwrap.dedent(body))
    return str(src)


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


# --- named-marker anchoring (fix round 1): a marker counts only on a traceback
# TERMINAL line, never as an incidental substring in a prose log line ----------

def test_incidental_marker_substring_is_not_deterministic():
    """The headline false positive: a transient crash whose log INCIDENTALLY
    carries an exception name in prose — the canonical real case is the startup
    warning `ImportError: flash_attn not available, falling back to eager`,
    which real loggers emit behind a timestamp/level prefix — then dies
    transiently under torchrun. The bare-substring scan misread the prose as a
    deterministic APPLICATION_ERROR → FAIL_JOB → no resume (worst direction).
    Line-anchoring must classify it as a transient WORKER_CRASH."""
    from flashruntime.protocol.v1alpha1 import FailureClass
    from flashruntime.recovery import classify
    from flashruntime.recovery.signals import from_local_launch

    log = (
        "2026-07-22 10:01:03 WARNING ImportError: flash_attn not available, "
        "falling back to eager attention\n"
        "2026-07-22 10:05:41 INFO step 500 loss=0.42\n"
        "Traceback (most recent call last):\n"
        '  File ".../elastic/agent/server/api.py", line 733, in run\n'
        "    raise ChildFailedError(...)\n"
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n"
        "============================================================\n"
        "train.py FAILED\n"
    )
    sig = from_local_launch(1, log)
    assert sig.exit_deterministic is False
    assert classify(sig) is FailureClass.WORKER_CRASH


def test_true_traceback_terminal_line_is_deterministic():
    """A real CPython traceback whose TERMINAL line begins with the exception
    type (`ModuleNotFoundError: ...`, optionally dotted with its module) stays a
    deterministic APPLICATION_ERROR — line-anchoring drops only the prose false
    positive, never a genuine traceback terminal."""
    from flashruntime.protocol.v1alpha1 import FailureClass
    from flashruntime.recovery import classify
    from flashruntime.recovery.signals import from_local_launch

    log = (
        "Traceback (most recent call last):\n"
        '  File "train.py", line 1, in <module>\n'
        "    import x\n"
        "ModuleNotFoundError: No module named 'x'\n"
    )
    sig = from_local_launch(1, log)
    assert sig.exit_deterministic is True
    assert classify(sig) is FailureClass.APPLICATION_ERROR


def test_torchrun_childfailederror_wrapper_is_not_deterministic():
    """The e2e-critical carve-out: torchrun/elastic wraps EVERY worker death
    (transient ones included) in its own ChildFailedError traceback whose
    terminal line is the launcher's stack, not a user bug — and none of the
    deterministic markers appear at a line start. It must stay transient so a
    resumable crash retries; the kill-and-resume e2e depends on this."""
    from flashruntime.protocol.v1alpha1 import FailureClass
    from flashruntime.recovery import classify
    from flashruntime.recovery.signals import from_local_launch

    log = (
        "Traceback (most recent call last):\n"
        '  File ".../elastic/agent/server/api.py", line 733, in run\n'
        "    raise ChildFailedError(...)\n"
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n"
        "============================================================\n"
        "train.py FAILED\n"
        "  exitcode  : 3 (pid: 12345)\n"
        "  error_file: <N/A>\n"
    )
    sig = from_local_launch(1, log)
    assert sig.exit_deterministic is False
    assert classify(sig) is FailureClass.WORKER_CRASH
