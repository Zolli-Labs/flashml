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
