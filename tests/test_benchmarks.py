"""Tests for the honest evaluation suite (Task 10).

Two layers:
  * framework pieces (registry / report / schema) — plain, fast, TDD'd here;
  * scenario smoke tests — assert each scenario RUNS at repeats=1 (never assert
    a measured number: the whole point of the suite is that numbers are
    measured, not baked into an assertion). Marked ``bench_smoke`` so CI can
    select them (`-m bench_smoke`) and the default unit run skips them (they
    spawn real subprocesses / torchrun and take seconds).
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks import registry, report
from benchmarks.registry import ResultRow


# --------------------------------------------------------------------------
# registry + schema
# --------------------------------------------------------------------------
def test_registry_lists_all_scenarios():
    assert set(registry.SCENARIOS) == {
        "launch_overhead",
        "loop_overhead",
        "recovery_economics",
        "hpo_sweep",
        "adoption_cost",
        "fault_recovery_matrix",
        "checkpoint_integrity",
        "crash_storm",
    }


def test_every_scenario_satisfies_the_protocol():
    for name, scenario in registry.SCENARIOS.items():
        assert scenario.name == name
        assert isinstance(scenario.hypothesis, str) and scenario.hypothesis
        assert callable(scenario.run)
        assert isinstance(registry.SCENARIOS[name], registry.Scenario)


def test_resultrow_round_trips():
    row = ResultRow(
        scenario="launch_overhead",
        unit="seconds",
        median=0.42,
        p10=0.40,
        p90=0.45,
        repeats=5,
        comparators={"bare_torchrun_s": 2.6},
        notes=["ray not installed — comparator skipped"],
    )
    again = ResultRow.model_validate(json.loads(row.model_dump_json()))
    assert again == row
    assert again.comparators["bare_torchrun_s"] == 2.6


def test_resultrow_defaults_are_empty_containers():
    row = ResultRow(scenario="x", unit="seconds", median=1.0, p10=1.0, p90=1.0, repeats=3)
    assert row.comparators == {}
    assert row.notes == []


# --------------------------------------------------------------------------
# report: honesty mechanics
# --------------------------------------------------------------------------
_FIXTURE = {
    "schema": "bench_v1",
    "host": {"os": "macOS", "cpu": "Apple M-series", "cores": 8, "ram_gb": 16,
             "python": "3.12", "torch": "2.13.0", "flashruntime": "0.1.0"},
    "rows": [
        {"scenario": "launch_overhead", "unit": "seconds", "median": 0.42,
         "p10": 0.40, "p90": 0.45, "repeats": 5, "comparators": {"bare_torchrun_s": 2.6},
         "notes": ["measured on CPU"]},
    ],
}


def test_report_renders_a_table_from_a_fixture():
    table = report.render_markdown(_FIXTURE["rows"])
    assert "launch_overhead" in table
    assert "0.42" in table
    assert "| " in table  # a markdown table


def test_report_refuses_rows_with_too_few_repeats_outside_smoke():
    rows = [{"scenario": "x", "unit": "seconds", "median": 1.0, "p10": 1.0,
             "p90": 1.0, "repeats": 2, "comparators": {}, "notes": []}]
    with pytest.raises(ValueError, match="repeats"):
        report.render_markdown(rows)


def test_report_smoke_mode_renders_and_labels():
    rows = [{"scenario": "x", "unit": "seconds", "median": 1.0, "p10": 1.0,
             "p90": 1.0, "repeats": 1, "comparators": {}, "notes": []}]
    table = report.render_markdown(rows, smoke=True)
    assert "smoke run — not representative" in table
    assert "x" in table


def test_report_full_document_renders_host_block():
    text = report.render_document(_FIXTURE)
    assert "launch_overhead" in text
    assert "flashruntime" in text  # host row present


# --------------------------------------------------------------------------
# scenario smoke tests — assert they RUN, never assert numbers
# --------------------------------------------------------------------------
def _assert_runs(name: str):
    row = registry.SCENARIOS[name].run(repeats=1)
    assert isinstance(row, ResultRow)
    assert row.scenario == name
    assert row.repeats == 1
    assert row.unit


@pytest.mark.bench_smoke
def test_launch_overhead_runs():
    pytest.importorskip("torch")
    import shutil
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")
    _assert_runs("launch_overhead")


@pytest.mark.bench_smoke
def test_loop_overhead_runs():
    pytest.importorskip("torch")
    _assert_runs("loop_overhead")


@pytest.mark.bench_smoke
def test_recovery_economics_runs():
    pytest.importorskip("torch")
    import shutil
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")
    _assert_runs("recovery_economics")


@pytest.mark.bench_smoke
def test_hpo_sweep_runs():
    pytest.importorskip("sklearn")
    _assert_runs("hpo_sweep")


@pytest.mark.bench_smoke
def test_adoption_cost_runs():
    _assert_runs("adoption_cost")


# --------------------------------------------------------------------------
# Task 2 — the `bench_stress` marker (registered AND deselected by default,
# exactly like bench_smoke; introspected from pytest's own config so a drop of
# either half — the markers list or the addopts deselection — fails here)
# --------------------------------------------------------------------------
def test_bench_stress_marker_is_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    assert any(m.startswith("bench_stress") for m in markers), markers


def test_bench_stress_is_deselected_by_default(pytestconfig):
    # the default addopts must exclude bench_stress just as it excludes bench_smoke
    addopts = " ".join(pytestconfig.getini("addopts"))
    assert "not bench_stress" in addopts, addopts
    assert "not bench_smoke" in addopts, addopts  # unchanged alongside it


# --------------------------------------------------------------------------
# Task 1 — the additive `section` field (resilience rows live beside perf ones)
# --------------------------------------------------------------------------
def test_resultrow_section_defaults_to_performance():
    row = ResultRow(scenario="x", unit="seconds", median=1.0, p10=1.0, p90=1.0, repeats=3)
    assert row.section == "performance"


def test_old_baseline_json_without_section_still_validates():
    # a pre-Task-1 row (no `section` key) must still load — additive default
    legacy = {"scenario": "launch_overhead", "unit": "seconds", "median": 0.42,
              "p10": 0.4, "p90": 0.45, "repeats": 5, "comparators": {}, "notes": []}
    again = ResultRow.model_validate(legacy)
    assert again.section == "performance"


def test_resultrow_section_round_trips_as_resilience():
    row = ResultRow(scenario="fault_recovery_matrix", section="resilience",
                    unit="correct/5", median=5.0, p10=5.0, p90=5.0, repeats=1)
    again = ResultRow.model_validate(json.loads(row.model_dump_json()))
    assert again.section == "resilience"


# --------------------------------------------------------------------------
# Task 1 — faults.py helpers (unit; the three T2/T3 consume verbatim)
# --------------------------------------------------------------------------
def test_write_crashy_trainer_materializes_and_compiles(tmp_path):
    from benchmarks import faults

    for crash in (None, "import_error", "systemexit_mid", "hang_after_step"):
        script = faults.write_crashy_trainer(tmp_path, steps=8, checkpoint_every=2, crash=crash)
        assert isinstance(script, Path) and script.is_file()
        # compiles as valid Python (a broken heredoc would fail here, not at run)
        py_compile.compile(str(script), doraise=True)


def test_write_crashy_trainer_import_error_names_a_missing_module(tmp_path):
    from benchmarks import faults

    script = faults.write_crashy_trainer(tmp_path, steps=8, checkpoint_every=2, crash="import_error")
    text = script.read_text()
    assert "import definitely_not_a_module" in text  # top-level ⇒ dies at import


def test_write_crashy_trainer_runs_to_completion_and_checkpoints(tmp_path):
    # a crash=None script actually runs (stdlib+ft only, no torch) and leaves a
    # valid manifest tree — proves the resume substrate the matrix depends on.
    from benchmarks import faults
    from flashruntime.checkpoint.local import latest_valid_manifest

    script = faults.write_crashy_trainer(tmp_path, steps=8, checkpoint_every=2, crash=None)
    ckpt = tmp_path / "ckpt"
    env = {**os.environ, "FLASHML_CKPT_DIR": str(ckpt), "FLASHML_OUTPUT_DIR": str(tmp_path)}
    proc = subprocess.run([sys.executable, str(script), "--steps", "8", "--checkpoint-every", "2"],
                          cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    manifest = latest_valid_manifest(ckpt)
    assert manifest is not None and manifest.step == 8


def test_kill_child_fires_on_a_live_subprocess():
    from benchmarks import faults

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    class _StubRun:  # the polling seam: kill_child only needs `.attempts`
        attempts = [{"pid": str(proc.pid), "state": "RUNNING"}]

    try:
        fired = faults.kill_child(_StubRun(), when=lambda: True, timeout_s=3.0)
        assert fired is True
        assert proc.wait(timeout=3) is not None  # actually died
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)


def test_kill_child_does_not_fire_when_predicate_stays_false():
    from benchmarks import faults

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    class _StubRun:
        attempts = [{"pid": str(proc.pid), "state": "RUNNING"}]

    try:
        fired = faults.kill_child(_StubRun(), when=lambda: False, timeout_s=0.3)
        assert fired is False
        assert proc.poll() is None  # still alive — nothing was signalled
    finally:
        proc.kill()
        proc.wait(timeout=3)


def test_kill_child_never_signals_a_finished_attempts_reused_pid():
    # pid-reuse guard (reviewer-mandated): after the launched child self-exits,
    # the driver settles its attempt row (state != RUNNING / finished_at set) and
    # the OS may hand that pid number to an UNRELATED live process. kill_child
    # must refuse to signal such a row — a stub whose newest attempt is finished,
    # with a real foreign live process sitting on that pid, must return False and
    # leave the foreign process untouched (never a false "fired").
    from benchmarks import faults

    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    class _StubRun:  # newest attempt already terminated; its pid now belongs to `foreign`
        attempts = [{"pid": str(foreign.pid), "state": "SUCCEEDED", "finished_at": 123.0}]

    try:
        fired = faults.kill_child(_StubRun(), when=lambda: True, timeout_s=0.3)
        assert fired is False           # the finished attempt is never signalled
        assert foreign.poll() is None   # the foreign pid-holder is untouched
    finally:
        foreign.kill()
        foreign.wait(timeout=3)


def test_corrupt_newest_part_disqualifies_the_newest_manifest(tmp_path):
    from benchmarks import faults
    from flashruntime.checkpoint.local import latest_valid_manifest, write_manifest

    for step in (2, 4):
        d = tmp_path / f"step-{step:06d}"
        d.mkdir()
        (d / "model.pt").write_bytes(b"\x5a" * 4096)
        write_manifest(d, job_id="local", attempt_id="local", step=step)

    assert latest_valid_manifest(tmp_path).step == 4  # newest before corruption

    corrupted = faults.corrupt_newest_part(tmp_path)
    assert corrupted is not None and corrupted.is_file()
    assert corrupted.parent.name == "step-000004"  # it hit the newest step's part
    # newest manifest now fails hash verification ⇒ recovery falls back to step 2
    assert latest_valid_manifest(tmp_path).step == 2


def test_corrupt_newest_part_returns_none_on_empty_tree(tmp_path):
    from benchmarks import faults

    assert faults.corrupt_newest_part(tmp_path) is None


# --------------------------------------------------------------------------
# Task 1 — the S1 scenario smoke (spawns real runs; bench_smoke)
# --------------------------------------------------------------------------
def test_registry_lists_fault_recovery_matrix():
    assert "fault_recovery_matrix" in registry.SCENARIOS


@pytest.mark.bench_smoke
def test_fault_recovery_matrix_runs():
    from benchmarks.scenarios import fault_recovery_matrix

    row = fault_recovery_matrix.run(repeats=1)
    assert isinstance(row, ResultRow)
    assert row.scenario == "fault_recovery_matrix"
    assert row.section == "resilience"
    assert row.unit == "correct/5"
    assert len(row.notes) >= 5  # one verdict line per case


# --------------------------------------------------------------------------
# Task 2 — the S2 checkpoint_integrity chaos scenario (bench_smoke N=3 /
# bench_stress N=20; spawns real runs + a torch naive comparator)
# --------------------------------------------------------------------------
def test_registry_lists_checkpoint_integrity():
    assert "checkpoint_integrity" in registry.SCENARIOS


@pytest.mark.bench_smoke
def test_checkpoint_integrity_smoke_runs():
    # smoke variant: N=3 iterations. The naive comparator writes/loads a real
    # torch archive, so torch is required to exercise the full row.
    pytest.importorskip("torch")
    from benchmarks.scenarios import checkpoint_integrity

    row = checkpoint_integrity.run(repeats=3)
    assert isinstance(row, ResultRow)
    assert row.scenario == "checkpoint_integrity"
    assert row.section == "resilience"
    assert row.unit == "integrity_rate"
    assert row.median <= 1.0                          # a rate, MEASURED (never > 1)
    assert row.comparators["iterations"] == 3
    assert "torn_writes_hit" in row.comparators       # kills that landed in a write window
    assert "naive_torch_save_failure_rate" in row.comparators
    assert row.notes                                  # documents technique + observed modes


@pytest.mark.bench_stress
def test_checkpoint_integrity_stress_runs():
    # stress variant: the full N=20 chaos loop. Deselected by default (bench_stress).
    pytest.importorskip("torch")
    from benchmarks.scenarios import checkpoint_integrity

    row = checkpoint_integrity.run(repeats=20)
    assert isinstance(row, ResultRow)
    assert row.scenario == "checkpoint_integrity"
    assert row.section == "resilience"
    assert row.unit == "integrity_rate"
    assert row.median <= 1.0
    assert row.comparators["iterations"] == 20


# --------------------------------------------------------------------------
# Task 3 — the S3 crash_storm goodput scenario (bench_smoke 4-trial /
# bench_stress full 16-trial; spawns a real fan-out of crash+resume runs).
# The scenario needs no torch — the crash trainer is stdlib+flashruntime only.
# --------------------------------------------------------------------------
def test_registry_lists_crash_storm():
    assert "crash_storm" in registry.SCENARIOS


@pytest.mark.bench_smoke
def test_crash_storm_smoke_runs():
    # smoke variant: a 4-trial storm (evens 0,2 crash on their fresh attempt).
    # This test asserts the row SHAPE + counted fields present and within bounds —
    # NOT a measured outcome. Two distinct kinds of assertion live here:
    #   * completed <= 4 and 0 <= goodput <= 1 are BOUNDS on MEASURED numbers —
    #     completion is an observed outcome (if the storm resumes fewer than 4
    #     trials, that ships as the measured number, so we bound, never ==);
    #   * crashed_first_attempt == 2 is a MECHANISM check, not a measured-outcome
    #     assertion: the crash arming is DETERMINISTIC (even trial index ⇒ armed),
    #     so exactly the two even trials (0, 2) must have crashed-then-resumed.
    #     Asserting 2 tests that the fan-out wired the per-trial crash correctly,
    #     the same way faults.py's unit tests assert the injection fired.
    from benchmarks.scenarios import crash_storm

    row = crash_storm._storm(n_trials=4, repeats=1)
    assert isinstance(row, ResultRow)
    assert row.scenario == "crash_storm"
    assert row.section == "resilience"
    assert row.unit == "completed/4"
    assert row.median <= 4.0                                   # completions COUNTED (bound, never ==)
    gp = row.comparators["goodput_fraction"]
    assert 0.0 <= gp <= 1.0                                    # a fraction, MEASURED
    assert row.comparators["crashed_first_attempt"] == 2.0    # MECHANISM check: evens 0,2 armed
    assert row.comparators["manual_interventions"] == 0.0     # derived (fully automatic), not asserted-as-outcome
    assert "wallclock_penalty_fraction" in row.comparators
    assert row.notes                                          # documents the goodput accounting


@pytest.mark.bench_stress
def test_crash_storm_full_runs():
    # full variant: the 16-trial storm (8 evens crash+resume). Deselected by
    # default (bench_stress) — it spawns 24 storm launches + 16 clean launches.
    from benchmarks.scenarios import crash_storm

    row = crash_storm._storm(n_trials=16, repeats=1)
    assert isinstance(row, ResultRow)
    assert row.scenario == "crash_storm"
    assert row.section == "resilience"
    assert row.unit == "completed/16"
    assert row.median <= 16.0                                 # completions COUNTED (bound)
    assert 0.0 <= row.comparators["goodput_fraction"] <= 1.0
    assert row.comparators["crashed_first_attempt"] <= 8.0    # at most the 8 even trials armed


# --------------------------------------------------------------------------
# Task 3 — the goodput accounting is a pure function (testable without running
# the storm). Pins the CONSERVATIVE executed-steps formula and its worked
# example, exactly the way checkpoint_integrity's _integrity is pinned.
# --------------------------------------------------------------------------
def test_goodput_worked_example_16_trials():
    from benchmarks.scenarios.crash_storm import _goodput

    # 8 crashed (steps 8, resumed_from 4) + 8 clean (steps 8, resumed_from 0):
    #   useful = 16*8 = 128;  executed = 8*(8+4) + 8*(8+0) = 96 + 64 = 160
    #   goodput = 128/160 = 0.8;  crashed_first_attempt = 8
    trials = [{"steps": 8, "resumed_from": 4}] * 8 + [{"steps": 8, "resumed_from": 0}] * 8
    frac, useful, executed, crashed = _goodput(trials)
    assert (useful, executed, crashed) == (128, 160, 8)
    assert frac == 0.8


def test_goodput_clean_trials_are_full_goodput():
    from benchmarks.scenarios.crash_storm import _goodput

    # no crashes ⇒ resumed_from 0 everywhere ⇒ executed == useful ⇒ 1.0
    frac, useful, executed, crashed = _goodput([{"steps": 8, "resumed_from": 0}] * 4)
    assert crashed == 0
    assert useful == executed == 32
    assert frac == 1.0


def test_goodput_is_nan_safe_on_empty_trials():
    from benchmarks.scenarios.crash_storm import _goodput

    # a storm that completed zero trials ⇒ 0 executed ⇒ 0.0 (never a fabricated 1.0)
    frac, useful, executed, crashed = _goodput([])
    assert (useful, executed, crashed) == (0, 0, 0)
    assert frac == 0.0


# --------------------------------------------------------------------------
# Task 2 (fix round 1) — integrity_rate is measured over TORN-WRITE HITS ONLY.
# Pure-function coverage of the rate computation, testable without running
# chaos: a window-missed iteration (the kill never fired — a clean uninterrupted
# run) must NEVER count toward the denominator as a trivial 1.0 success that
# would silently inflate the rate on a slower box.
# --------------------------------------------------------------------------
def test_integrity_rate_excludes_window_missed_iterations():
    from benchmarks.scenarios.checkpoint_integrity import _integrity

    # (fired, survived) per iteration. The middle row's kill never landed in a
    # window (fired=False) — it is EXCLUDED from the denominator entirely, not
    # counted as a free 1.0 success alongside the two real in-window kills.
    rate, hits, missed = _integrity([(True, True), (False, True), (True, True)])
    assert rate == 1.0
    assert hits == 2      # only the two in-window kills form the denominator
    assert missed == 1    # the window-missed row, reported separately


def test_integrity_rate_is_survived_hits_over_hits():
    from benchmarks.scenarios.checkpoint_integrity import _integrity

    rate, hits, missed = _integrity([(True, False), (True, True)])
    assert rate == 0.5    # one of two in-window kills survived
    assert hits == 2
    assert missed == 0


def test_integrity_rate_not_measurable_when_no_kill_landed():
    from benchmarks.scenarios.checkpoint_integrity import _integrity

    # all-missed: no kill ever landed in a window ⇒ the guarantee was never
    # tested, so the rate is NOT measurable. The helper returns 0.0 — NaN-safe,
    # and deliberately NOT a fabricated 1.0 — with hits=0 so the caller emits the
    # "not measurable this run" note.
    rate, hits, missed = _integrity([(False, True), (False, False)])
    assert hits == 0
    assert missed == 2
    assert rate == 0.0    # never a fabricated 1.0
