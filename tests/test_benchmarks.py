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

import pytest

from benchmarks import registry, report
from benchmarks.registry import ResultRow


# --------------------------------------------------------------------------
# registry + schema
# --------------------------------------------------------------------------
def test_registry_lists_five_scenarios():
    assert set(registry.SCENARIOS) == {
        "launch_overhead",
        "loop_overhead",
        "recovery_economics",
        "hpo_sweep",
        "adoption_cost",
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
