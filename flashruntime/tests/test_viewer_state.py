# tests/test_viewer_state.py
"""state.collect() enrichment for the flow map: per-attempt telemetry tails
and rank heartbeats, plus the top-level `monitor` sample. All reads must be
TOTAL — torn/invalid files degrade to partial data, never exceptions."""

from __future__ import annotations

import json

from flashruntime.viewer.state import collect


def _run_dir(tmp_path, attempts):
    doc = {
        "contract": "viewer_v1",
        "workload": {"command": ["python", "train.py"], "mode": "single", "source": "."},
        "state": "RUNNING",
        "started_at": 0.0,
        "finished_at": None,
        "max_restarts": 0,
        "attempts": attempts,
        "events": [],
        "trials": [],
    }
    (tmp_path / "run.json").write_text(json.dumps(doc))
    return tmp_path


def _attempt(tmp_path, name):
    d = tmp_path / "job" / name
    d.mkdir(parents=True)
    return {
        "attempt_id": name,
        "job_id": "local",
        "state": "RUNNING",
        "pid": "4000",
        "started_at": 0.0,
        "finished_at": None,
        "output_dir": str(d),
    }


def _sample(ts, cpu):
    return {
        "ts": ts,
        "machine": {"hostname": "box", "cpu_percent": cpu, "limited": False},
        "processes": [{"pid": 4000}],
    }


def test_attempts_enriched_with_telemetry_and_ranks(tmp_path):
    a = _attempt(tmp_path, "task-000")
    out = tmp_path / "job" / "task-000"
    with open(out / "telemetry.jsonl", "w") as f:
        f.write(json.dumps(_sample(1.0, 10.0)) + "\n")
        f.write(json.dumps(_sample(2.0, 20.0)) + "\n")
        f.write('{"torn')  # unterminated last line — writer mid-append
    ranks = out / "ranks"
    ranks.mkdir()
    (ranks / "rank-1.json").write_text(json.dumps({"rank": 1, "pid": 4002, "step": 5}))
    (ranks / "rank-0.json").write_text(json.dumps({"rank": 0, "pid": 4001, "step": 5}))
    (ranks / "rank-2.json").write_text("{torn")  # must be skipped, not fatal

    snap = collect(_run_dir(tmp_path, [a]))
    row = snap["attempts"][0]
    assert [t["ts"] for t in row["telemetry"]] == [1.0, 2.0]  # torn line skipped
    assert [r["rank"] for r in row["ranks"]] == [0, 1]  # sorted, torn skipped
    assert snap["monitor"]["ts"] == 2.0  # newest sample becomes the monitor


def test_monitor_picks_newest_across_attempts(tmp_path):
    a0 = _attempt(tmp_path, "task-000")
    a1 = _attempt(tmp_path, "task-001")
    with open(tmp_path / "job" / "task-000" / "telemetry.jsonl", "w") as f:
        f.write(json.dumps(_sample(5.0, 50.0)) + "\n")
    with open(tmp_path / "job" / "task-001" / "telemetry.jsonl", "w") as f:
        f.write(json.dumps(_sample(9.0, 90.0)) + "\n")
    snap = collect(_run_dir(tmp_path, [a0, a1]))
    assert snap["monitor"]["machine"]["cpu_percent"] == 90.0


def test_absent_telemetry_degrades_to_empty(tmp_path):
    a = _attempt(tmp_path, "task-000")
    snap = collect(_run_dir(tmp_path, [a]))
    row = snap["attempts"][0]
    assert row["telemetry"] == [] and row["ranks"] == []
    assert snap["monitor"] is None


def test_missing_output_dir_still_total(tmp_path):
    a = _attempt(tmp_path, "task-000")
    a["output_dir"] = str(tmp_path / "job" / "vanished")
    snap = collect(_run_dir(tmp_path, [a]))
    assert "error" not in snap
    assert snap["attempts"][0]["telemetry"] == []
    assert snap["attempts"][0]["ranks"] == []
