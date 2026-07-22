# tests/test_sdk_submit.py
"""flash.submit(): compile → launch → wait → collect, on real subprocesses.
Uses stdlib-only child scripts so the test needs no ML frameworks."""

from __future__ import annotations

import json
import sys
import textwrap


def _write_script(tmp_path, body: str) -> str:
    src = tmp_path / "userproj"
    src.mkdir(exist_ok=True)
    (src / "train.py").write_text(textwrap.dedent(body))
    return str(src)


def test_single_command_collects_metrics(tmp_path):
    import flashruntime as flash

    source = _write_script(
        tmp_path,
        """
        import json
        json.dump({"accuracy": 0.9}, open("metrics.json", "w"))
        """,
    )
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": source}),
        output_dir=tmp_path / "out",
    )
    assert run.state.value == "SUCCEEDED"
    assert run.trials == [{"accuracy": 0.9}]
    assert any(p.name == "metrics.json" for p in run.artifacts)


def test_failure_surfaces_as_failed_with_logs(tmp_path):
    import flashruntime as flash

    source = _write_script(tmp_path, "print('boom'); raise SystemExit(3)")
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": source}),
        output_dir=tmp_path / "out",
    )
    assert run.state.value == "FAILED"
    assert "boom" in run.logs()


def test_fanout_runs_each_param_set_and_picks_best(tmp_path):
    import flashruntime as flash

    source = _write_script(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser(); ap.add_argument("--x", type=float)
        args = ap.parse_args()
        json.dump({"x": args.x, "score": args.x * 2}, open("metrics.json", "w"))
        """,
    )
    run = flash.submit(
        flash.CommandWorkload(
            command=f"{sys.executable} train.py --x {{x}}",
            source={"path": source},
            task_params=[{"x": 1}, {"x": 3}, {"x": 2}],
            outputs=flash.OutputSpec(primary_metric="score"),
        ),
        output_dir=tmp_path / "out",
    )
    assert run.state.value == "SUCCEEDED"
    assert len(run.trials) == 3
    assert run.best_trial()["x"] == 3  # highest score wins
    assert run.best_trial(metric="score", maximize=False)["x"] == 1


def test_stale_outputs_from_previous_trial_not_recollected(tmp_path):
    import flashruntime as flash

    # writes metrics.json only when --x != 2: trial x=2 must NOT inherit x=1's file
    source = _write_script(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser(); ap.add_argument("--x", type=int)
        args = ap.parse_args()
        if args.x != 2:
            json.dump({"x": args.x}, open("metrics.json", "w"))
        """,
    )
    run = flash.submit(
        flash.CommandWorkload(
            command=f"{sys.executable} train.py --x {{x}}",
            source={"path": source},
            task_params=[{"x": 1}, {"x": 2}],
        ),
        output_dir=tmp_path / "out",
    )
    assert [t["x"] for t in run.trials] == [1]
