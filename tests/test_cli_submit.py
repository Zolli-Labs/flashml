# tests/test_cli_submit.py
"""`flashruntime submit CMD` — the shell front door for command workloads.

Drives `main([...])` directly (no subprocess) with stdlib-only child
scripts, so the test needs neither a running coordinator nor ML frameworks.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from flashruntime.service.cli import main


def _write_script(tmp_path, body: str) -> str:
    src = tmp_path / "userproj"
    src.mkdir(exist_ok=True)
    (src / "train.py").write_text(textwrap.dedent(body))
    return str(src)


def test_submit_success_exits_zero_and_writes_run_json(tmp_path, capsys):
    src = _write_script(
        tmp_path,
        """
        import json
        json.dump({"accuracy": 0.9}, open("metrics.json", "w"))
        """,
    )
    out = tmp_path / "out"
    rc = main(
        ["submit", f"{sys.executable} train.py", "--source", src, "--output-dir", str(out), "--no-watch"]
    )
    assert rc == 0
    assert (out / "run.json").is_file()
    stdout = capsys.readouterr().out
    assert "SUCCEEDED" in stdout


def test_submit_failing_script_exits_one(tmp_path, capsys):
    src = _write_script(tmp_path, "raise SystemExit(3)")
    out = tmp_path / "out"
    rc = main(
        ["submit", f"{sys.executable} train.py", "--source", src, "--output-dir", str(out), "--no-watch"]
    )
    assert rc == 1
    assert "FAILED" in capsys.readouterr().out


def test_submit_reports_trial_count_for_fanout(tmp_path, capsys):
    src = _write_script(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser(); ap.add_argument("--x", type=int)
        args = ap.parse_args()
        json.dump({"x": args.x}, open("metrics.json", "w"))
        """,
    )
    out = tmp_path / "out"
    rc = main(
        [
            "submit",
            f"{sys.executable} train.py --x {{x}}",
            "--source",
            src,
            "--output-dir",
            str(out),
            "--task-params",
            '[{"x": 1}, {"x": 2}, {"x": 3}]',
            "--no-watch",
        ]
    )
    assert rc == 0
    assert "3" in capsys.readouterr().out  # trials: 3


def test_submit_bad_task_params_is_clean_error_exit_two(tmp_path, capsys):
    src = _write_script(tmp_path, "pass")
    rc = main(
        [
            "submit",
            f"{sys.executable} train.py",
            "--source",
            src,
            "--task-params",
            "{not valid json}",
            "--no-watch",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "task-params" in err.lower()
    # a clean one-line message, not a Python traceback
    assert "Traceback" not in err


def test_submit_watch_prints_honest_placeholder(tmp_path, capsys):
    src = _write_script(
        tmp_path,
        """
        import json
        json.dump({"ok": 1}, open("metrics.json", "w"))
        """,
    )
    out = tmp_path / "out"
    rc = main(
        ["submit", f"{sys.executable} train.py", "--source", src, "--output-dir", str(out), "--watch"]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "viewer" in stdout.lower()  # honest placeholder until T7


def test_unknown_command_is_argparse_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2  # argparse usage error
