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


def test_submit_watch_opens_viewer_and_prints_url(tmp_path, capsys, monkeypatch):
    # --watch opens the live viewer: it prints a loopback URL. webbrowser.open
    # is stubbed so the test never actually pops a browser.
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda u, *a, **k: opened.append(u) or True)

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
    assert "http://127.0.0.1:" in stdout  # the viewer URL is printed
    assert opened and opened[0].startswith("http://127.0.0.1:")  # browser opened at it


def test_unknown_command_is_argparse_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2  # argparse usage error


# --- submit-spec: the restored pre-0.1.0 JobSpec POST (was `submit`) ---------


def test_submit_spec_posts_jobspec_to_api(tmp_path, capsys, monkeypatch):
    """`submit-spec FILE` reads a JobSpec YAML and POSTs it to the coordinator.

    The handler's real plumbing is `httpx.post`, so we monkeypatch that (no
    running coordinator) and assert the URL and the parsed payload.
    """
    import httpx

    spec_file = tmp_path / "job.yaml"
    spec_file.write_text("apiVersion: flashml.dev/v1alpha1\nkind: Job\nname: my-sweep\n")

    captured: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"id": "job-123", "state": "PENDING"}

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = main(["--api", "http://api.example:8100", "submit-spec", str(spec_file)])
    assert rc == 0
    assert captured["url"] == "http://api.example:8100/v1alpha1/jobs"
    assert captured["json"] == {
        "apiVersion": "flashml.dev/v1alpha1",
        "kind": "Job",
        "name": "my-sweep",
    }
    assert "job-123" in capsys.readouterr().out


def test_submit_spec_missing_file_exits_nonzero_with_clear_message(tmp_path):
    """A missing spec file fails before any HTTP call (verbatim pre-0.1.0
    behavior: `open()` raises), surfacing a clear, filename-bearing error —
    which at the CLI boundary is a non-zero exit."""
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError) as exc:
        main(["--api", "http://api.example:8100", "submit-spec", str(missing)])
    assert str(missing) in str(exc.value)
