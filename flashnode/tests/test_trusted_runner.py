"""TrustedArgvRunner: pool argv payloads, no container, /work rewritten.

The compiled argv names CONTAINER paths (`python /work/inputs/code/train.py`
— the docker runners bind the workdir at /work). Here there is no container,
so every argv token starting with /work is rewritten onto the real workdir.
Rewriting tokens, not substrings: an argument that merely CONTAINS "/work"
is the submitter's business.
"""

import json
from pathlib import Path

import pytest

from flashnode.executor.runner import TaskExecutionError
from flashnode.executor.trusted_runner import TrustedArgvRunner


def _payload(argv):
    return {"argv": argv, "isolation": {"tier": "sandboxed", "allowFallback": True},
            "pool": "p-1"}


def test_runs_the_argv_with_work_prefix_rewritten(tmp_path):
    workdir = tmp_path
    code = workdir / "inputs" / "code"
    code.mkdir(parents=True)
    (code / "train.py").write_text(
        "import json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'metrics.json').write_text(json.dumps({'ok': True}))\n"
    )
    runner = TrustedArgvRunner()
    outdir = runner.run(
        _payload(["python", "/work/inputs/code/train.py", "/work/out"]),
        workdir, {"code": code},
    )
    assert json.loads((outdir / "metrics.json").read_text()) == {"ok": True}
    assert runner.last_exit_code == 0


def test_refuses_a_payload_without_argv(tmp_path):
    with pytest.raises(TaskExecutionError):
        TrustedArgvRunner().run({"module": "flashml_workloads.sklearn_trial"},
                                tmp_path, {})


def test_refuses_a_non_list_argv(tmp_path):
    with pytest.raises(TaskExecutionError):
        TrustedArgvRunner().run(_payload("python /work/x.py"), tmp_path, {})


def test_nonzero_exit_raises_and_records_the_code(tmp_path):
    (tmp_path / "inputs").mkdir()
    runner = TrustedArgvRunner()
    with pytest.raises(TaskExecutionError):
        runner.run(_payload(["python", "-c", "raise SystemExit(3)"]), tmp_path, {})
    assert runner.last_exit_code == 3


def test_environment_is_scrubbed(tmp_path, monkeypatch):
    """The task must not inherit the agent's secrets — same whitelist as
    SubprocessRunner (task_env). The out path travels as an argv token so
    the /work rewrite applies to it."""
    monkeypatch.setenv("FLASHNODE_MACHINE_TOKEN", "fmk_secret")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'env.txt').write_text(str('FLASHNODE_MACHINE_TOKEN' in os.environ))\n"
    )
    runner = TrustedArgvRunner()
    runner.run(_payload(["python", str(probe), "/work/out"]), tmp_path, {})
    assert (tmp_path / "out" / "env.txt").read_text() == "False"


def test_image_digest_is_always_empty():
    """No container ran; claiming an image digest would be fabricated
    evidence (the SubprocessRunner rule)."""
    assert TrustedArgvRunner().last_image_digest == ""
