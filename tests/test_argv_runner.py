from pathlib import Path
from unittest import mock

import pytest

from flashnode.executor.argv_runner import ArgvDockerRunner
from flashnode.executor.runner import TaskExecutionError

IMAGES = frozenset({"ghcr.io/zolli/trainer:1.0"})


def _runner(**kw):
    return ArgvDockerRunner(allowed_images=IMAGES, **kw)


def _payload(**over):
    base = {"argv": ["python", "train.py"], "image": "ghcr.io/zolli/trainer:1.0",
            "env": {"LR": "0.05"}, "task_id": "task-000"}
    base.update(over)
    return base


_MISSING = object()   # distinct from every legitimate-but-invalid argv value


@pytest.mark.parametrize("bad", [_MISSING, None, [], "python train.py", [1, 2]])
def test_bad_argv_refused_before_any_subprocess(tmp_path, bad):
    payload = _payload()
    if bad is _MISSING:
        payload.pop("argv")          # payload carrying no argv key at all
    else:
        payload["argv"] = bad        # present but malformed
    with mock.patch("subprocess.run") as run:
        with pytest.raises(TaskExecutionError):
            _runner().run(payload, tmp_path, {})
    run.assert_not_called()      # a check that runs after launching is not a check


def test_non_allowlisted_image_refused_before_any_subprocess(tmp_path):
    with mock.patch("subprocess.run") as run:
        with pytest.raises(TaskExecutionError, match="not allowlisted"):
            _runner().run(_payload(image="evil/image:1"), tmp_path, {})
    run.assert_not_called()


def test_image_cannot_smuggle_a_docker_flag(tmp_path):
    """A hostile image value must never reach docker's flag parser."""
    with mock.patch("subprocess.run") as run:
        with pytest.raises(TaskExecutionError):
            _runner().run(_payload(image="--privileged"), tmp_path, {})
    run.assert_not_called()


def test_bad_env_key_refused(tmp_path):
    with mock.patch("subprocess.run") as run:
        with pytest.raises(TaskExecutionError, match="env"):
            _runner().run(_payload(env={"BAD KEY": "v"}), tmp_path, {})
    run.assert_not_called()


def test_argv_lands_after_the_image_so_flags_are_inert(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "metrics.json").write_text("{}")
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr=b"")
        _runner().run(_payload(argv=["--privileged"]), tmp_path, {})
    cmd = run.call_args[0][0]
    assert cmd.index("--privileged") > cmd.index("ghcr.io/zolli/trainer:1.0")


def test_missing_metrics_json_fails_the_task(tmp_path):
    (tmp_path / "out").mkdir()
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr=b"")
        with pytest.raises(TaskExecutionError, match="metrics.json"):
            _runner().run(_payload(), tmp_path, {})


def test_nonzero_exit_reports_stderr_tail(tmp_path):
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=1, stderr=b"boom")
        with pytest.raises(TaskExecutionError, match="boom"):
            _runner().run(_payload(), tmp_path, {})


def test_output_size_cap_enforced(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "metrics.json").write_text("{}")
    (tmp_path / "out" / "big.bin").write_bytes(b"x" * 2048)
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr=b"")
        with pytest.raises(TaskExecutionError, match="output"):
            _runner(max_output_bytes=1024).run(_payload(), tmp_path, {})
