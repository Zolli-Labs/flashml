import re
import subprocess
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


def test_timeout_kills_the_container_by_name(tmp_path):
    """The docker CLIENT process dying on timeout does not stop the daemon-
    side container — we must issue an explicit `docker kill <name>` for the
    SAME container that `docker run --name <name>` launched. A test that
    only checks "some kill happened" would pass against a bug that kills an
    unrelated (or no) container, so we assert the name matches exactly.
    """
    with mock.patch("subprocess.run") as run:
        run.side_effect = [
            subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=1),
            mock.Mock(returncode=0, stderr=b""),  # the docker kill call
        ]
        with pytest.raises(TaskExecutionError, match="wall clock"):
            _runner(timeout_seconds=1).run(_payload(), tmp_path, {})

    assert run.call_count == 2
    run_cmd = run.call_args_list[0][0][0]
    kill_cmd = run.call_args_list[1][0][0]
    name = run_cmd[run_cmd.index("--name") + 1]
    assert kill_cmd[:2] == ["docker", "kill"]
    assert kill_cmd[2] == name


def test_timeout_kill_failure_does_not_mask_the_timeout_error(tmp_path):
    """docker kill is best-effort — the container may already be gone by
    the time we ask. That must not swallow the original timeout error."""
    with mock.patch("subprocess.run") as run:
        run.side_effect = [
            subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=1),
            RuntimeError("no such container"),
        ]
        with pytest.raises(TaskExecutionError, match="wall clock"):
            _runner(timeout_seconds=1).run(_payload(), tmp_path, {})


@pytest.mark.parametrize("task_id", ["../evil", "a b", ""])
def test_container_name_is_docker_legal_even_with_hostile_task_id(tmp_path, task_id):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "metrics.json").write_text("{}")
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr=b"")
        _runner().run(_payload(task_id=task_id), tmp_path, {})
    cmd = run.call_args[0][0]
    name = cmd[cmd.index("--name") + 1]
    assert re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", name)


def test_container_name_unique_across_concurrent_tasks(tmp_path):
    """Same task_id, two attempts — names must not collide."""
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "metrics.json").write_text("{}")
    names = []
    for _ in range(2):
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stderr=b"")
            _runner().run(_payload(), tmp_path, {})
        cmd = run.call_args[0][0]
        names.append(cmd[cmd.index("--name") + 1])
    assert names[0] != names[1]
