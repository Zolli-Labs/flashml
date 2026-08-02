import re
from pathlib import Path
from unittest import mock

import pytest

from flashnode.executor.hardening import CONTAINER_WORKDIR, container_name, harden_args


def test_harden_args_carries_the_full_security_contract(tmp_path):
    args = harden_args(tmp_path, cpus=2.0, memory_gb=4.0)
    joined = " ".join(args)
    assert "--network none" in joined
    assert "--read-only" in joined
    assert "--cap-drop=ALL" in joined
    assert "--security-opt=no-new-privileges" in joined
    assert "--pids-limit=512" in joined
    assert "noexec" in joined and "nosuid" in joined       # tmpfs flags
    assert f"{tmp_path}:{CONTAINER_WORKDIR}" in joined


def test_memory_swap_equals_memory():
    """Without this, --memory is bypassable via swap — the cap is a
    suggestion rather than a limit."""
    args = harden_args(Path("/tmp/x"), cpus=1.0, memory_gb=4.0)
    assert args[args.index("--memory") + 1] == "4.0g"
    assert args[args.index("--memory-swap") + 1] == "4.0g"


def test_runs_as_the_invoking_user_not_root():
    import os
    args = harden_args(Path("/tmp/x"), cpus=1.0, memory_gb=1.0)
    assert args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


# -- F5/F12: container_name lives here so docker_runner and argv_runner
# cannot drift on the naming (and therefore kill-by-name) contract ------------


def test_container_name_is_docker_legal_even_with_hostile_task_id():
    for task_id in ["../evil", "a b", "", None]:
        name = container_name(task_id)
        assert re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", name)


def test_container_name_unique_across_calls():
    assert container_name("same-task-id") != container_name("same-task-id")


# -- GPUs (spec §5) -----------------------------------------------------------
#
# `--gpus` is the only flag here that WIDENS what a container can touch, so
# it is the only one that must be absent unless the payload asked for it.


def _golden_flags(workdir: Path) -> list[str]:
    """The flag list exactly as it stood before `--gpus` existed.

    Written out rather than computed, so a change to harden_args has to
    change this literal too — every job running today depends on it.
    """
    import os
    return [
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        *(["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []),
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=512",
        "--cpus", "2.0",
        "--memory", "4.0g",
        "--memory-swap", "4.0g",
        "--ulimit", "nofile=1024:1024",
        "-v", f"{workdir}:{CONTAINER_WORKDIR}",
        "-w", CONTAINER_WORKDIR,
    ]


def test_the_no_gpu_flag_list_is_byte_for_byte_what_it_always_was(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASHNODE_LOCAL_DATA", raising=False)
    assert harden_args(tmp_path, cpus=2.0, memory_gb=4.0) == _golden_flags(tmp_path)


def test_a_gpu_request_adds_exactly_one_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASHNODE_LOCAL_DATA", raising=False)
    args = harden_args(tmp_path, cpus=2.0, memory_gb=4.0, gpus=1)
    assert args[args.index("--gpus") + 1] == "1"
    # Nothing else moved: strip the pair and the golden list is back.
    at = args.index("--gpus")
    assert args[:at] + args[at + 2:] == _golden_flags(tmp_path)


def test_two_gpus(tmp_path):
    args = harden_args(tmp_path, cpus=2.0, memory_gb=4.0, gpus=2)
    assert args[args.index("--gpus") + 1] == "2"


@pytest.mark.parametrize(
    "gpus", [None, 0, -1, -1.0, 1.5, "1", "all", True, False, [1], {"count": 1}, object()]
)
def test_nothing_but_a_positive_int_produces_a_gpus_flag(tmp_path, gpus):
    """Defence in depth, not validation: the coordinator's placement gate
    already refused a task with a requirement of this shape, so reaching
    here means gate and payload disagree. `True` is called out explicitly —
    bool is an int subclass, and `--gpus True` is not a thing.

    `"all"` is refused too. It is valid docker syntax and that is precisely
    the danger: it hands the container every device on a volunteer's machine
    off a string in an untrusted payload.
    """
    assert "--gpus" not in harden_args(tmp_path, cpus=1.0, memory_gb=1.0, gpus=gpus)


def test_doctor_still_calls_harden_args_the_old_way(tmp_path):
    """`doctor.check_hardened_run` passes neither local_inputs nor gpus. It
    must keep working untouched — a GPU-less volunteer runs it far more
    often than anyone runs a GPU job."""
    from flashnode.doctor import check_hardened_run

    seen = []

    def run(argv, **kw):
        seen.append(list(argv))
        return mock.Mock(returncode=0, stdout=b"flashnode-doctor", stderr=b"")

    result = check_hardened_run(run, tmp_path)
    assert result.status == "ok", result.detail
    assert "--gpus" not in seen[0]


# -- and the runners have to actually pass it ---------------------------------
#
# `--gpus` in harden_args with neither runner forwarding the payload key is a
# feature that is "implemented" and dead — which is exactly how the
# equivalent change died last time. It is invisible from either end: a GPU
# job denied its device runs on the CPU, slowly, and still SUCCEEDS.


def _fake_ok_run(outdir: Path):
    def run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "metrics.json").write_text("{}")
        return mock.Mock(returncode=0, stdout=b"", stderr=b"")
    return run


IMAGE = "ghcr.io/zolli/trainer:1.0"


def _argv_runner():
    from flashnode.executor.argv_runner import ArgvDockerRunner

    return ArgvDockerRunner(allowed_images=frozenset({IMAGE}))


def _module_runner():
    from flashnode.executor.docker_runner import DockerRunner

    return DockerRunner(allowed_images=frozenset({IMAGE}))


def _run_argv(payload, tmp_path):
    with mock.patch("subprocess.run", side_effect=_fake_ok_run(tmp_path / "out")) as sp:
        _argv_runner().run(
            {"argv": ["python", "train.py"], "image": IMAGE, "task_id": "t1", **payload},
            tmp_path, {},
        )
    return sp.call_args_list[0].args[0]


def _run_module(payload, tmp_path):
    with mock.patch("subprocess.run", side_effect=_fake_ok_run(tmp_path / "out")) as sp:
        _module_runner().run(
            {"module": "flashml_workloads.sgd_trainer", "image": IMAGE,
             "task_id": "t1", **payload},
            tmp_path, {},
        )
    return sp.call_args_list[0].args[0]


def test_argv_runner_forwards_the_gpu_request(tmp_path):
    cmd = _run_argv({"gpus": 1}, tmp_path)
    assert cmd[cmd.index("--gpus") + 1] == "1"


def test_docker_runner_forwards_the_gpu_request(tmp_path):
    cmd = _run_module({"gpus": 2}, tmp_path)
    assert cmd[cmd.index("--gpus") + 1] == "2"


def test_neither_runner_asks_for_a_gpu_when_the_payload_does_not(tmp_path):
    assert "--gpus" not in _run_argv({}, tmp_path)
    assert "--gpus" not in _run_module({}, tmp_path)


def test_gpus_all_reaches_docker_from_neither_runner(tmp_path):
    """`{"gpus": "all"}` is valid docker syntax for "every device on this
    machine", straight out of an untrusted payload. It must die in
    hardening, from both directions."""
    assert "all" not in _run_argv({"gpus": "all"}, tmp_path)
    assert "all" not in _run_module({"gpus": "all"}, tmp_path)
