import re
from pathlib import Path

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
