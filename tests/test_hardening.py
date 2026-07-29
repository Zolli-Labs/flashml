from pathlib import Path

from flashnode.executor.hardening import CONTAINER_WORKDIR, harden_args


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
