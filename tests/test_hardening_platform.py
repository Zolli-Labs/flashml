"""Windows-hosts plan (flashml-cloud docs/superpowers/plans/
2026-08-01-windows-hosts.md), Task 2: platform-conditional `--user`.

`os.getuid`/`os.getgid` don't exist on Windows, so
flashnode/flashnode/executor/hardening.py drops the `--user` flag there —
but ONLY there, and ONLY because the curated images
(flashml-cloud/images/*/Dockerfile) now declare a fixed non-root USER
(Task 1, which landed first — see that plan's "trap at the centre").

Every test here asserts the FULL expected argv, not merely presence or
absence of `--user`. A narrower test (e.g. "assert '--user' not in argv")
would also pass a broken fix that accidentally dropped `--cap-drop=ALL` or
some other hardening flag along with it — the whole point of hardening.py
existing as a single seam (see its module docstring) is that no runner can
silently lose a flag.

This runs on macOS with `sys.platform`/`os.getuid` faked. It verifies the
argv flashnode CONSTRUCTS, not that Docker Desktop on a real Windows
machine accepts it — see Task 4's documentation for that distinction.
"""
from __future__ import annotations

import os
import sys

import pytest

from flashnode.executor.hardening import CONTAINER_WORKDIR, harden_args

CPUS = 2.0
MEMORY_GB = 4.0


def _expected_common_flags() -> list[str]:
    return [
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
    ]


def _tail_flags(pids_limit: int = 512) -> list[str]:
    return [
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--pids-limit={pids_limit}",
        "--cpus", str(CPUS),
        "--memory", f"{MEMORY_GB}g",
        "--memory-swap", f"{MEMORY_GB}g",
        "--ulimit", "nofile=1024:1024",
    ]


def _expected_argv(workdir, user_flag: list[str]) -> list[str]:
    return [
        *_expected_common_flags(),
        *user_flag,
        *_tail_flags(),
        "-v", f"{workdir}:{CONTAINER_WORKDIR}",
        "-w", CONTAINER_WORKDIR,
    ]


def test_posix_argv_unchanged_full_expected_argv(tmp_path):
    """POSIX behaviour must not change at all (Global Constraints). Assert
    the WHOLE argv, in order, not just that --user is present."""
    args = harden_args(tmp_path, cpus=CPUS, memory_gb=MEMORY_GB)
    expected = _expected_argv(
        tmp_path, user_flag=["--user", f"{os.getuid()}:{os.getgid()}"]
    )
    assert args == expected


def test_windows_argv_omits_user_but_keeps_every_other_hardening_flag(
    tmp_path, monkeypatch
):
    """The fix that matters: dropping --user on Windows must not drop
    anything else. A test that only checked '--user' not in args would
    also pass a fix that accidentally dropped --cap-drop=ALL alongside it.
    """
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delattr(os, "getgid", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    args = harden_args(tmp_path, cpus=CPUS, memory_gb=MEMORY_GB)

    expected = _expected_argv(tmp_path, user_flag=[])
    assert args == expected
    assert "--user" not in args
    # Every other hardening flag is still there, unweakened.
    for flag in ("--network", "--read-only", "--cap-drop=ALL",
                 "--security-opt=no-new-privileges", "--pids-limit=512"):
        assert flag in args


def test_unrecognised_platform_raises_rather_than_omitting(tmp_path, monkeypatch):
    """No os.getuid AND not win32: fail closed. An unrecognised platform
    must not quietly produce a root container (flashruntime/CLAUDE.md rule
    3: security fields fail closed)."""
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delattr(os, "getgid", raising=False)
    monkeypatch.setattr(sys, "platform", "some-future-os")

    with pytest.raises(RuntimeError):
        harden_args(tmp_path, cpus=CPUS, memory_gb=MEMORY_GB)


def test_posix_user_flag_position_matches_original_contract(tmp_path):
    # The original code placed --user right after the tmpfs flag and right
    # before --cap-drop=ALL. Pin the position so a reorder is visible.
    args = harden_args(tmp_path, cpus=CPUS, memory_gb=MEMORY_GB)
    tmpfs_idx = args.index("--tmpfs")
    cap_idx = args.index("--cap-drop=ALL")
    user_idx = args.index("--user")
    assert tmpfs_idx < user_idx < cap_idx
