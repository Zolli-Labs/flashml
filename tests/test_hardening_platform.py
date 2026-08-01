"""Windows-hosts plan (flashml-cloud docs/superpowers/plans/
2026-08-01-windows-hosts.md), Tasks 2 and 3.

Task 2: `os.getuid`/`os.getgid` don't exist on Windows, so
flashnode/flashnode/executor/hardening.py drops the `--user` flag there —
but ONLY there, and ONLY because the curated images
(flashml-cloud/images/*/Dockerfile) now declare a fixed non-root USER
(Task 1, which landed first — see that plan's "trap at the centre").

Every Task 2 test asserts the FULL expected argv, not merely presence or
absence of `--user`. A narrower test (e.g. "assert '--user' not in argv")
would also pass a broken fix that accidentally dropped `--cap-drop=ALL` or
some other hardening flag along with it — the whole point of hardening.py
existing as a single seam (see its module docstring) is that no runner can
silently lose a flag.

Task 3: the `-v {workdir}:/work` bind-mount source must render safely for
Windows paths (`C:\\Users\\...`), whose drive-letter colon and backslashes
would otherwise collide with the `src:dst` split. Tested with a synthetic
`PureWindowsPath` — deliberately NOT gated on `sys.platform == "win32"`, so
it always runs in CI regardless of the host running these tests.

Everything here runs on macOS with `sys.platform`/`os.getuid` faked. It
verifies the argv flashnode CONSTRUCTS, not that Docker Desktop on a real
Windows machine accepts it — see Task 4's documentation for that
distinction.
"""
from __future__ import annotations

import os
import sys
from pathlib import PureWindowsPath

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


# ---------------------------------------------------------------------------
# Task 3: Windows bind-mount path handling
# ---------------------------------------------------------------------------


def test_posix_bind_mount_source_is_byte_identical(tmp_path):
    args = harden_args(tmp_path, cpus=CPUS, memory_gb=MEMORY_GB)
    v_idx = args.index("-v")
    assert args[v_idx + 1] == f"{tmp_path}:{CONTAINER_WORKDIR}"


def test_windows_bind_mount_source_converts_drive_letter_path():
    """A synthetic PureWindowsPath, deliberately not gated on
    sys.platform == 'win32' — this must run unconditionally in CI or the
    bug it guards against (backslashes + drive-letter colon breaking the
    `-v src:dst` split) comes straight back. A real WindowsPath (used when
    flashnode actually runs on Windows) IS a PureWindowsPath, so the same
    code path handles both.
    """
    windows_workdir = PureWindowsPath(
        r"C:\Users\phong\AppData\Local\flashnode\work\abc123"
    )

    args = harden_args(windows_workdir, cpus=CPUS, memory_gb=MEMORY_GB)

    v_idx = args.index("-v")
    bind_arg = args[v_idx + 1]

    # Exactly one ':' — the src:dst separator. A raw Windows path here
    # would contain two (drive colon + separator), which is the actual
    # bug: `docker run -v C:\Users\...:/work` mis-splits.
    assert bind_arg.count(":") == 1
    src, dst = bind_arg.split(":", 1)
    assert dst == CONTAINER_WORKDIR
    assert "\\" not in src
    assert src == "/c/Users/phong/AppData/Local/flashnode/work/abc123"


def test_windows_bind_mount_source_lowercases_drive_letter():
    windows_workdir = PureWindowsPath(r"D:\flashml\work")
    args = harden_args(windows_workdir, cpus=CPUS, memory_gb=MEMORY_GB)
    bind_arg = args[args.index("-v") + 1]
    assert bind_arg.startswith("/d/")


def test_windows_bind_mount_source_without_drive_letter_raises():
    # A driveless Windows path (e.g. a UNC share) has no drive letter to
    # rewrite into Docker Desktop's /c/... form — refuse rather than
    # produce a bind mount that silently points nowhere (the same failure
    # class the plan flags for a workdir outside Docker Desktop's shared
    # directories: an empty mount, not an error).
    windows_workdir = PureWindowsPath(r"\Users\phong\work")
    with pytest.raises(ValueError):
        harden_args(windows_workdir, cpus=CPUS, memory_gb=MEMORY_GB)


def test_windows_argv_full_expected_argv_with_converted_bind_source(monkeypatch):
    """Combine Task 2 and Task 3: on a faked Windows platform with a real
    Windows-shaped workdir, the full argv must match exactly — no --user,
    a converted -v source, and every other flag untouched."""
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delattr(os, "getgid", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    windows_workdir = PureWindowsPath(r"C:\Users\phong\flashnode-work")
    args = harden_args(windows_workdir, cpus=CPUS, memory_gb=MEMORY_GB)

    expected = _expected_argv(
        "/c/Users/phong/flashnode-work", user_flag=[]
    )
    assert args == expected
