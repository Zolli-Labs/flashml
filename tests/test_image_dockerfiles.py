"""Task 1 of the Windows-hosts plan: the curated images must each declare a
non-root ``USER`` as their last user-affecting instruction.

This is the safety prerequisite for Task 2 (flashnode/flashnode/executor/
hardening.py making ``--user`` platform-conditional). Windows omits
``--user`` because ``os.getuid``/``os.getgid`` don't exist there; that is
only safe if the image itself already runs as non-root. These tests parse
the Dockerfiles directly (no Docker daemon needed) so they run everywhere,
including CI without a build step.

Moved here from flashml-cloud on 2026-08-01 with the Dockerfiles they
describe. They no longer import the cloud API's CURATED map — the alias
set is derived from the image directories, which are the source of truth
in this repo. That the API and the published images agree is checked
separately, and from the position that matters, by flashml-cloud's
test_every_curated_image_is_anonymously_pullable.

See ``images/README.md``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "images"

EXPECTED_ALIASES = {"python-slim", "sklearn", "pytorch-cpu", "pytorch-cuda"}

# Names Docker treats as root regardless of spelling.
_ROOT_NAMES = {"root", "0"}


def _dockerfile_path(alias: str) -> Path:
    return IMAGES_DIR / alias / "Dockerfile"


def _user_instructions(dockerfile: Path) -> list[str]:
    """Every ``USER`` instruction's argument, in file order.

    Continuation lines (trailing ``\\``) aren't a concern for ``USER`` in
    practice, so a plain line scan is sufficient and keeps this independent
    of any Dockerfile-parsing library.
    """
    users = []
    for raw_line in dockerfile.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].upper() == "USER":
            users.append(parts[1].strip())
    return users


def _uid_of(user_arg: str) -> str:
    # "10001:10001" -> "10001"; a bare "10001" -> "10001".
    return user_arg.split(":", 1)[0]


@pytest.fixture(params=sorted(EXPECTED_ALIASES))
def alias(request):
    return request.param


def test_every_image_directory_is_covered_by_these_tests():
    # If images.py's alias set ever changes, this test file needs to grow
    # (or shrink) with it rather than silently checking a stale subset.
    on_disk = {p.name for p in IMAGES_DIR.iterdir() if p.is_dir()}
    assert EXPECTED_ALIASES == on_disk, (
        'a new image directory must be added to EXPECTED_ALIASES so these '
        'safety checks cover it — an unchecked image is one that may run as root'
    )


def test_dockerfile_exists(alias):
    assert _dockerfile_path(alias).is_file(), (
        f"images/{alias}/Dockerfile is missing"
    )


def test_dockerfile_declares_a_user_instruction(alias):
    users = _user_instructions(_dockerfile_path(alias))
    assert users, f"images/{alias}/Dockerfile has no USER instruction"


def test_final_user_is_non_root(alias):
    users = _user_instructions(_dockerfile_path(alias))
    assert users, f"images/{alias}/Dockerfile has no USER instruction"
    last_uid = _uid_of(users[-1])
    assert last_uid not in _ROOT_NAMES, (
        f"images/{alias}/Dockerfile's last USER ({users[-1]!r}) is root — "
        "Windows hosts omit --user and rely entirely on this"
    )


def test_final_user_is_not_nobody(alias):
    # `nobody`'s uid varies by base image/distro, which makes /work
    # ownership unpredictable across the three curated images — the plan
    # explicitly calls for a dedicated fixed uid instead.
    users = _user_instructions(_dockerfile_path(alias))
    last = users[-1]
    assert "nobody" not in last.lower(), (
        f"images/{alias}/Dockerfile uses 'nobody' — use a dedicated fixed "
        "uid instead so /work ownership is predictable"
    )


def test_user_is_the_last_user_affecting_instruction(alias):
    # A later `USER root` (even one immediately followed by another
    # non-root USER) would be a red flag: something ran as root after the
    # image believed it had already dropped privileges. Only the single
    # final USER line matters for the running container, but this test
    # specifically guards against files that fluctuate.
    users = _user_instructions(_dockerfile_path(alias))
    # Every USER call before the last one must ALSO already be non-root in
    # this codebase's Dockerfiles (root-owned setup is expected to happen
    # via RUN as the default root user before the first USER switch, not
    # via an explicit `USER root`).
    for earlier in users[:-1]:
        assert _uid_of(earlier) not in _ROOT_NAMES, (
            f"images/{alias}/Dockerfile explicitly switches to root "
            f"({earlier!r}) after an earlier USER — this can undo an "
            "earlier privilege drop even though it isn't the last line"
        )


def test_uid_matches_across_all_three_images():
    uids = {}
    for alias_name in sorted(EXPECTED_ALIASES):
        users = _user_instructions(_dockerfile_path(alias_name))
        assert users, f"images/{alias_name}/Dockerfile has no USER instruction"
        uids[alias_name] = _uid_of(users[-1])
    distinct = set(uids.values())
    assert len(distinct) == 1, (
        f"curated images disagree on uid, so /work ownership depends on "
        f"which image a task uses: {uids}"
    )


def test_final_uid_is_numeric_and_fixed():
    # Enforce "a dedicated uid (e.g. 10001)" — numeric, not a bare
    # username that resolves differently per base image.
    for alias_name in sorted(EXPECTED_ALIASES):
        users = _user_instructions(_dockerfile_path(alias_name))
        uid = _uid_of(users[-1])
        assert uid.isdigit(), (
            f"images/{alias_name}/Dockerfile's final USER uid {uid!r} is "
            "not numeric — a named user can resolve to a different uid "
            "per base image, breaking cross-image /work ownership"
        )


def test_dockerfile_base_image_reference_is_documented():
    # Not a hard requirement of the plan, but every curated Dockerfile
    # should at least declare a FROM so it's buildable in Plan 7.
    for alias_name in sorted(EXPECTED_ALIASES):
        text = _dockerfile_path(alias_name).read_text()
        assert any(
            line.strip().upper().startswith("FROM")
            for line in text.splitlines()
        ), f"images/{alias_name}/Dockerfile has no FROM instruction"
