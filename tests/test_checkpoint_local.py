"""Parts-first / manifest-last on a plain filesystem: a crash mid-write
leaves no manifest; a corrupted part disqualifies its manifest on read."""

from __future__ import annotations

import pytest


def _make_step(root, step: int, content: bytes = b"weights"):
    from flashruntime.checkpoint.local import write_manifest

    d = root / f"step-{step:06d}"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(content)
    return write_manifest(d, job_id="j1", attempt_id="a1", step=step)


def test_manifest_written_last_with_verified_hashes(tmp_path):
    from flashruntime.checkpoint.local import MANIFEST_NAME

    manifest = _make_step(tmp_path, 10)
    assert (tmp_path / "step-000010" / MANIFEST_NAME).is_file()
    assert manifest.parts[0].key == "model.pt"
    assert manifest.validation.value == "hash_verified"


def test_zero_parts_refused(tmp_path):
    from flashruntime.checkpoint.local import write_manifest

    d = tmp_path / "step-000001"
    d.mkdir()
    with pytest.raises(ValueError, match="no part files"):
        write_manifest(d, job_id="j1", attempt_id="a1", step=1)


def test_latest_valid_picks_newest_step(tmp_path):
    from flashruntime.checkpoint.local import latest_valid_manifest

    _make_step(tmp_path, 10)
    _make_step(tmp_path, 20)
    assert latest_valid_manifest(tmp_path).step == 20


def test_corrupted_part_disqualifies_its_manifest(tmp_path):
    from flashruntime.checkpoint.local import latest_valid_manifest

    _make_step(tmp_path, 10)
    _make_step(tmp_path, 20)
    (tmp_path / "step-000020" / "model.pt").write_bytes(b"CORRUPTED")
    assert latest_valid_manifest(tmp_path).step == 10  # falls back, never loads bad state


def test_no_checkpoints_is_none(tmp_path):
    from flashruntime.checkpoint.local import latest_valid_manifest

    assert latest_valid_manifest(tmp_path / "missing") is None


def test_manifest_that_is_a_directory_does_not_crash_scan(tmp_path):
    """A step dir whose manifest.json is itself a DIRECTORY makes read_text
    raise IsADirectoryError (an OSError subclass). The scan must treat that
    manifest as nonexistent and fall back to the last good one, never crash."""
    from flashruntime.checkpoint.local import MANIFEST_NAME, latest_valid_manifest

    _make_step(tmp_path, 10)
    bad = tmp_path / "step-000020"
    bad.mkdir()
    (bad / "model.pt").write_bytes(b"weights")
    (bad / MANIFEST_NAME).mkdir()  # manifest.json is a directory ⇒ read_text raises OSError
    assert latest_valid_manifest(tmp_path).step == 10
