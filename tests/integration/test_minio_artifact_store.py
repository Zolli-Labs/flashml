"""Real-MinIO roundtrip for the S3-compatible artifact store.

Needs FLASHML_TEST_MINIO_ENDPOINT (see this folder's README); skips
otherwise. This is the integration proof that the store the unit tests mock
behaves the same against real object storage — including the sha256 the
checkpoint catalog's parts-first commit depends on.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid

import pytest


@pytest.fixture()
def store(minio_config):
    pytest.importorskip("minio", reason="pip install 'flashruntime[artifacts]'")
    from flashruntime.artifacts.store import S3CompatibleArtifactStore

    s = S3CompatibleArtifactStore(
        endpoint=minio_config.endpoint,
        bucket=minio_config.bucket,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
    )
    s.ensure_bucket()
    return s


def test_put_get_roundtrip_with_verified_sha256(store, tmp_path):
    payload = uuid.uuid4().hex.encode() * 1000
    src = tmp_path / "part-rank0.bin"
    src.write_bytes(payload)
    key = f"integration-tests/{uuid.uuid4().hex[:8]}/part-rank0.bin"

    async def scenario():
        record = await store.put_file(src, key)
        assert record.sha256 == hashlib.sha256(payload).hexdigest()
        assert record.size_bytes == len(payload)
        assert await store.exists(key)

        dest = tmp_path / "restored.bin"
        await store.get_file(key, dest)
        assert dest.read_bytes() == payload

        listed = await store.list_prefix(key.rsplit("/", 1)[0])
        assert any(r.object_key == key for r in listed)

    asyncio.run(scenario())


def test_exists_is_false_for_missing_key(store):
    async def scenario():
        assert not await store.exists(f"integration-tests/absent-{uuid.uuid4().hex}")

    asyncio.run(scenario())
