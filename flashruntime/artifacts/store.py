from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from flashruntime.protocol.v1alpha1 import ArtifactRecord

ARTIFACT_SCHEME = "artifact://"


def artifact_uri_to_key(uri: str) -> str:
    if not uri.startswith(ARTIFACT_SCHEME):
        raise ValueError(f"not an artifact URI: {uri}")
    return uri[len(ARTIFACT_SCHEME):]


def key_to_artifact_uri(object_key: str) -> str:
    return f"{ARTIFACT_SCHEME}{object_key}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@runtime_checkable
class ArtifactStore(Protocol):
    backend: str
    bucket: str

    async def put_file(self, local_path: Path, object_key: str) -> ArtifactRecord: ...

    async def get_file(self, object_key: str, destination: Path) -> None: ...

    async def exists(self, object_key: str) -> bool: ...

    async def list_prefix(self, prefix: str) -> list[ArtifactRecord]: ...


class S3CompatibleArtifactStore:
    """MinIO (and any S3-compatible endpoint) via the `minio` client.

    Also usable against OSS's S3-compatible mode, but prefer OSSArtifactStore
    for OSS — OSS is not a drop-in S3 (addressing style, ETag semantics, STS
    tokens differ), so the native client is the honest integration.
    """

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        region: str | None = None,
        backend: str = "minio",
    ):
        from minio import Minio

        self.backend = backend
        self.bucket = bucket
        self._client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key,
            secure=secure, region=region,
        )

    def ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)

    def _record(self, object_key: str, etag: str | None, size: int | None,
                sha256: str | None = None) -> ArtifactRecord:
        return ArtifactRecord(
            uri=key_to_artifact_uri(object_key),
            backend=self.backend,  # type: ignore[arg-type]
            bucket=self.bucket,
            object_key=object_key,
            etag=etag,
            sha256=sha256,
            size_bytes=size,
        )

    async def put_file(self, local_path: Path, object_key: str) -> ArtifactRecord:
        local_path = Path(local_path)

        def _put():
            digest = _sha256(local_path)
            result = self._client.fput_object(self.bucket, object_key, str(local_path))
            return self._record(
                object_key, result.etag, local_path.stat().st_size, digest
            )

        return await asyncio.to_thread(_put)

    async def get_file(self, object_key: str, destination: Path) -> None:
        await asyncio.to_thread(
            self._client.fget_object, self.bucket, object_key, str(destination)
        )

    async def exists(self, object_key: str) -> bool:
        def _stat() -> bool:
            from minio.error import S3Error

            try:
                self._client.stat_object(self.bucket, object_key)
                return True
            except S3Error as exc:
                if exc.code in ("NoSuchKey", "NoSuchObject"):
                    return False
                raise

        return await asyncio.to_thread(_stat)

    async def list_prefix(self, prefix: str) -> list[ArtifactRecord]:
        def _list():
            objects = self._client.list_objects(self.bucket, prefix=prefix, recursive=True)
            return [
                self._record(o.object_name, o.etag, o.size)
                for o in objects
            ]

        return await asyncio.to_thread(_list)


class OSSArtifactStore:
    """Alibaba OSS via the native oss2 SDK. Supports STS security tokens."""

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        security_token: str | None = None,
    ):
        import oss2

        self.backend = "oss"
        self.bucket = bucket
        if security_token:
            auth = oss2.StsAuth(access_key, secret_key, security_token)
        else:
            auth = oss2.Auth(access_key, secret_key)
        self._bucket = oss2.Bucket(auth, endpoint, bucket)

    def _record(self, object_key: str, etag: str | None, size: int | None,
                sha256: str | None = None) -> ArtifactRecord:
        return ArtifactRecord(
            uri=key_to_artifact_uri(object_key),
            backend="oss",
            bucket=self.bucket,
            object_key=object_key,
            etag=etag,
            sha256=sha256,
            size_bytes=size,
        )

    async def put_file(self, local_path: Path, object_key: str) -> ArtifactRecord:
        local_path = Path(local_path)

        def _put():
            digest = _sha256(local_path)
            result = self._bucket.put_object_from_file(object_key, str(local_path))
            return self._record(
                object_key, result.etag, local_path.stat().st_size, digest
            )

        return await asyncio.to_thread(_put)

    async def get_file(self, object_key: str, destination: Path) -> None:
        await asyncio.to_thread(
            self._bucket.get_object_to_file, object_key, str(destination)
        )

    async def exists(self, object_key: str) -> bool:
        return await asyncio.to_thread(self._bucket.object_exists, object_key)

    async def list_prefix(self, prefix: str) -> list[ArtifactRecord]:
        def _list():
            import oss2

            records = []
            for obj in oss2.ObjectIterator(self._bucket, prefix=prefix):
                records.append(self._record(obj.key, obj.etag, obj.size))
            return records

        return await asyncio.to_thread(_list)


def store_from_env() -> ArtifactStore:
    """Build the artifact store from FLASHML_ARTIFACT_* env vars.

    FLASHML_ARTIFACT_BACKEND=minio (default) | oss
    Shared: FLASHML_ARTIFACT_ENDPOINT, FLASHML_ARTIFACT_BUCKET,
            FLASHML_ARTIFACT_ACCESS_KEY, FLASHML_ARTIFACT_SECRET_KEY
    minio:  FLASHML_ARTIFACT_SECURE=true|false, FLASHML_ARTIFACT_REGION
    oss:    FLASHML_ARTIFACT_SECURITY_TOKEN (optional STS)
    """
    backend = os.environ.get("FLASHML_ARTIFACT_BACKEND", "minio")
    endpoint = os.environ["FLASHML_ARTIFACT_ENDPOINT"]
    bucket = os.environ["FLASHML_ARTIFACT_BUCKET"]
    access_key = os.environ["FLASHML_ARTIFACT_ACCESS_KEY"]
    secret_key = os.environ["FLASHML_ARTIFACT_SECRET_KEY"]

    if backend == "oss":
        return OSSArtifactStore(
            endpoint=endpoint,
            bucket=bucket,
            access_key=access_key,
            secret_key=secret_key,
            security_token=os.environ.get("FLASHML_ARTIFACT_SECURITY_TOKEN") or None,
        )
    if backend == "minio":
        return S3CompatibleArtifactStore(
            endpoint=endpoint,
            bucket=bucket,
            access_key=access_key,
            secret_key=secret_key,
            secure=os.environ.get("FLASHML_ARTIFACT_SECURE", "false").lower() == "true",
            region=os.environ.get("FLASHML_ARTIFACT_REGION") or None,
        )
    raise ValueError(f"unknown artifact backend: {backend!r} (expected minio|oss)")
