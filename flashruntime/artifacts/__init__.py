"""Backend-neutral artifact storage.

Public artifact identity is always an `artifact://` URI; the physical
location (MinIO bucket locally, OSS bucket on Alibaba) is an ArtifactRecord
detail. Implementations: S3CompatibleArtifactStore (MinIO and S3-compatible
endpoints) and OSSArtifactStore (native Alibaba OSS via oss2).
"""

from flashruntime.artifacts.store import (
    ArtifactStore,
    OSSArtifactStore,
    S3CompatibleArtifactStore,
    artifact_uri_to_key,
    key_to_artifact_uri,
    store_from_env,
)

__all__ = [
    "ArtifactStore",
    "OSSArtifactStore",
    "S3CompatibleArtifactStore",
    "artifact_uri_to_key",
    "key_to_artifact_uri",
    "store_from_env",
]
