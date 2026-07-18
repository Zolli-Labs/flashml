from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from flashruntime.adapters.base import Storage, StorageError


class LocalStorage(Storage):
    """Blob store backed by a temp directory. Keys are '/'-separated paths,
    e.g. '{job_id}/shard_0.npy' -- mirrors the layout a remote NetworkVolume
    connector would use."""

    def __init__(self, root: str | None = None):
        self._root = Path(root) if root else Path(tempfile.mkdtemp(prefix="flashruntime-"))
        self._owns_root = root is None

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if self._root.resolve() not in path.parents and path != self._root.resolve():
            raise StorageError(f"key escapes storage root: {key!r}")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise StorageError(f"no such key: {key!r}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete_prefix(self, prefix: str) -> None:
        path = self._path(prefix)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()

    def close(self) -> None:
        if self._owns_root and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
