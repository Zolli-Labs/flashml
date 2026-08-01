"""Per-coordinator credential store for the device profile.

Tokens are keyed by *normalized* coordinator URL (trailing slash stripped) so
one machine can join several pools without one `flashnode login` clobbering
another. The file is JSON, chmod 0600 after every write. A corrupt or
unparseable file must never crash the agent — it simply behaves as if no
token were saved (`load_token` returns `None`).

This is the manual-token half. The interactive device flow (browser code)
needs a server and arrives with the cloud API plan; it will reuse this store
and the `CoordinatorClient` header plumbing that reads from it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CREDENTIALS_PATH = "~/.flashnode/credentials.json"


def credentials_path() -> Path:
    raw = os.environ.get("FLASHNODE_CREDENTIALS") or DEFAULT_CREDENTIALS_PATH
    return Path(raw).expanduser()


def _normalize(coordinator: str) -> str:
    return coordinator.rstrip("/")


def _read_all() -> dict[str, str]:
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _write_all(tokens: dict[str, str]) -> Path:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokens, indent=2, sort_keys=True))
    os.chmod(path, 0o600)
    return path


def save_token(coordinator: str, token: str) -> Path:
    tokens = _read_all()
    tokens[_normalize(coordinator)] = token
    return _write_all(tokens)


def load_token(coordinator: str) -> str | None:
    return _read_all().get(_normalize(coordinator))


def clear_token(coordinator: str) -> bool:
    tokens = _read_all()
    key = _normalize(coordinator)
    if key not in tokens:
        return False
    del tokens[key]
    _write_all(tokens)
    return True
