"""Stable FlashNode identity.

One ID per node, created on first run and persisted in the state directory
(a hostPath volume in the Kubernetes profile, so the ID survives pod
restarts and represents the *node*, not the pod). Ed25519 signing keys are a
documented future step — the POC identity is an opaque random ID.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

DEFAULT_STATE_DIR = "/var/lib/flashnode"


def state_dir() -> Path:
    return Path(os.environ.get("FLASHNODE_STATE_DIR", DEFAULT_STATE_DIR))


def load_or_create_node_id() -> str:
    path = state_dir() / "node-id"
    if path.exists():
        node_id = path.read_text().strip()
        if node_id:
            return node_id
    node_id = f"fn-{uuid.uuid4().hex[:16]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(node_id + "\n")
    tmp.replace(path)
    return node_id
