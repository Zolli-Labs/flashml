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

# Per-user, not system-wide. /var/lib/flashnode was the default and is not
# writable by an ordinary account on macOS or Windows, so the very first
# command a volunteer runs died with
#   PermissionError: [Errno 13] Permission denied: '/var/lib/flashnode'
# before enrolment could start, and with nothing to suggest that a directory
# choice was the problem.
#
# ~/.flashnode also matches where the credential store already writes
# (identity/credentials.py), so a machine's identity and its token sit
# together and `rm -rf ~/.flashnode` fully un-enrols it.
#
# The Kubernetes profile sets FLASHNODE_STATE_DIR explicitly
# (flashml-cloud/infra/base/flashnode.yaml) to a hostPath volume, so there
# the ID still survives pod restarts and still identifies the *node*.
DEFAULT_STATE_DIR = "~/.flashnode"


def state_dir() -> Path:
    raw = os.environ.get("FLASHNODE_STATE_DIR") or DEFAULT_STATE_DIR
    return Path(raw).expanduser()


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
