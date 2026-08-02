"""Minimal in-cluster Kubernetes API access — stdlib only.

The agent needs exactly one call (GET its own Node object) so it can report
allocatable resources and allow-listed labels. A full kubernetes client
would be attack surface and megabytes of dependencies on a contributor's
machine; a 30-line REST call is inspectable.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


def in_cluster() -> bool:
    return os.path.exists(f"{SA_DIR}/token")


def get_own_node(node_name: str) -> dict | None:
    """Fetch this agent's Node object; None when not in-cluster or on any
    failure (the agent then degrades to host-level probes)."""
    if not in_cluster() or not node_name:
        return None
    try:
        with open(f"{SA_DIR}/token") as f:
            token = f.read().strip()
        host = os.environ["KUBERNETES_SERVICE_HOST"]
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        ctx = ssl.create_default_context(cafile=f"{SA_DIR}/ca.crt")
        req = urllib.request.Request(
            f"https://{host}:{port}/api/v1/nodes/{node_name}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            return json.load(resp)
    except Exception:
        return None
