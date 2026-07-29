"""Hardware/software discovery for the node the agent runs on.

psutil + platform probes; Kubernetes allocatable values (when available from
the API) take precedence over raw host numbers because they are what the
scheduler can actually place on the node.
"""

from __future__ import annotations

import os
import platform
import socket

import psutil

from flashruntime.protocol.v1alpha1 import (
    NodeCapabilities,
    NodeEnvironment,
    NodeRegistration,
)

# Kubernetes reports arch as amd64/arm64; platform.machine() says x86_64/arm64.
_ARCH_MAP = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64", "amd64": "amd64"}

# Only labels in these namespaces are reported upstream — never arbitrary
# cluster metadata.
_ALLOWED_LABEL_PREFIXES = ("flashml.dev/", "kubernetes.io/arch", "kubernetes.io/os",
                           "node.kubernetes.io/instance-type",
                           "topology.kubernetes.io/")


def classify_environment() -> NodeEnvironment:
    value = os.environ.get("FLASHNODE_ENVIRONMENT", "local").lower()
    try:
        return NodeEnvironment(value)
    except ValueError:
        return NodeEnvironment.LOCAL


def filter_node_labels(labels: dict[str, str]) -> dict[str, str]:
    return {
        k: v for k, v in labels.items()
        if any(k.startswith(p) for p in _ALLOWED_LABEL_PREFIXES)
    }


def _parse_k8s_cpu(value: str) -> float:
    return float(value[:-1]) / 1000 if value.endswith("m") else float(value)


def _parse_k8s_memory(value: str) -> int:
    units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
             "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4}
    for suffix, mult in units.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * mult)
    return int(value)


def discover(node_id: str, kubernetes_node: str,
             node_meta: dict | None = None,
             argv_capable: bool = False) -> NodeRegistration:
    """Build the registration payload. `node_meta` is the Kubernetes Node
    object (status/metadata) when the agent has API access; None degrades to
    host-level probes only."""
    allocatable = (node_meta or {}).get("status", {}).get("allocatable", {})
    labels = filter_node_labels((node_meta or {}).get("metadata", {}).get("labels", {}))

    if allocatable:
        cpu = _parse_k8s_cpu(allocatable.get("cpu", "0"))
        memory = _parse_k8s_memory(allocatable.get("memory", "0"))
    else:
        cpu = float(psutil.cpu_count() or 0)
        memory = psutil.virtual_memory().total

    arch = _ARCH_MAP.get(platform.machine().lower(), platform.machine().lower())
    environment = classify_environment()
    sandbox_capable = (
        os.environ.get("FLASHNODE_SANDBOX_CAPABLE", "").lower() == "true"
        or labels.get("flashml.dev/sandbox-capable") == "true"
    )

    from flashnode import __version__

    return NodeRegistration(
        node_id=node_id,
        kubernetes_node=kubernetes_node,
        hostname=socket.gethostname(),
        capabilities=NodeCapabilities(
            cpu_cores=cpu,
            memory_bytes=memory,
            gpus=[],  # GPU probing is a documented follow-up; never guess.
            os=labels.get("kubernetes.io/os", platform.system().lower()),
            architecture=labels.get("kubernetes.io/arch", arch),
        ),
        environment=environment,
        sandbox_capable=sandbox_capable,
        # Set by the agent when it is actually running an argv-capable
        # runner — never inferred, so the coordinator's fail-closed gate
        # cannot be satisfied by a node that merely has docker installed.
        argv_capable=argv_capable,
        pool=labels.get("flashml.dev/pool", os.environ.get("FLASHNODE_POOL", "local")),
        runtime_profile=os.environ.get("FLASHNODE_RUNTIME_PROFILE", "kubernetes"),
        labels=labels,
        agent_version=__version__,
    )
