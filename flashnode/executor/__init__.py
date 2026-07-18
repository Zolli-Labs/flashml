"""Sandboxed task execution.

Pulls allowlisted container images and runs task attempts non-root with
explicit CPU/memory/disk/time limits. No host Docker socket, no privileged
mode. Docker/rootless OCI first; gVisor/Kata tiers later.
"""
