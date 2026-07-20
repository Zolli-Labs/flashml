"""Kubernetes/KubeRay environment smoke checks.

These verify the *test environment contract* the KubeRay backend assumes
(cluster reachable, KubeRay CRDs installed, flashml nodes labeled) — so a
failing end-to-end run can be split into "environment broken" vs "backend
broken". The backend's manifest-generation logic itself is unit-tested in
tests/test_kuberay_backend.py with no cluster at all.
"""

from __future__ import annotations

import json
import subprocess

import pytest


def _kubectl_json(*args: str) -> dict:
    out = subprocess.run(
        ["kubectl", *args, "-o", "json"], capture_output=True, timeout=20, check=True
    )
    return json.loads(out.stdout)


def test_kuberay_crds_present(kuberay_available):
    crds = _kubectl_json("get", "crd")
    names = {item["metadata"]["name"] for item in crds["items"]}
    assert "rayjobs.ray.io" in names
    assert "rayclusters.ray.io" in names


def test_flashml_compute_nodes_labeled(kubernetes_available):
    """The local profile labels worker nodes for scheduling; without them a
    RayJob would pend forever with no obvious cause."""
    nodes = _kubectl_json("get", "nodes", "-l", "flashml.dev/compute=true")
    if not nodes["items"]:
        pytest.skip("no nodes labeled flashml.dev/compute=true (not the flashml-poc cluster?)")
    assert all(
        any(c["type"] == "Ready" and c["status"] == "True" for c in n["status"]["conditions"])
        for n in nodes["items"]
    )
