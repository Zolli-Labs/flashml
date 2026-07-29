from flashnode.inventory.capabilities import (
    classify_environment,
    discover,
    filter_node_labels,
)
from flashruntime.protocol.v1alpha1 import NodeEnvironment


def test_environment_classification(monkeypatch):
    monkeypatch.setenv("FLASHNODE_ENVIRONMENT", "edge")
    assert classify_environment() is NodeEnvironment.EDGE
    monkeypatch.setenv("FLASHNODE_ENVIRONMENT", "nonsense")
    assert classify_environment() is NodeEnvironment.LOCAL  # fail closed to local
    monkeypatch.delenv("FLASHNODE_ENVIRONMENT")
    assert classify_environment() is NodeEnvironment.LOCAL


def test_label_filtering_only_allows_known_namespaces():
    labels = {
        "flashml.dev/pool": "local",
        "kubernetes.io/arch": "arm64",
        "topology.kubernetes.io/zone": "z1",
        "secret-internal/label": "nope",
        "app": "nope",
    }
    filtered = filter_node_labels(labels)
    assert "flashml.dev/pool" in filtered
    assert "kubernetes.io/arch" in filtered
    assert "topology.kubernetes.io/zone" in filtered
    assert "secret-internal/label" not in filtered
    assert "app" not in filtered


def test_discover_prefers_kubernetes_allocatable(monkeypatch):
    monkeypatch.setenv("FLASHNODE_ENVIRONMENT", "local")
    node_meta = {
        "metadata": {"labels": {"kubernetes.io/arch": "arm64",
                                "flashml.dev/pool": "local"}},
        "status": {"allocatable": {"cpu": "5500m", "memory": "9Gi"}},
    }
    reg = discover("fn-test1", "kind-worker", node_meta)
    assert reg.capabilities.cpu_cores == 5.5
    assert reg.capabilities.memory_bytes == 9 * 1024**3
    assert reg.capabilities.architecture == "arm64"
    assert reg.pool == "local"
    assert reg.kubernetes_node == "kind-worker"
    assert reg.schema_version == "v1alpha1"


def test_discover_degrades_without_kubernetes(monkeypatch):
    monkeypatch.delenv("FLASHNODE_SANDBOX_CAPABLE", raising=False)
    reg = discover("fn-test2", "", None)
    assert reg.capabilities.cpu_cores and reg.capabilities.cpu_cores > 0
    assert reg.capabilities.memory_bytes and reg.capabilities.memory_bytes > 0
    assert reg.sandbox_capable is False
    assert reg.capabilities.gpus == []  # never guessed


def test_argv_capable_defaults_false():
    reg = discover("node-1", kubernetes_node="", node_meta=None)
    assert reg.argv_capable is False


def test_argv_capable_when_requested():
    reg = discover("node-1", kubernetes_node="", node_meta=None, argv_capable=True)
    assert reg.argv_capable is True


def test_argv_capable_implies_sandbox_capable(monkeypatch):
    # ArgvDockerRunner is container-only by construction — there is no
    # unsandboxed way to run it — so a node advertising argv_capable must
    # also advertise sandbox_capable, or the coordinator's tier gate for
    # `tier: "sandboxed"` command tasks will never let it claim work.
    monkeypatch.delenv("FLASHNODE_SANDBOX_CAPABLE", raising=False)
    reg = discover("node-1", kubernetes_node="", node_meta=None, argv_capable=True)
    assert reg.sandbox_capable is True


def test_argv_incapable_node_keeps_sandbox_capable_false(monkeypatch):
    # No regression: without argv and without the env/label sources, a node
    # must not be advertised as sandbox-capable.
    monkeypatch.delenv("FLASHNODE_SANDBOX_CAPABLE", raising=False)
    reg = discover("node-1", kubernetes_node="", node_meta=None, argv_capable=False)
    assert reg.sandbox_capable is False


def test_env_and_label_sandbox_capable_still_work_without_argv(monkeypatch):
    # The existing env-var and label paths must still set sandbox_capable
    # when argv_capable is False (e.g. --runner docker on a sandboxed host).
    monkeypatch.setenv("FLASHNODE_SANDBOX_CAPABLE", "true")
    reg = discover("node-1", kubernetes_node="", node_meta=None, argv_capable=False)
    assert reg.sandbox_capable is True

    monkeypatch.delenv("FLASHNODE_SANDBOX_CAPABLE", raising=False)
    node_meta = {"metadata": {"labels": {"flashml.dev/sandbox-capable": "true"}}}
    reg = discover("node-1", kubernetes_node="", node_meta=node_meta, argv_capable=False)
    assert reg.sandbox_capable is True
