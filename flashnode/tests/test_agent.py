import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from flashnode.agent.daemon import Agent


class _CloudStub(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict]] = []
    fail_first_n = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if _CloudStub.fail_first_n > 0:
            _CloudStub.fail_first_n -= 1
            self.send_response(500)
            self.end_headers()
            return
        _CloudStub.requests.append((self.path, body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def _run_stub():
    server = HTTPServer(("127.0.0.1", 0), _CloudStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def make_agent(monkeypatch, tmp_path, port):
    monkeypatch.setenv("FLASHNODE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHNODE_CLOUD_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("K8S_NODE_NAME", "test-node")
    return Agent()


def test_register_retries_until_cloud_accepts(monkeypatch, tmp_path):
    server = _run_stub()
    _CloudStub.requests = []
    _CloudStub.fail_first_n = 2  # first two attempts 500 -> retry with backoff
    agent = make_agent(monkeypatch, tmp_path, server.server_port)
    agent.register()
    server.shutdown()
    paths = [p for p, _ in _CloudStub.requests]
    assert paths == ["/v1alpha1/nodes/register"]
    body = _CloudStub.requests[0][1]
    assert body["node_id"] == agent.node_id
    assert body["kubernetes_node"] == "test-node"
    assert body["schema_version"] == "v1alpha1"


def test_heartbeat_and_graceful_termination(monkeypatch, tmp_path):
    server = _run_stub()
    _CloudStub.requests = []
    _CloudStub.fail_first_n = 0
    agent = make_agent(monkeypatch, tmp_path, server.server_port)
    assert agent.heartbeat_once() is True
    assert agent.heartbeat_once(status="terminating") is True
    server.shutdown()
    (p1, hb1), (p2, hb2) = _CloudStub.requests
    assert p1.endswith(f"/nodes/{agent.node_id}/heartbeat")
    assert hb1["status"] == "online"
    assert hb2["status"] == "terminating"


def test_heartbeat_failure_is_reported_not_raised(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path, 1)  # nothing listens on port 1
    assert agent.heartbeat_once() is False


def test_argv_runner_starts_without_an_image_allowlist_env_var(monkeypatch, capsys):
    """A volunteer sets no env var.

    The built-in namespace allowlist is what they run on, so an unset
    FLASHNODE_ALLOWED_IMAGES must NOT be a refusal to start. It used to be,
    which capped the fleet at the number of owners willing to hand-maintain a
    list of image references — and stranded every host on the old image each
    time we published a new one.

    Exercised by removing the `docker` binary so the CLI still exits early;
    what is asserted is *which* gate stopped it. The allowlist gate is gone,
    the docker-binary gate remains.
    """
    import shutil

    from flashnode.agent.cli import main

    monkeypatch.delenv("FLASHNODE_ALLOWED_IMAGES", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    assert main(["work", "--runner", "argv"]) == 2
    err = capsys.readouterr().err
    assert "docker" in err
    assert "FLASHNODE_ALLOWED_IMAGES" not in err


@pytest.mark.parametrize("runner,expect_module_capable", [
    ("argv", False),      # F1: an argv-only volunteer has no module runner
    ("docker", True),
    ("subprocess", True),
])
def test_work_cli_sets_module_capable_from_runner_choice(
    monkeypatch, tmp_path, runner, expect_module_capable
):
    """flashnode work must advertise module_capable=False only for
    --runner argv — a docker/subprocess node still runs module tasks fine."""
    import flashnode.inventory.capabilities as capabilities_mod
    from flashnode.agent import cli
    from flashnode.executor import CoordinatorClient, ExecutorLoop

    monkeypatch.setenv("FLASHNODE_ALLOWED_IMAGES", "img:1")
    monkeypatch.setenv("FLASHNODE_STATE_DIR", str(tmp_path))

    # --runner docker and --runner argv both refuse to start on a host that
    # cannot run tasks — `flashnode work` now runs the doctor's checks as a
    # fail-closed gate (flashnode/doctor.py) instead of the old
    # `shutil.which("docker") is None`. That gate is correct and has its own
    # tests in test_work_gate.py; here it is environment noise standing
    # between us and the behaviour under test, which is purely how
    # module_capable is derived from the runner choice.
    #
    # Report a healthy host. Without this the test passes on a developer
    # machine with Docker AND the probe image cached, and fails everywhere
    # else — which is the shape of the bug that first hit flashnode in CI on
    # macOS (GitHub's macOS runners ship no Docker daemon).
    #
    # Stubbing run_checks rather than `shutil.which` because `which` is now
    # only the first of six checks; faking it alone lets the gate proceed to
    # `docker version` and a real `docker run`.
    monkeypatch.setattr("flashnode.doctor.run_checks", lambda **kw: [])

    captured = {}
    real_discover = capabilities_mod.discover

    def spy_discover(*args, **kwargs):
        reg = real_discover(*args, **kwargs)
        captured["module_capable"] = reg.module_capable
        return reg

    monkeypatch.setattr(capabilities_mod, "discover", spy_discover)
    monkeypatch.setattr(CoordinatorClient, "register", lambda self, reg: None)
    monkeypatch.setattr(ExecutorLoop, "run", lambda self, max_tasks=None: 0)

    rc = cli.main(["work", "--runner", runner, "--coordinator", "http://localhost:1"])
    assert rc == 0
    assert captured["module_capable"] is expect_module_capable
