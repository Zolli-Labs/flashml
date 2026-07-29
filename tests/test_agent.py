import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def test_argv_runner_requires_an_image_allowlist(monkeypatch, capsys):
    """Refuse to start rather than silently degrade to an unsandboxed tier."""
    from flashnode.agent.cli import main
    monkeypatch.delenv("FLASHNODE_ALLOWED_IMAGES", raising=False)
    assert main(["work", "--runner", "argv"]) == 2
    assert "FLASHNODE_ALLOWED_IMAGES" in capsys.readouterr().err
