# tests/test_viewer_server.py
"""The stdlib run viewer: state.collect() (the /api/state assembler) and
RunViewerServer (the read-only HTTP surface a submit opens on the run dir).

The through-line of every test here is the exception-safety contract: a
viewer reads a LIVE run's directory, so it must render a partial snapshot of
any on-disk state (torn file, missing dir, half-written last line) and never
raise — the run's story must never be interrupted or killed by someone
looking at it.
"""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request

from flashruntime.checkpoint.local import write_manifest
from flashruntime.viewer import collect
from flashruntime.viewer.server import RunViewerServer


def _fake_run(tmp_path, *, contract="viewer_v1", metrics_lines=3, with_ckpt=True):
    """Build a run dir shaped exactly like sdk.Run writes it: run.json at the
    root, a per-attempt dir (`<job>/<attempt>/`) holding metrics.jsonl +
    launcher.log, and the job ckpt root as that dir's sibling
    (`<job>/ckpt/step-*/manifest.json`)."""
    job, attempt = "local", "task-000"
    attempt_dir = tmp_path / job / attempt
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": i, "loss": 1.0 / (i + 1)}) + "\n" for i in range(metrics_lines))
    )
    (attempt_dir / "launcher.log").write_text("epoch 0\nepoch 1\ndone\n")
    if with_ckpt:
        step_dir = tmp_path / job / "ckpt" / "step-000010"
        step_dir.mkdir(parents=True)
        (step_dir / "model.pt").write_bytes(b"weights")
        write_manifest(step_dir, job_id=job, attempt_id=attempt, step=10)
    doc = {
        "contract": contract,
        "workload": {"command": ["python", "train.py"], "mode": "coordinated", "source": "/src"},
        "state": "SUCCEEDED",
        "started_at": 1.0,
        "finished_at": 2.0,
        "max_restarts": 0,
        "attempts": [
            {
                "attempt_id": attempt,
                "job_id": job,
                "state": "SUCCEEDED",
                "pid": "123",
                "started_at": 1.0,
                "finished_at": 2.0,
                "output_dir": str(attempt_dir),
            }
        ],
        "events": [{"ts": 1.0, "type": "LAUNCH_STARTED", "message": "task-000 launched"}],
        "trials": [{"loss": 0.5}],
    }
    (tmp_path / "run.json").write_text(json.dumps(doc, indent=2))
    return tmp_path


# --------------------------------------------------------------------------
# state.collect
# --------------------------------------------------------------------------


def test_collect_assembles_snapshot(tmp_path):
    snap = collect(_fake_run(tmp_path))
    assert snap["contract"] == "viewer_v1"
    assert snap["state"] == "SUCCEEDED"
    # per-attempt metrics tail (whatever keys exist are passed through)
    metrics = snap["attempts"][0]["metrics"]
    assert [m["step"] for m in metrics] == [0, 1, 2]
    assert "loss" in metrics[0]
    # launcher.log tail
    assert "done" in snap["attempts"][0]["log_tail"]
    # checkpoint manifest listed with step + re-verified validation + parts
    ck = snap["checkpoints"]
    assert [m["step"] for m in ck] == [10]
    assert ck[0]["validation"] == "hash_verified"
    assert ck[0]["parts"] == 1
    assert ck[0]["latest_valid"] is True


def test_collect_metrics_tail_capped_at_500(tmp_path):
    run_dir = _fake_run(tmp_path, metrics_lines=650)
    snap = collect(run_dir)
    metrics = snap["attempts"][0]["metrics"]
    assert len(metrics) == 500  # last 500 points, not all 650
    assert metrics[-1]["step"] == 649


def test_collect_torn_run_json_returns_error_not_crash(tmp_path):
    (tmp_path / "run.json").write_text('{"contract": "viewer_v1", "attempts": [')  # half a doc
    snap = collect(tmp_path)  # must NOT raise
    assert "error" in snap


def test_collect_missing_run_json_returns_error(tmp_path):
    snap = collect(tmp_path / "does-not-exist")
    assert "error" in snap


def test_collect_unknown_contract_returns_error(tmp_path):
    snap = collect(_fake_run(tmp_path, contract="viewer_v99"))
    assert "error" in snap
    assert "contract" in snap["error"]


def test_collect_survives_torn_manifest(tmp_path):
    """A torn manifest.json in one step dir must not drop the valid one or
    crash the snapshot — the partial-snapshot rule applied to checkpoints."""
    run_dir = _fake_run(tmp_path)  # has valid step-000010
    torn = tmp_path / "local" / "ckpt" / "step-000020"
    torn.mkdir(parents=True)
    (torn / "manifest.json").write_text("{ not json")
    snap = collect(run_dir)
    assert [m["step"] for m in snap["checkpoints"]] == [10]  # valid one survives


def test_collect_survives_missing_attempt_dir(tmp_path):
    """An attempt whose output_dir was never created (or vanished) yields an
    empty metrics/log, never an exception."""
    run_dir = _fake_run(tmp_path)
    doc = json.loads((run_dir / "run.json").read_text())
    doc["attempts"][0]["output_dir"] = str(tmp_path / "ghost")
    (run_dir / "run.json").write_text(json.dumps(doc))
    snap = collect(run_dir)
    assert snap["attempts"][0]["metrics"] == []
    assert snap["attempts"][0]["log_tail"] == ""


# --------------------------------------------------------------------------
# RunViewerServer
# --------------------------------------------------------------------------


def test_server_api_state_matches_collect(tmp_path):
    run_dir = _fake_run(tmp_path)
    server = RunViewerServer(run_dir)
    url = server.start()
    try:
        assert url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(url + "/api/state") as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
        assert body["contract"] == "viewer_v1"
        assert [m["step"] for m in body["checkpoints"]] == [10]
        assert [m["step"] for m in body["attempts"][0]["metrics"]] == [0, 1, 2]
    finally:
        server.stop()


def test_server_root_serves_page(tmp_path):
    server = RunViewerServer(_fake_run(tmp_path))
    url = server.start()
    try:
        with urllib.request.urlopen(url + "/") as resp:
            assert resp.status == 200
            assert b"flashruntime" in resp.read().lower()
    finally:
        server.stop()


def test_server_unknown_path_404(tmp_path):
    server = RunViewerServer(_fake_run(tmp_path))
    url = server.start()
    try:
        try:
            urllib.request.urlopen(url + "/nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.stop()


def test_server_docs_not_built_404(tmp_path):
    # No _docs dir → /docs answers 404 with the honest "docs not built" body.
    server = RunViewerServer(_fake_run(tmp_path), docs_dir=tmp_path / "no-docs")
    url = server.start()
    try:
        try:
            urllib.request.urlopen(url + "/docs")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            assert b"docs not built" in exc.read()
    finally:
        server.stop()


def test_server_serves_docs_when_built(tmp_path):
    docs = tmp_path / "_docs"
    docs.mkdir()
    (docs / "index.html").write_text("<h1>hello docs</h1>")
    server = RunViewerServer(_fake_run(tmp_path), docs_dir=docs)
    url = server.start()
    try:
        with urllib.request.urlopen(url + "/docs/") as resp:
            assert resp.status == 200
            assert b"hello docs" in resp.read()
    finally:
        server.stop()


def test_server_docs_path_traversal_blocked(tmp_path):
    """A crafted /docs/../<secret> must not escape the docs root. Uses a raw
    http.client request so the '..' reaches the server unnormalized (urllib
    would collapse it client-side and hide the bug)."""
    docs = tmp_path / "_docs"
    docs.mkdir()
    (docs / "index.html").write_text("ok")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    server = RunViewerServer(_fake_run(tmp_path), docs_dir=docs)
    url = server.start()
    try:
        host_port = url.removeprefix("http://")
        conn = http.client.HTTPConnection(host_port)
        conn.request("GET", "/docs/../secret.txt")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        assert resp.status == 404
        assert b"TOP SECRET" not in body
    finally:
        server.stop()


def test_server_stop_frees_port(tmp_path):
    server = RunViewerServer(_fake_run(tmp_path))
    url = server.start()
    port = int(url.rsplit(":", 1)[1])
    server.stop()
    # port is free again: we can bind it ourselves right after stop().
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))  # would raise if the server still held it
    finally:
        s.close()
