"""Executor checkpoint relay — written before the implementation (TDD).

Tasks are network-isolated by design (`--network none` in the docker tier),
so the *agent* is the checkpoint courier: before a run it fetches the
task's latest valid checkpoint from the coordinator and hands it to the
task as the `resume` input; during the run it ships every new checkpoint
file (upload → register part → commit manifest). The payoff scenario, with
the real sgd_trainer: attempt 1 crashes at step 6 having checkpointed step
4 — attempt 2 (any machine) resumes from 4 and finishes, bit-identical.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from flashnode.executor import CoordinatorClient, ExecutorLoop, SubprocessRunner

TRAINER = "flashml_workloads.sgd_trainer"

DATASET = b"1.0,1.0,0\n2.0,1.0,0\n3.0,3.0,1\n4.0,2.0,1\n0.0,1.0,0\n5.0,4.0,1\n2.0,4.0,1\n1.0,0.0,0\n"


class CheckpointStub:
    """Coordinator double speaking the lease + artifact + checkpoint wire."""

    def __init__(self):
        self.leases: list[dict] = []
        self.artifacts: dict[str, bytes] = {"jobs/j1/input/train.csv": DATASET}
        self.completed: list[str] = []
        self.failed: list[str] = []
        self.parts: dict[tuple, dict] = {}  # (job,task,step,key) -> part
        self.manifests: dict[tuple, list[dict]] = {}  # (job,task) -> [manifest]

    def make_lease(self, attempt: int):
        lease = {
            "lease_id": f"ls-t1-a{attempt}", "task_id": "t1", "job_id": "j1",
            "node_id": "n1", "attempt_number": attempt,
            "deadline": (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat(),
            "payload": {
                "module": TRAINER, "task_id": "t1",
                "params": {"steps": 10, "checkpoint_every": 4, "lr": 0.1,
                           "seed": 7, "kill_at_step": 6},
                "inputs": {"dataset": "artifact://jobs/j1/input/train.csv"},
                "output_prefix": "jobs/j1/t1/",
                "checkpoint": {},  # relay on
            },
        }
        self.leases.append(lease)

    def request(self, method, path, body=None, content_type=None):
        ckpt = re.match(r"/v1alpha1/jobs/(.+)/tasks/(.+)/checkpoints/(\w+[-\w]*)", path)
        if ckpt:
            job, task, verb = ckpt.groups()
            if verb == "latest":
                ms = self.manifests.get((job, task), [])
                if not ms:
                    return 404, b'{"detail": "no valid checkpoint"}'
                return 200, json.dumps(max(ms, key=lambda m: m["step"])).encode()
            data = json.loads(body)
            if verb == "parts":
                p = data["part"]
                self.parts[(job, task, data["step"], p["key"])] = p
                return 200, b'{"status": "registered"}'
            if verb == "commit":
                for p in data["expected_parts"]:
                    if (job, task, data["step"], p["key"]) not in self.parts:
                        return 409, b'{"detail": "missing part"}'
                manifest = {"manifest_id": f"ck-{data['step']}", "step": data["step"],
                            "parts": data["expected_parts"], "validation": "hash_verified"}
                self.manifests.setdefault((job, task), []).append(manifest)
                return 200, json.dumps(manifest).encode()
        if path == "/v1alpha1/leases/claim":
            if not self.leases:
                return 204, b""
            return 200, json.dumps(self.leases.pop(0)).encode()
        if "/attempts/" in path and path.endswith("/heartbeat"):
            return 200, b"{}"
        if path.endswith("/complete"):
            self.completed.append(path.split("/")[3])
            return 200, b'{"accepted": true}'
        if path.endswith("/fail"):
            self.failed.append(json.loads(body)["reason"])
            return 200, b"{}"
        if path.startswith("/v1alpha1/artifacts/"):
            key = path.removeprefix("/v1alpha1/artifacts/")
            if method == "PUT":
                self.artifacts[key] = body
                return 200, b"{}"
            return (200, self.artifacts[key]) if key in self.artifacts else (404, b"{}")
        if path.endswith("/nodes/register") or path.endswith("/n1/heartbeat"):
            return 200, b"{}"
        raise AssertionError(f"unexpected {method} {path}")


def _loop(stub, monkeypatch):
    client = CoordinatorClient("http://stub")
    monkeypatch.setattr(
        client, "_request",
        lambda m, p, body=None, content_type="", headers=None: stub.request(m, p, body),
    )
    return ExecutorLoop(
        client, "n1",
        runner=SubprocessRunner(allowed_modules=frozenset({TRAINER})),
        poll_seconds=0.05,
    )


def test_crash_then_cross_attempt_resume_via_relayed_checkpoints(stub_none, monkeypatch):
    stub = CheckpointStub()
    loop = _loop(stub, monkeypatch)

    # attempt 1: fresh run dies at step 6; the relay must still have shipped
    # the step-4 checkpoint before/at death
    stub.make_lease(attempt=1)
    loop.run(max_tasks=1, idle_exit=True)
    assert stub.failed and "exited 3" in stub.failed[0]
    latest = max(stub.manifests[("j1", "t1")], key=lambda m: m["step"])
    assert latest["step"] == 4
    assert "jobs/j1/t1/ckpt/step-000004.json" in stub.artifacts

    # attempt 2: executor downloads the latest checkpoint and the trainer
    # resumes — kill_at_step must not fire on a resumed run
    stub.make_lease(attempt=2)
    accepted = loop.run(max_tasks=1, idle_exit=True)
    assert accepted == 1 and stub.completed == ["ls-t1-a2"]
    metrics = json.loads(stub.artifacts["jobs/j1/t1/metrics.json"])
    assert metrics["resumed"] is True
    assert metrics["started_from_step"] == 4
    assert metrics["steps"] == 10


def test_relay_off_means_no_checkpoint_traffic(stub_none, monkeypatch):
    stub = CheckpointStub()
    lease_payloads_without_checkpoint = {
        "module": TRAINER, "task_id": "t1",
        "params": {"steps": 4, "checkpoint_every": 100, "lr": 0.1, "seed": 7},
        "inputs": {"dataset": "artifact://jobs/j1/input/train.csv"},
        "output_prefix": "jobs/j1/t1/",
    }
    stub.make_lease(attempt=1)
    stub.leases[0]["payload"] = lease_payloads_without_checkpoint
    loop = _loop(stub, monkeypatch)
    assert loop.run(max_tasks=1, idle_exit=True) == 1
    assert stub.manifests == {}  # no checkpoint key in payload → no relay


import pytest  # noqa: E402


@pytest.fixture()
def stub_none():
    """Placeholder fixture so both tests share a uniform signature."""
    return None