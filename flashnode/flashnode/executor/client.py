"""Outbound-only HTTP client for the FlashRuntime coordinator.

stdlib `urllib` on purpose: this code runs on strangers' machines, and every
dependency is attack surface (AGENTS.md). All calls are *outbound* to the
coordinator — the device never listens on anything.

Two heartbeats, two endpoints, never merged:
- node heartbeat   → /v1alpha1/nodes/{id}/heartbeat   ("this machine is up")
- attempt heartbeat→ /v1alpha1/attempts/{id}/heartbeat ("this task is alive")
A 410 on the attempt heartbeat means the lease is dead: the worker must stop
work on that task (`LeaseLost`).
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from flashruntime.protocol.v1alpha1 import (
    ExecutionEvidence,
    Lease,
    NodeHeartbeat,
    NodeRegistration,
)


class LeaseLost(Exception):
    """The coordinator refused the attempt heartbeat — stop working."""


class CoordinatorClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 15.0,
        join_code: str | None = None,
        token: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.join_code = join_code  # sent on register when the pool requires one
        self._token = token  # bearer credential for every authenticated call

    # -- auth -----------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Auth header merged into every request this class makes.

        Empty when no token is configured — correct for the self-hosted open
        profile, where absence of a credential simply means no header.
        """
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    # -- transport (patchable in tests) -------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        all_headers = self._headers()
        all_headers.update(headers or {})
        if body is not None:
            all_headers["Content-Type"] = content_type
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers=all_headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def _json(self, method: str, path: str, payload: dict | str | None = None) -> tuple[int, dict]:
        body = None
        if payload is not None:
            body = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
        status, raw = self._request(method, path, body)
        return status, (json.loads(raw) if raw else {})

    # -- node lifecycle -----------------------------------------------------

    def register(self, registration: NodeRegistration) -> None:
        headers = {"X-FlashML-Join-Code": self.join_code} if self.join_code else None
        status, raw = self._request(
            "POST",
            "/v1alpha1/nodes/register",
            registration.model_dump_json().encode(),
            headers=headers,
        )
        if status != 200:
            raise RuntimeError(f"registration failed ({status}): {raw.decode(errors='replace')}")

    def node_heartbeat(self, node_id: str) -> bool:
        status, _ = self._json(
            "POST",
            f"/v1alpha1/nodes/{node_id}/heartbeat",
            NodeHeartbeat(node_id=node_id).model_dump_json(),
        )
        return status == 200

    # -- lease lifecycle ----------------------------------------------------

    def claim(self, node_id: str) -> Lease | None:
        status, body = self._json("POST", "/v1alpha1/leases/claim", {"node_id": node_id})
        if status == 204:
            return None
        if status != 200:
            raise RuntimeError(f"claim failed ({status}): {body}")
        return Lease.model_validate(body)

    def attempt_heartbeat(self, lease_id: str) -> None:
        status, body = self._json("POST", f"/v1alpha1/attempts/{lease_id}/heartbeat")
        if status == 410:
            raise LeaseLost(str(body))
        if status != 200:
            raise RuntimeError(f"attempt heartbeat failed ({status}): {body}")

    def complete(
        self,
        lease_id: str,
        output_sha256: str,
        evidence: ExecutionEvidence | None = None,
    ) -> bool:
        """Commit this attempt, optionally saying what the run looked like.

        `evidence` is a keyword with a default so every existing caller keeps
        working, and the body is byte-for-byte the one previous releases sent
        when there is nothing to report — this agent may be talking to a
        coordinator that predates the field, and an empty block is not the
        same statement as no block.
        """
        payload: dict = {"output_sha256": output_sha256}
        if evidence is not None:
            payload["evidence"] = evidence.model_dump(mode="json")
        status, body = self._json(
            "POST", f"/v1alpha1/attempts/{lease_id}/complete", payload
        )
        if status != 200:
            raise RuntimeError(f"complete failed ({status}): {body}")
        return bool(body.get("accepted"))

    def fail(self, lease_id: str, reason: str) -> None:
        self._json("POST", f"/v1alpha1/attempts/{lease_id}/fail", {"reason": reason})

    # -- checkpoints (the agent is the courier; tasks stay offline) ----------

    def checkpoint_latest(self, job_id: str, task_id: str) -> dict | None:
        status, body = self._json(
            "GET", f"/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints/latest"
        )
        if status == 404:
            return None
        if status != 200:
            raise RuntimeError(f"checkpoint latest failed ({status}): {body}")
        return body

    def checkpoint_register_part(
        self, job_id: str, task_id: str, attempt_id: str, step: int, part: dict
    ) -> None:
        status, body = self._json(
            "POST",
            f"/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints/parts",
            {"attempt_id": attempt_id, "step": step, "part": part},
        )
        if status != 200:
            raise RuntimeError(f"checkpoint part registration failed ({status}): {body}")

    def checkpoint_commit(
        self,
        job_id: str,
        task_id: str,
        attempt_id: str,
        step: int,
        parts: list[dict],
        storage_prefix: str,
    ) -> dict:
        status, body = self._json(
            "POST",
            f"/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints/commit",
            {
                "attempt_id": attempt_id,
                "step": step,
                "expected_parts": parts,
                "storage_prefix": storage_prefix,
            },
        )
        if status != 200:
            raise RuntimeError(f"checkpoint commit refused ({status}): {body}")
        return body

    # -- artifacts (coordinator-hosted shared data) --------------------------

    def download_artifact(self, key: str, destination: Path) -> Path:
        status, raw = self._request("GET", f"/v1alpha1/artifacts/{key}")
        if status != 200:
            raise RuntimeError(f"artifact download failed ({status}): {key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return destination

    def upload_artifact(self, local_path: Path, key: str) -> str:
        data = Path(local_path).read_bytes()
        status, _ = self._request(
            "PUT", f"/v1alpha1/artifacts/{key}", data, content_type="application/octet-stream"
        )
        if status != 200:
            raise RuntimeError(f"artifact upload failed ({status}): {key}")
        return hashlib.sha256(data).hexdigest()
