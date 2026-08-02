"""FlashNode agent loop for the Kubernetes profile.

One agent per node (DaemonSet). It never launches workload containers, never
touches a container runtime socket, and only makes *outbound* HTTP calls to
FlashML Cloud: register once, then heartbeat. On SIGTERM it reports
`terminating` and exits.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request

from flashruntime.protocol.v1alpha1 import NodeHeartbeat, NodeRegistration

from flashnode.agent.kube import get_own_node
from flashnode.identity.store import load_or_create_node_id
from flashnode.inventory.capabilities import discover

log = logging.getLogger("flashnode")
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"flashnode","msg":%(message)s}',
)


def _jlog(msg: str, **kv) -> str:
    return json.dumps({"text": msg, **kv})


def _post(url: str, payload: str, timeout: float = 10) -> None:
    req = urllib.request.Request(
        url, data=payload.encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout):
        pass


class Agent:
    def __init__(self) -> None:
        self.cloud_url = os.environ.get("FLASHNODE_CLOUD_URL", "http://localhost:8000").rstrip("/")
        self.heartbeat_seconds = float(os.environ.get("FLASHNODE_HEARTBEAT_SECONDS", "10"))
        self.node_id = load_or_create_node_id()
        self.kubernetes_node = os.environ.get("K8S_NODE_NAME", "")
        self._stop = False

    def registration(self) -> NodeRegistration:
        node_meta = get_own_node(self.kubernetes_node)
        return discover(self.node_id, self.kubernetes_node, node_meta)

    def register(self) -> None:
        reg = self.registration()
        backoff = 1.0
        while not self._stop:
            try:
                _post(f"{self.cloud_url}/v1alpha1/nodes/register", reg.model_dump_json())
                log.info(_jlog("registered", node_id=self.node_id,
                               k8s_node=self.kubernetes_node,
                               pool=reg.pool, arch=reg.capabilities.architecture))
                return
            except (urllib.error.URLError, OSError) as exc:
                log.warning(_jlog("registration failed; retrying",
                                  error=str(exc), backoff_s=backoff))
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def heartbeat_once(self, status: str = "online") -> bool:
        hb = NodeHeartbeat(node_id=self.node_id, status=status)
        try:
            _post(f"{self.cloud_url}/v1alpha1/nodes/{self.node_id}/heartbeat",
                  hb.model_dump_json())
            return True
        except (urllib.error.URLError, OSError) as exc:
            log.warning(_jlog("heartbeat failed", error=str(exc)))
            return False

    def _handle_term(self, signum, frame) -> None:  # noqa: ARG002
        self._stop = True

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._handle_term)
        signal.signal(signal.SIGINT, self._handle_term)
        log.info(_jlog("flashnode agent starting", node_id=self.node_id,
                       cloud=self.cloud_url, k8s_node=self.kubernetes_node))
        self.register()
        while not self._stop:
            self.heartbeat_once()
            deadline = time.monotonic() + self.heartbeat_seconds
            while not self._stop and time.monotonic() < deadline:
                time.sleep(0.25)
        # Graceful goodbye — best effort, one attempt.
        self.heartbeat_once(status="terminating")
        log.info(_jlog("flashnode agent stopped", node_id=self.node_id))
        return 0


def main() -> int:
    return Agent().run()


if __name__ == "__main__":
    sys.exit(main())
