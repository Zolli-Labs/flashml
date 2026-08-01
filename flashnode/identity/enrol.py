"""Device-code enrolment: claim this machine from a browser.

The volunteer runs one command on the machine they are lending, reads a
short code off the terminal, and types it into a page they are already
signed in to — on any device, typically a phone. The machine never handles
the account password, and the person approving is provably the account
owner because the approval endpoint requires their session.

This is RFC 8628's device authorization grant, which exists for exactly
this shape of problem: a client that can display a code but cannot host a
login. The API implements the server half (`/v1alpha1/device/code`,
`/v1alpha1/device/token`); this is the client half, which was missing —
`flashnode login` required `--token`, a credential the enrolment flow never
hands out, so the documented path could not be walked.

Everything I/O-ish is injected (`http`, `sleep`, `now`) so the polling loop
is testable without a server or real time passing.
"""

from __future__ import annotations

import json
import platform as platform_mod
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

__all__ = [
    "DeviceCodeStart",
    "EnrolmentError",
    "describe_this_machine",
    "request_device_code",
    "poll_for_token",
]

# Cap on how long we keep polling if the server never tells us otherwise.
# The server's own expiry is authoritative; this only bounds the case where
# it hands back something unparseable.
MAX_POLL_SECONDS = 15 * 60
DEFAULT_INTERVAL = 5.0


class EnrolmentError(RuntimeError):
    """Enrolment could not complete. The message is shown to a human."""


class Http(Protocol):
    def __call__(self, url: str, payload: dict) -> tuple[int, dict]:
        """POST `payload` as JSON, return (status, decoded body)."""


@dataclass(frozen=True)
class DeviceCodeStart:
    device_code: str
    user_code: str
    verification_uri: str
    interval: float
    expires_at: str


def _post(url: str, payload: dict) -> tuple[int, dict]:
    """POST JSON with the stdlib. flashnode deliberately has no HTTP
    dependency — every byte a volunteer installs is a byte they did not ask
    for — and `agent/daemon.py` already reaches for urllib for the same
    reason."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        # A 400 is the protocol's "not yet" and carries a body worth
        # reading, so this is control flow, not an error path.
        raw = exc.read() or b"{}"
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"error": raw.decode("utf-8", "replace")[:200]}
    except urllib.error.URLError as exc:
        raise EnrolmentError(
            f"could not reach {url}: {exc.reason}. Check the --coordinator "
            "URL, and that you are online."
        ) from exc


def describe_this_machine() -> tuple[str, str]:
    """(hostname, platform) for the approval screen.

    Purely for the human deciding whether to approve: a list of machines
    reading "fn-9c2a…" tells them nothing, "phongs-macbook-air / macOS-15"
    tells them which laptop they are looking at. Never trusted by the
    server for anything.
    """
    try:
        hostname = socket.gethostname() or "unknown"
    except OSError:
        hostname = "unknown"
    return hostname[:120], platform_mod.platform()[:120]


def request_device_code(
    api_base: str,
    node_id: str,
    hostname: str,
    platform_name: str,
    http: Http = _post,
) -> DeviceCodeStart:
    status, body = http(
        f"{api_base.rstrip('/')}/v1alpha1/device/code",
        {"node_id": node_id, "hostname": hostname, "platform": platform_name},
    )
    if status != 200:
        detail = body.get("detail") or body.get("error") or f"HTTP {status}"
        raise EnrolmentError(f"could not start enrolment: {detail}")
    try:
        return DeviceCodeStart(
            device_code=body["device_code"],
            user_code=body["user_code"],
            verification_uri=body["verification_uri"],
            interval=float(body.get("interval") or DEFAULT_INTERVAL),
            expires_at=str(body.get("expires_at") or ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EnrolmentError(
            f"enrolment response was not in the expected shape: {body!r}"
        ) from exc


def poll_for_token(
    api_base: str,
    device_code: str,
    interval: float = DEFAULT_INTERVAL,
    http: Http = _post,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    max_seconds: float = MAX_POLL_SECONDS,
) -> str:
    """Block until a human approves, then return the machine token.

    The server answers unknown / unapproved / expired / already-redeemed
    with one indistinguishable `authorization_pending`, deliberately, so a
    caller cannot probe which codes exist. That means we cannot report
    *why* nothing happened — only that nothing did — so the timeout message
    has to name the likely causes itself.
    """
    url = f"{api_base.rstrip('/')}/v1alpha1/device/token"
    deadline = now() + max_seconds
    # Never poll faster than the server asked, and never so slowly that an
    # approval sits unnoticed. A server that returns 0 or something absurd
    # must not turn this into a hot loop against its own API.
    wait = min(max(float(interval or DEFAULT_INTERVAL), 1.0), 30.0)

    while True:
        status, body = http(url, {"device_code": device_code})
        if status == 200 and body.get("token"):
            return str(body["token"])
        if status == 400 and body.get("error") == "authorization_pending":
            server_interval = body.get("interval")
            if server_interval:
                wait = min(max(float(server_interval), 1.0), 30.0)
        elif status != 400:
            detail = body.get("detail") or body.get("error") or f"HTTP {status}"
            raise EnrolmentError(f"enrolment failed: {detail}")

        if now() >= deadline:
            raise EnrolmentError(
                "timed out waiting for approval. The code may have expired, "
                "or it was never approved — run `flashnode login` again to "
                "get a fresh one."
            )
        sleep(wait)
