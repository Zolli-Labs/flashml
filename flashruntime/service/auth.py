"""Who is this caller? — the node-authentication seam.

Deliberately knows nothing about leases or authorization. This module answers
only "which node_id does this token belong to"; what that node may WRITE is a
separate question answered by the lease store (service/modea.py). Keeping them
apart is what lets the cloud replace authentication (Plan 3) without touching
authorization.

Default is OPEN: `flashruntime/CLAUDE.md` rule 4 makes the self-hosted local
coordinator a first-class mode, and requiring credentials on a laptop would
break it. An operator exposing the coordinator turns enforcement on with
FLASHML_NODE_TOKENS, and FLASHML_REQUIRE_NODE_AUTH=1 makes startup fail closed
if they forget (see service/modea.py).
"""

from __future__ import annotations

import hmac
from typing import Protocol, runtime_checkable

__all__ = [
    "AuthConfigError",
    "NodeAuthenticator",
    "OpenAuthenticator",
    "StaticTokenAuthenticator",
    "authenticator_from_env",
]


class AuthConfigError(RuntimeError):
    """The authenticator configuration is unusable. Raised at construction —
    never at request time, so a misconfiguration cannot silently admit
    callers."""


@runtime_checkable
class NodeAuthenticator(Protocol):
    @property
    def enforcing(self) -> bool:
        """True when callers must present a valid token."""

    def authenticate(self, token: str | None) -> str | None:
        """Return the caller's node_id, or None to deny."""


class OpenAuthenticator:
    """Self-hosted default: no credentials, no scoping. Behavior identical to
    the coordinator before this seam existed."""

    @property
    def enforcing(self) -> bool:
        return False

    def authenticate(self, token: str | None) -> str | None:  # noqa: ARG002
        return None


class StaticTokenAuthenticator:
    """Token → node_id from configuration. The self-hosted multi-machine case,
    and the test double for the cloud's authenticator."""

    @property
    def enforcing(self) -> bool:
        return True

    def __init__(self, tokens: dict[str, str]):
        for token, node_id in tokens.items():
            if not token:
                raise AuthConfigError(
                    f"empty token configured for node {node_id!r}: an empty token "
                    "would authenticate every caller that sends none"
                )
        self._tokens = dict(tokens)

    def authenticate(self, token: str | None) -> str | None:
        if not token:
            return None
        # compare_digest against every candidate: a dict lookup leaks token
        # contents through timing, and the candidate set is small.
        for candidate, node_id in self._tokens.items():
            if hmac.compare_digest(candidate, token):
                return node_id
        return None


def authenticator_from_env(env: dict[str, str] | None = None) -> NodeAuthenticator:
    import os

    env = os.environ if env is None else env
    raw = (env.get("FLASHML_NODE_TOKENS") or "").strip()
    if not raw:
        return OpenAuthenticator()

    tokens: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if pair.count(":") != 1:
            raise AuthConfigError(
                f"FLASHML_NODE_TOKENS entry {pair!r} is not 'node_id:token'"
            )
        node_id, token = (p.strip() for p in pair.split(":"))
        if token in tokens:
            raise AuthConfigError(
                f"duplicate token shared by {tokens[token]!r} and {node_id!r}: "
                "shared tokens make attribution and revocation meaningless"
            )
        tokens[token] = node_id
    if not tokens:
        return OpenAuthenticator()
    return StaticTokenAuthenticator(tokens)
