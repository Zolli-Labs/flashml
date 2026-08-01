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

Two credential classes, because two kinds of caller write here:

- **node tokens** (FLASHML_NODE_TOKENS) — a volunteer machine. Identified as
  a `node_id`, and everything it writes is confined to the leases it holds.
- **operator tokens** (FLASHML_OPERATOR_TOKENS) — a *driver*: the federated
  averaging / K-means reducers run inside the trusted cloud API, hold no
  lease, and legitimately write keys (`jobs/{job}/round-NNN/weights.json`,
  shard CSVs) that belong to no task. They are authenticated and attributable
  but not lease-scoped. Volunteers never receive one.

An operator token is deliberately NOT a node: `authenticate()` returns None
for it, so it can never claim, complete, or fail somebody's lease. It only
lifts the lease-scope confinement on writes.
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

    def is_operator(self, token: str | None) -> bool:
        """True for a driver credential: authenticated and attributable, but
        NOT lease-scoped. False for node tokens and for anything unknown."""


class OpenAuthenticator:
    """Self-hosted default: no credentials, no scoping. Behavior identical to
    the coordinator before this seam existed."""

    @property
    def enforcing(self) -> bool:
        return False

    def authenticate(self, token: str | None) -> str | None:  # noqa: ARG002
        return None

    def is_operator(self, token: str | None) -> bool:  # noqa: ARG002
        # Nothing is enforced, so nothing needs the exemption either.
        return False


class StaticTokenAuthenticator:
    """Token → node_id from configuration. The self-hosted multi-machine case,
    and the test double for the cloud's authenticator."""

    @property
    def enforcing(self) -> bool:
        return True

    def __init__(
        self,
        tokens: dict[str, str],
        operator_tokens: dict[str, str] | None = None,
    ):
        for token, node_id in tokens.items():
            if not token:
                raise AuthConfigError(
                    f"empty token configured for node {node_id!r}: an empty token "
                    "would authenticate every caller that sends none"
                )
        operators = dict(operator_tokens or {})
        for token, name in operators.items():
            if not token:
                raise AuthConfigError(
                    f"empty token configured for operator {name!r}: an empty token "
                    "would authenticate every caller that sends none"
                )
            if token in tokens:
                # One string that is both a lease-scoped node and an
                # unscoped driver is a privilege escalation waiting to be
                # found, and it makes revocation ambiguous.
                raise AuthConfigError(
                    f"operator token for {name!r} collides with the node token "
                    f"for {tokens[token]!r}: a credential must belong to exactly "
                    "one class"
                )
        self._tokens = dict(tokens)
        self._operators = operators

    @staticmethod
    def _usable(token: str | None) -> bool:
        """A token we can even compare. `hmac.compare_digest` raises
        TypeError on a non-ASCII `str`, which would turn a malformed bearer
        header into an unauthenticated 500 — and a 500-vs-401 difference is
        an oracle. Deny instead."""
        return isinstance(token, str) and bool(token) and token.isascii()

    def authenticate(self, token: str | None) -> str | None:
        if not self._usable(token):
            return None
        # compare_digest against every candidate: a dict lookup leaks token
        # contents through timing, and the candidate set is small.
        for candidate, node_id in self._tokens.items():
            if hmac.compare_digest(candidate, token):
                return node_id
        return None

    def is_operator(self, token: str | None) -> bool:
        if not self._usable(token):
            return False
        for candidate in self._operators:
            if hmac.compare_digest(candidate, token):
                return True
        return False


def _parse_pairs(raw: str, var: str, shape: str) -> dict[str, str]:
    """`name:token` pairs → `{token: name}`."""
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if pair.count(":") != 1:
            raise AuthConfigError(f"{var} entry {pair!r} is not '{shape}'")
        name, token = (p.strip() for p in pair.split(":"))
        if token in out:
            raise AuthConfigError(
                f"duplicate token shared by {out[token]!r} and {name!r}: "
                "shared tokens make attribution and revocation meaningless"
            )
        out[token] = name
    return out


def authenticator_from_env(env: dict[str, str] | None = None) -> NodeAuthenticator:
    import os

    env = os.environ if env is None else env
    raw = (env.get("FLASHML_NODE_TOKENS") or "").strip()
    raw_ops = (env.get("FLASHML_OPERATOR_TOKENS") or "").strip()
    if not raw and not raw_ops:
        return OpenAuthenticator()

    tokens = _parse_pairs(raw, "FLASHML_NODE_TOKENS", "node_id:token")
    operators = _parse_pairs(raw_ops, "FLASHML_OPERATOR_TOKENS", "name:token")
    if not tokens and not operators:
        return OpenAuthenticator()
    # Configuring ONLY operator tokens still enforces. Ignoring them would
    # leave a coordinator its operator believes is credentialed wide open;
    # a loud "no node can write" beats a silent open door.
    return StaticTokenAuthenticator(tokens, operators)
