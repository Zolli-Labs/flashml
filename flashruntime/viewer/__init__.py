"""The stdlib run viewer: a read-only window onto a live run's directory.

Two pieces, both stdlib-only so `import flashruntime.viewer` stays clean-core
importable (no numpy/torch/fastapi): `state.collect()` assembles the
`/api/state` snapshot from disk, and `server.RunViewerServer` serves it (plus
the page and the docs) over HTTP. The only flashruntime dependency is
`flashruntime.checkpoint.local` for manifest hash-verification — itself
core-safe (stdlib + pydantic).

Contract source: the `viewer_v1` run.json written by `flashruntime.sdk.Run`.
The viewer consumes that versioned contract and NOTHING else about the SDK's
internals, so the two evolve independently (spec §2b, horizontal
extensibility).
"""

from __future__ import annotations

from flashruntime.viewer.state import collect

__all__ = ["collect"]
