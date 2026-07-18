from __future__ import annotations

import importlib
from typing import Any, Callable, Protocol

from flashruntime.adapters.base import Storage

EntrypointFn = Callable[[dict[str, Any], Storage], dict[str, Any]]


class SupportsEntrypoint(Protocol):
    def __call__(self, payload: dict[str, Any], storage: Storage) -> dict[str, Any]: ...


def resolve_entrypoint(entrypoint: str) -> EntrypointFn:
    """Load a task handler from a 'module.path:function' string.

    This is the one place a connector is allowed to know about algorithm code
    -- it never inspects payloads, it just invokes whatever entrypoint the
    algorithm declared, the same way a RunPod connector would deploy that
    dotted path as the body of an Endpoint.
    """
    module_name, _, func_name = entrypoint.partition(":")
    if not module_name or not func_name:
        raise ValueError(f"invalid entrypoint {entrypoint!r}, expected 'module.path:function'")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise ValueError(f"entrypoint {entrypoint!r}: no function {func_name!r} in {module_name!r}") from None
