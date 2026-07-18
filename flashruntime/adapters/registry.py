from __future__ import annotations

from typing import Callable

from flashruntime.adapters.base import Provider

_REGISTRY: dict[str, Callable[..., Provider]] = {}


def register(name: str) -> Callable[[type[Provider]], type[Provider]]:
    def decorator(cls: type[Provider]) -> type[Provider]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_provider(name: str, **kwargs) -> Provider:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown provider '{name}'. Registered providers: {sorted(_REGISTRY)}"
        ) from None
    return factory(**kwargs)


def registered_providers() -> list[str]:
    return sorted(_REGISTRY)
