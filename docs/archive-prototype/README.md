> **ARCHIVED (July 2026).** These documents describe the pre-Kubernetes
> prototype engine (`engine/`, `algorithms/`, `adapters/`, `storage/`),
> which still works behind the `[prototype]` extra but is not the
> product's direction. Current architecture: workspace `HANDBOOK.md` and
> `docs/SYSTEM_OVERVIEW.md`. Kept for reference only.

# Prototype documentation (carried over)

These documents were written for the FlashML prototype library — the code
that now lives in this repo as `engine/`, `algorithms/`, `adapters/`
(formerly `providers/`), and `storage/`. They describe that working system
and its original connector-library roadmap.

**Status:** historical but useful. The product direction has since widened —
see [`../SYSTEM_OVERVIEW.md`](../SYSTEM_OVERVIEW.md) for the current
three-product architecture (FlashNode / FlashRuntime / FlashML Cloud). Where
these documents and the system overview disagree, the system overview wins.
Package paths written as `flashml.*` correspond to `flashruntime.*` here.

Start at [`INDEX.md`](INDEX.md).
