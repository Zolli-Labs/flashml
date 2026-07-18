# AGENTS.md — flashruntime

Context for AI coding agents (Claude Code, Codex) working in this repository.

## What this repo is

The **open-source fault-tolerant distributed ML runtime** of the FlashML
system (Zolli Labs). It owns the public protocol: job specs, task graphs,
leases, heartbeats, checkpoint manifests, failure taxonomy, recovery state
machine, adapters, CLI/SDK. Full product context: `docs/SYSTEM_OVERVIEW.md`.

Sibling repos (cloned side-by-side under `~/Work/Zolli-Labs/`):
- `../flashnode` — public host agent; depends on this repo.
- `../flashml-cloud` — private managed control plane; depends on this repo.

## Hard rules

1. **Dependency direction:** this repo imports NOTHING from `flashnode` or
   `flashml-cloud`. Ever. They import us.
2. **This repo goes public at launch** (Apache-2.0). No secrets, no keys, no
   private business logic, no references to private-repo internals in code or
   commit history. Secrets only via `.env` (gitignored).
3. **Schemas are versioned.** Any wire-visible message or spec carries a
   schema version. Security-relevant fields fail closed.
4. **The runtime must stay useful without the cloud** — self-hosted local
   coordinator is a first-class mode, not a demo shim.
5. Recovery actions are typed, deterministic, and logged. No LLM-driven
   recovery decisions.

## Current state (July 2026)

- `engine/`, `algorithms/`, `adapters/`, `storage/` = working prototype
  (provider-agnostic training with a local thread-pool backend), carried over
  from the original FlashML library. Tests pass: `pytest`.
- `protocol/`, `leases/`, `checkpoint/`, `recovery/`, `scheduler/` =
  docstring-only scaffolds for the target architecture. Build here; migrate
  prototype code in as the protocol lands.

## Dev workflow

```bash
uv venv && uv pip install -e ".[sklearn,dev]"
pytest                    # 8 tests + docs link checker
```

Python ≥3.10, Pydantic for new schemas, pytest for everything. Match existing
code style; keep modules small and boundaries explicit.
