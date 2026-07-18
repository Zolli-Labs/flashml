# AGENTS.md — flashnode

Context for AI coding agents (Claude Code, Codex) working in this repository.

## What this repo is

The **open-source host agent** of the FlashML system (Zolli Labs). It runs on
contributors' machines and executes third-party ML tasks, so it must be
inspectable, minimal, and explicit about permissions. Full product context:
`docs/SYSTEM_OVERVIEW.md`.

Sibling repos (cloned side-by-side under `~/Work/Zolli-Labs/`):
- `../flashruntime` — public protocol + runtime. **This repo will depend on
  it** for all wire schemas (add the dependency when the protocol package
  lands; install editable: `uv pip install -e ../flashruntime -e .`).
- `../flashml-cloud` — private control plane. We talk to it only through the
  versioned flashruntime protocol. **Never import it.**

## Hard rules

1. **This repo goes public at launch** (Apache-2.0). No secrets, no private
   business logic, nothing in history you wouldn't publish.
2. **Wire messages come from `flashruntime.protocol`** — never define a
   duplicate schema here, never copy one from flashml-cloud.
3. **Security contract is non-negotiable** (see README): outbound-only
   connection, non-root execution, allowlisted images, explicit resource
   limits, no host Docker socket, no privileged mode. Any change that
   loosens these needs explicit human sign-off.
4. The host owner sees exactly what a workload may do before it runs.
5. Trust through transparency: log every task assignment, image digest,
   permission grant, and artifact commit.

## Current state (July 2026)

Docstring-only scaffold: each subpackage's `__init__.py` documents what it
will contain. `agent/cli.py` is a stub entry point. No agent loop yet —
implementation order is identity → inventory → agent connection → executor →
telemetry → artifacts → benchmark.

## Dev workflow

```bash
uv venv && uv pip install -e ".[dev]"
flashnode            # prints usage from the stub CLI
pytest               # once tests exist
```

Python ≥3.10, asyncio, psutil, websockets, cryptography (Ed25519). Keep the
agent dependency-light — every dependency is attack surface on someone
else's machine.
