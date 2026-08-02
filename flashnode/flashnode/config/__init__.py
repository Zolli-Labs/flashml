"""Host-owner policy: the machine's owner has the last word.

`HostPolicy` is the contract between FlashNode and the person whose
machine it runs on: which resources tasks may use, which code may run,
and how much of the machine FlashML may occupy. It loads from
`<state_dir>/policy.json` (the same state dir that holds the node id) so
the owner edits one file, and `flashnode status` *(planned)* can show
exactly which limits apply — AGENTS hard rule 4: the owner sees what a
workload may do before it runs.

Safety rules baked into this module:
- **Conservative defaults**: a policy-less install is the *most*
  restrictive configuration, not the least (1 task at a time, no network
  for tasks, modest quotas). Generosity is always an explicit owner act.
- **Fail closed on garbage**: an unparseable/invalid policy.json raises —
  the agent refuses to start rather than silently running with defaults
  the owner believes they overrode.
- **Owner overrides only narrow, never widen, safety**: `allowed_modules`
  /`allowed_images` = None defers to the built-in allowlists; a non-None
  value INTERSECTS with them at enforcement time (an owner can forbid
  more, never permit code the build doesn't trust).

Wiring points for the implementer (each a small, separate slice):
- `agent/cli.py _work`: load_host_policy(state_dir) → cap ExecutorLoop
  concurrency, pass quotas to runners;
- runners: enforce `max_task_cpus`/`max_task_memory_gb` (docker flags
  already exist; the subprocess tier gains rlimits);
- executor workdir: enforce `workdir_quota_gb` before download (fixes the
  "no disk-space guard" debt in HANDBOOK §6).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

__all__ = ["HostPolicy", "load_host_policy"]


class HostPolicy(BaseModel):
    """Everything a host owner may restrict. All defaults are deliberately
    tight — see module notes."""

    max_concurrent_tasks: int = Field(
        default=1, ge=1, description="Parallel task slots this machine offers"
    )
    max_task_cpus: float = Field(
        default=2.0, gt=0, description="CPU cores any single task may use"
    )
    max_task_memory_gb: float = Field(
        default=2.0, gt=0, description="RAM any single task may use"
    )
    workdir_quota_gb: float = Field(
        default=5.0,
        gt=0,
        description="Max disk per task workdir (inputs + outputs + checkpoints)",
    )
    allow_network_tasks: bool = Field(
        default=False,
        description="Whether tasks may have network access. False keeps the "
        "docker tier's --network none and (once enforced) sandboxes the "
        "subprocess tier; flipping this is a real trust decision",
    )
    allowed_modules: list[str] | None = Field(
        default=None,
        description="None = defer to the built-in module allowlist; a list "
        "INTERSECTS with it (owner can only narrow)",
    )
    allowed_images: list[str] | None = Field(
        default=None,
        description="Same narrowing rule as allowed_modules, for the docker tier",
    )
    active_hours: tuple[int, int] | None = Field(
        default=None,
        description="Optional (start_hour, end_hour) local-time window in "
        "which the agent claims work — outside it the loop idles but keeps "
        "heartbeating so the node shows as present-but-unavailable",
    )


def load_host_policy(state_dir: str | Path) -> HostPolicy:
    """Load `<state_dir>/policy.json`, defaults when absent, loud when broken.

    Returns: the owner's HostPolicy, or `HostPolicy()` if no file exists
    (defaults ARE the no-file behavior — documented, conservative).
    Raises: ValueError naming policy.json for unparseable JSON or invalid
    fields — never silently substitutes defaults over an owner's broken
    edit (they believed they had set limits; honoring that belief matters
    more than starting up).
    """
    path = Path(state_dir) / "policy.json"
    if not path.exists():
        return HostPolicy()
    try:
        raw = json.loads(path.read_text())
        return HostPolicy.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"invalid host policy at {path}: {exc} — fix or delete the file "
            "(deleting restores the conservative defaults)"
        ) from exc
