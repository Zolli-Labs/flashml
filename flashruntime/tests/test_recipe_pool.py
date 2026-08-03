"""Pool must survive the JobSpec → task-payload hop, and the waiver is
coupled to it. The local_inputs/gpus comments in recipes/command.py warn
that both ends of this hop have tests that pass while it is broken — so
these tests exercise the real expander, not a hand-built payload."""

from __future__ import annotations

import pytest

from flashruntime.protocol.v1alpha1 import JobSpec
from flashruntime.recipes.command import CommandRecipe


def _spec(pool="any", allow_fallback=False) -> JobSpec:
    return JobSpec.model_validate({
        "metadata": {"name": "pool-job"},
        "spec": {
            "image": {"repository": "ghcr.io/zolli-labs/flashml-python-slim",
                      "tag": "2026.08.1"},
            "workload": {
                "type": "command",
                "parameters": {"command": ["python", "train.py"]},
            },
            "resources": {"minimumWorkers": 1, "maximumWorkers": 1},
            "placement": {"pool": pool},
            "isolation": {"tier": "sandboxed", "allowFallback": allow_fallback},
        },
    })


def test_pool_is_stamped_into_every_task_payload():
    tasks = CommandRecipe().expand("job-1", _spec(pool="p-1", allow_fallback=True))
    assert all(t.payload["pool"] == "p-1" for t in tasks)


def test_any_stays_absent_never_stamped():
    """Absent stays absent — the no-pool path must keep exercising the
    key-missing branch, exactly as gpus and local_inputs do."""
    tasks = CommandRecipe().expand("job-1", _spec(pool="any"))
    assert all("pool" not in t.payload for t in tasks)


def test_waiver_without_a_pool_is_refused():
    with pytest.raises(ValueError, match="allowFallback"):
        CommandRecipe().expand("job-1", _spec(pool="any", allow_fallback=True))


def test_waiver_with_a_pool_is_accepted_and_travels():
    tasks = CommandRecipe().expand("job-1", _spec(pool="p-1", allow_fallback=True))
    assert tasks[0].payload["isolation"] == {"tier": "sandboxed", "allowFallback": True}
