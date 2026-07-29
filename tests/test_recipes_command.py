"""CommandRecipe.expand() must refuse to expand a command job that would run
unsandboxed: argv execution is container-only, and the placement gate (see
tests/test_service_command_recipe.py) only protects nodes that were never
handed an unsandboxed command task in the first place. This is the
submission-side complement to that placement-side gate.
"""

from __future__ import annotations

import pytest

from flashruntime.protocol.v1alpha1 import ImageSpec, IsolationSpec
from flashruntime.recipes.command import CommandRecipe
from flashruntime.workloads.command import CommandWorkload, to_jobspec


def _job(tier="sandboxed", allow_fallback=False):
    wl = CommandWorkload(
        command="python train.py",
        image=ImageSpec(repository="ghcr.io/zolli/trainer", tag="1.0"),
        isolation=IsolationSpec(tier=tier, allowFallback=allow_fallback),
    )
    return to_jobspec(wl, name="j")


def test_sandboxed_tier_is_accepted():
    assert CommandRecipe().expand("job-a", _job()) != []


def test_standard_tier_rejected_by_default():
    with pytest.raises(ValueError, match="sandboxed"):
        CommandRecipe().expand("job-a", _job(tier="standard"))


def test_standard_tier_allowed_with_coordinator_opt_in(monkeypatch):
    """Deliberately a coordinator-side env var: a submitter must never be
    able to downgrade the isolation their own code runs under."""
    monkeypatch.setenv("FLASHML_ALLOW_UNSANDBOXED_ARGV", "1")
    assert CommandRecipe().expand("job-a", _job(tier="standard")) != []


def test_allow_fallback_rejected_for_command_jobs():
    with pytest.raises(ValueError, match="allowFallback"):
        CommandRecipe().expand("job-a", _job(allow_fallback=True))
