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


# -- unpack_inputs -----------------------------------------------------------
#
# The field that carries "extract this input into a directory before the
# command runs" from the submitter to flashnode. flashnode already honours
# it; until the recipe forwarded it, nothing in the system ever emitted one,
# so every repo job's `python /work/inputs/code/train.py` looked for its
# entrypoint inside a gzip blob and failed with "file not found".


def _job_with(params: dict) -> object:
    """A sandboxed command job whose workload parameters carry `params`."""
    spec = _job()
    spec.spec.workload.parameters.update(params)
    return spec


def _payload(params: dict) -> dict:
    return CommandRecipe().expand("job-a", _job_with(params))[0].payload


def test_unpack_inputs_is_forwarded_into_the_task_payload():
    payload = _payload({
        "inputs": {"code": "artifact://repos/abc.tar.gz"},
        "unpack_inputs": ["code"],
    })
    assert payload["unpack_inputs"] == ["code"]
    assert payload["inputs"] == {"code": "artifact://repos/abc.tar.gz"}


def test_absent_unpack_inputs_leaves_the_payload_key_absent():
    """Absent, not `[]`.

    flashnode branches on `payload.get("unpack_inputs")`, and the missing-key
    branch is the one where every input keeps its plain-file behaviour byte
    for byte. An empty list would mean the same thing today and would stop
    that path from ever being exercised by a real payload.
    """
    payload = _payload({"inputs": {"data": "artifact://sets/x.npz"}})
    assert "unpack_inputs" not in payload


def test_unpack_inputs_is_copied_not_aliased():
    """The payload must not share a list with the spec: a later mutation of
    one would silently rewrite the other."""
    params = {
        "inputs": {"code": "artifact://repos/abc.tar.gz"},
        "unpack_inputs": ["code"],
    }
    payload = _payload(params)
    params["unpack_inputs"].append("other")
    assert payload["unpack_inputs"] == ["code"]


@pytest.mark.parametrize(
    "value",
    ["code", {"code": True}, ["code", 3], [None]],
    ids=["bare-string", "dict", "non-string-member", "none-member"],
)
def test_unpack_inputs_must_be_a_list_of_strings(value):
    with pytest.raises(ValueError, match="list of input names"):
        CommandRecipe().expand("job-a", _job_with({
            "inputs": {"code": "artifact://repos/abc.tar.gz"},
            "unpack_inputs": value,
        }))


def test_unpack_inputs_must_name_a_declared_input():
    """Fail on the submitter's side.

    flashnode refuses this too — but only after the task has been expanded,
    leased and claimed, so the submitter sees a node-side failure instead of
    the spec error it is.
    """
    with pytest.raises(ValueError, match="not declared in 'inputs'"):
        CommandRecipe().expand("job-a", _job_with({
            "inputs": {"data": "artifact://sets/x.npz"},
            "unpack_inputs": ["code"],
        }))


def test_unpack_inputs_rejects_duplicates():
    with pytest.raises(ValueError, match="more than once"):
        CommandRecipe().expand("job-a", _job_with({
            "inputs": {"code": "artifact://repos/abc.tar.gz"},
            "unpack_inputs": ["code", "code"],
        }))


@pytest.mark.parametrize(
    "unpack, match",
    [
        ("code", "list of input names"),
        (["nope"], "not declared in 'inputs'"),
        (["code", "code"], "more than once"),
    ],
    ids=["not-a-list", "undeclared", "duplicate"],
)
def test_bad_unpack_inputs_surfaces_as_an_expansion_error(unpack, match):
    """Through the service's expansion entry point, where a recipe's
    ValueError becomes the ExpansionError the API turns into a 422 — not a
    500, and not a task that leases and then dies on a volunteer's box."""
    from flashruntime.service.modea import ExpansionError, expand_tasks

    with pytest.raises(ExpansionError, match=match):
        expand_tasks("job-a", _job_with({
            "inputs": {"code": "artifact://repos/abc.tar.gz"},
            "unpack_inputs": unpack,
        }))
