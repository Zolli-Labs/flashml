"""`FLASHNODE_LOCAL_DATA` — a host owner lends a directory to tasks by LABEL.

    FLASHNODE_LOCAL_DATA="patients=/srv/data/patients-2026,labs=/srv/labs"

Two invariants carry the whole feature and are what these tests exist to pin:

1. **The data never moves.** Nothing is uploaded; the coordinator learns the
   label NAMES and never the host paths. A path is host-private — it leaks the
   owner's directory layout, their username, and often the dataset's identity.
2. **A label is a name, never a path fragment.** It is matched against the
   owner's map and used as one container-side directory segment; it is never
   joined to a host path. The charset test below keeps that true even if
   someone later changes the assumption.

The mount assertions read the CONSTRUCTED argv, so they need no Docker daemon
— same style as test_hardening.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from flashnode.config.local_data import (
    LocalDataError,
    load_local_data,
    parse_local_data,
)
from flashnode.executor.argv_runner import ArgvDockerRunner
from flashnode.executor.hardening import CONTAINER_WORKDIR, harden_args
from flashnode.executor.runner import TaskExecutionError
from flashnode.inventory.capabilities import discover

ENV = "FLASHNODE_LOCAL_DATA"


# -- parsing ------------------------------------------------------------------


def test_parses_label_path_pairs():
    assert parse_local_data("a=/x,b=/y") == {"a": "/x", "b": "/y"}


def test_parses_a_single_pair_and_tolerates_surrounding_whitespace():
    assert parse_local_data("  patients = /srv/data/patients-2026 ") == {
        "patients": "/srv/data/patients-2026"
    }


@pytest.mark.parametrize("raw", [None, "", "   ", ",", " , "])
def test_unset_or_empty_is_no_local_data(raw):
    """The default path: a host that never opted in advertises nothing and
    mounts nothing."""
    assert parse_local_data(raw) == {}


def test_load_local_data_reads_the_env_var(monkeypatch):
    monkeypatch.setenv(ENV, "labs=/srv/labs")
    assert load_local_data() == {"labs": "/srv/labs"}
    monkeypatch.delenv(ENV)
    assert load_local_data() == {}


@pytest.mark.parametrize(
    "raw",
    [
        "good=/x,garbage",       # an entry with no '=' at all
        "good=/x,=/y",           # empty label
        "good=/x,b=",            # empty path
        "good=/x,b=relative",    # a relative path resolves against the agent's
                                 # cwd, which is not what the owner meant
    ],
)
def test_a_malformed_entry_is_refused_not_partially_accepted(raw):
    """Half-applying the owner's intent is the dangerous outcome: they believe
    two directories are exposed, the agent exposes one, and the mismatch shows
    up as an unplaceable job hours later. Refuse the whole value."""
    with pytest.raises(LocalDataError):
        parse_local_data(raw)


def test_a_duplicate_label_is_refused():
    """Two paths cannot both be `patients`; silently keeping the last one
    would expose a directory the owner did not intend for that name."""
    with pytest.raises(LocalDataError):
        parse_local_data("patients=/srv/a,patients=/srv/b")


@pytest.mark.parametrize(
    "label",
    ["pa/tients", "..", ".", "../../etc", "pat ients", "pat*", "pat:ients", "él"],
)
def test_a_label_outside_the_safe_charset_is_rejected(label):
    """A label is never joined to a filesystem path — but it IS used as one
    container-side directory segment, and it is the key an untrusted payload
    looks up. Restricting it to [A-Za-z0-9._-] (and never '.'/'..') means a
    later refactor that DOES join it to a path cannot escape a directory."""
    with pytest.raises(LocalDataError) as exc:
        parse_local_data(f"{label}=/srv/x")
    assert label in str(exc.value)


def test_a_path_that_cannot_be_a_bind_mount_source_is_refused():
    """`docker -v` splits its argument on ':'. A host path containing one
    would silently reinterpret as src:dst:opts — refuse it at parse time
    rather than construct a mount that means something else."""
    with pytest.raises(LocalDataError):
        parse_local_data("labs=/srv/we:ird")


# -- advertising: names travel, paths do not ----------------------------------


def test_discover_advertises_the_label_names(monkeypatch):
    monkeypatch.setenv(ENV, "patients=/srv/data/patients-2026,labs=/srv/labs")
    reg = discover("node-1", kubernetes_node="", node_meta=None)
    assert reg.local_datasets == ["labs", "patients"]   # sorted: stable on the wire


def test_discover_never_sends_the_host_paths(monkeypatch):
    """A host path is host-private. It must not travel to the coordinator in
    any field — not local_datasets, not labels, not by accident."""
    monkeypatch.setenv(ENV, "patients=/srv/data/patients-2026,labs=/srv/labs")
    reg = discover("node-1", kubernetes_node="", node_meta=None)
    wire = reg.model_dump_json()
    assert "/srv/data/patients-2026" not in wire
    assert "/srv/labs" not in wire
    assert "/srv" not in wire
    assert "patients" in wire and "labs" in wire        # the names DID travel


def test_discover_advertises_nothing_when_the_owner_opted_out(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert discover("node-1", kubernetes_node="", node_meta=None).local_datasets == []


def test_discover_refuses_to_start_on_a_malformed_value(monkeypatch):
    """Fail closed, like config/HostPolicy: the owner must learn their typo
    now, not by wondering why their dataset is never used."""
    monkeypatch.setenv(ENV, "patients=/srv/a,garbage")
    with pytest.raises(LocalDataError):
        discover("node-1", kubernetes_node="", node_meta=None)


# -- mounting -----------------------------------------------------------------


def test_requested_label_is_mounted_read_only_at_inputs(tmp_path):
    args = harden_args(
        tmp_path, cpus=1.0, memory_gb=1.0,
        local_inputs=["patients"],
        local_data={"patients": "/srv/data/patients-2026", "labs": "/srv/labs"},
    )
    assert f"/srv/data/patients-2026:{CONTAINER_WORKDIR}/inputs/patients:ro" in args
    # exactly one -v beyond the workdir mount, and the label the task did NOT
    # ask for is not mounted
    assert not any("/srv/labs" in a for a in args)


def test_the_mount_is_read_only_not_merely_present(tmp_path):
    """Read-only is the entire consent model: the owner lends the data, they
    do not hand a stranger's code write access to it."""
    args = harden_args(
        tmp_path, cpus=1.0, memory_gb=1.0,
        local_inputs=["patients"], local_data={"patients": "/srv/p"},
    )
    mount = next(a for a in args if a.startswith("/srv/p:"))
    assert mount.endswith(":ro")


def test_several_labels_all_mount(tmp_path):
    args = harden_args(
        tmp_path, cpus=1.0, memory_gb=1.0,
        local_inputs=["patients", "labs"],
        local_data={"patients": "/srv/p", "labs": "/srv/l"},
    )
    assert f"/srv/p:{CONTAINER_WORKDIR}/inputs/patients:ro" in args
    assert f"/srv/l:{CONTAINER_WORKDIR}/inputs/labs:ro" in args


def test_a_label_the_host_has_not_mapped_is_refused_and_named(tmp_path):
    with pytest.raises(TaskExecutionError) as exc:
        harden_args(
            tmp_path, cpus=1.0, memory_gb=1.0,
            local_inputs=["patients"], local_data={"labs": "/srv/l"},
        )
    assert "patients" in str(exc.value)


def test_a_hostile_label_in_the_payload_is_refused(tmp_path):
    """The payload is attacker-influenced from job submission onward. It must
    not be able to name a mount destination of its choosing."""
    for label in ["../../root", "/etc", "a b", ".."]:
        with pytest.raises(TaskExecutionError):
            harden_args(
                tmp_path, cpus=1.0, memory_gb=1.0,
                local_inputs=[label], local_data={"labs": "/srv/l"},
            )


def test_local_inputs_must_be_a_list_of_strings(tmp_path):
    for bad in ["patients", 3, [1], {"patients": "/x"}]:
        with pytest.raises(TaskExecutionError):
            harden_args(
                tmp_path, cpus=1.0, memory_gb=1.0,
                local_inputs=bad, local_data={"patients": "/srv/p"},
            )


def test_the_default_path_is_byte_for_byte_unchanged(tmp_path, monkeypatch):
    """No FLASHNODE_LOCAL_DATA, no local_inputs: every existing host must
    produce exactly the argv it produced before this feature existed."""
    monkeypatch.delenv(ENV, raising=False)
    before = harden_args(tmp_path, cpus=2.0, memory_gb=4.0)
    for extra in ({}, {"local_inputs": None}, {"local_inputs": []}):
        assert harden_args(tmp_path, cpus=2.0, memory_gb=4.0, **extra) == before


def test_no_local_inputs_mounts_nothing_even_when_the_host_offers_data(tmp_path,
                                                                       monkeypatch):
    monkeypatch.setenv(ENV, "patients=/srv/p")
    args = harden_args(tmp_path, cpus=1.0, memory_gb=1.0)
    assert not any("/srv/p" in a for a in args)


def test_a_broken_host_config_only_fails_the_tasks_that_need_it(tmp_path, monkeypatch):
    """The owner's typo must not take down every unrelated task on the
    machine — the map is read only when a task actually asks for a label."""
    monkeypatch.setenv(ENV, "garbage")
    harden_args(tmp_path, cpus=1.0, memory_gb=1.0)              # no request: fine
    with pytest.raises(TaskExecutionError):
        harden_args(tmp_path, cpus=1.0, memory_gb=1.0, local_inputs=["patients"])


def test_the_host_map_defaults_to_the_env_var(tmp_path, monkeypatch):
    """The runners do not have to know where the map comes from — the owner's
    env var is the single source."""
    monkeypatch.setenv(ENV, "patients=/srv/p")
    args = harden_args(tmp_path, cpus=1.0, memory_gb=1.0, local_inputs=["patients"])
    assert f"/srv/p:{CONTAINER_WORKDIR}/inputs/patients:ro" in args


# -- the runners actually forward the request ---------------------------------
#
# A mount is only as real as the payload key reaching it: without these the
# feature would be "implemented" in hardening.py and dead in every runner.


def _argv_payload(**over):
    base = {"argv": ["python", "train.py"], "image": "ghcr.io/zolli/trainer:1.0",
            "task_id": "task-000", "local_inputs": ["patients"]}
    base.update(over)
    return base


def _fake_ok_run(outdir: Path):
    def run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            (outdir / "metrics.json").write_text("{}")
        return mock.Mock(returncode=0, stdout=b"", stderr=b"")
    return run


def test_argv_runner_forwards_local_inputs_to_the_mount(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "patients=/srv/p")
    runner = ArgvDockerRunner(allowed_images=frozenset({"ghcr.io/zolli/trainer:1.0"}))
    with mock.patch("subprocess.run", side_effect=_fake_ok_run(tmp_path / "out")) as sp:
        runner.run(_argv_payload(), tmp_path, {})
    cmd = sp.call_args_list[0].args[0]
    assert f"/srv/p:{CONTAINER_WORKDIR}/inputs/patients:ro" in cmd


def test_argv_runner_refuses_a_task_this_host_cannot_serve(tmp_path, monkeypatch):
    """This node should never have been leased the task (the coordinator's
    placement gate reads the same advertisement), so reaching here means the
    two disagree — fail the task, do not run it half-fed."""
    monkeypatch.setenv(ENV, "labs=/srv/l")
    runner = ArgvDockerRunner(allowed_images=frozenset({"ghcr.io/zolli/trainer:1.0"}))
    with mock.patch("subprocess.run") as sp:
        with pytest.raises(TaskExecutionError, match="patients"):
            runner.run(_argv_payload(), tmp_path, {})
    sp.assert_not_called()      # refused before docker was ever invoked


def test_argv_runner_unchanged_without_local_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "patients=/srv/p")
    runner = ArgvDockerRunner(allowed_images=frozenset({"ghcr.io/zolli/trainer:1.0"}))
    payload = _argv_payload()
    payload.pop("local_inputs")
    with mock.patch("subprocess.run", side_effect=_fake_ok_run(tmp_path / "out")) as sp:
        runner.run(payload, tmp_path, {})
    assert not any("/srv/p" in a for a in sp.call_args_list[0].args[0])


def test_docker_runner_forwards_local_inputs_to_the_mount(tmp_path, monkeypatch):
    from flashnode.executor.docker_runner import DockerRunner

    monkeypatch.setenv(ENV, "patients=/srv/p")
    runner = DockerRunner(allowed_images=frozenset({"ghcr.io/zolli/trainer:1.0"}))
    payload = {"module": "flashml_workloads.sgd_trainer",
               "image": "ghcr.io/zolli/trainer:1.0", "task_id": "t1",
               "local_inputs": ["patients"]}
    with mock.patch("subprocess.run", side_effect=_fake_ok_run(tmp_path / "out")) as sp:
        runner.run(payload, tmp_path, {})
    cmd = sp.call_args_list[0].args[0]
    assert f"/srv/p:{CONTAINER_WORKDIR}/inputs/patients:ro" in cmd


def test_nothing_about_the_host_path_reaches_the_task_spec(tmp_path, monkeypatch):
    """DockerRunner writes spec.json for the workload. The task sees the
    container path; the host path is not the task's business either."""
    from flashnode.executor.docker_runner import DockerRunner

    monkeypatch.setenv(ENV, "patients=/srv/data/patients-2026")
    runner = DockerRunner(allowed_images=frozenset({"ghcr.io/zolli/trainer:1.0"}))
    payload = {"module": "flashml_workloads.sgd_trainer",
               "image": "ghcr.io/zolli/trainer:1.0", "task_id": "t1",
               "local_inputs": ["patients"]}
    with mock.patch("subprocess.run", side_effect=_fake_ok_run(tmp_path / "out")):
        runner.run(payload, tmp_path, {})
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert "/srv/data/patients-2026" not in json.dumps(spec)
