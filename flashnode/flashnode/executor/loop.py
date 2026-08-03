"""The device executor loop: FlashNode's Mode A work cycle.

    register → [ claim → download inputs → run (heartbeating) → upload
    outputs → commit ] → repeat

Everything is *pull*: the device makes outbound calls only. While a task
runs, a background thread renews the attempt lease; if the coordinator
answers 410 (lease expired/superseded — e.g. this machine was presumed dead
and the task reassigned), the result is thrown away and never committed —
and even a bug here is caught by the coordinator's idempotent commit, which
rejects late duplicates. Defense in depth: polite client, unforgiving
server.

Failures are *reported, not raised*: a task error calls `fail()` (the task
requeues elsewhere), a coordinator outage backs off and retries. The loop
only exits on stop/max_tasks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from flashruntime.protocol.v1alpha1 import ExecutionEvidence, Lease

from flashnode.executor.archives import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_MEMBERS,
    ArchiveError,
    extract_archive_safely,
)
from flashnode.executor.client import CoordinatorClient, LeaseLost
from flashnode.executor.evidence import ResourceSampler
from flashnode.executor.runner import SubprocessRunner, TaskExecutionError

log = logging.getLogger("flashnode.executor")

#: An input name becomes a directory name under the task's workdir when the
#: input is unpacked, so it has to be one harmless path segment. The payload
#: is attacker-influenced all the way from the job submission, and an input
#: called ``../../.ssh`` must be a refused task, not a written directory.
_SAFE_INPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _jlog(msg: str, **kv) -> str:
    return json.dumps({"text": msg, **kv})


class _AttemptHeartbeat(threading.Thread):
    """Renews one attempt lease until stopped; flags the lease as lost on 410."""

    def __init__(self, client: CoordinatorClient, lease: Lease):
        super().__init__(daemon=True)
        self._client = client
        self._lease = lease
        self._stop = threading.Event()
        window = max(2.0, (lease.deadline.timestamp() - time.time()) / 3.0)
        self._interval = window
        self.lost = False

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._client.attempt_heartbeat(self._lease.lease_id)
            except LeaseLost:
                self.lost = True
                return
            except Exception as exc:  # transient coordinator trouble: keep trying
                log.warning(_jlog("attempt heartbeat error", error=str(exc)))

    def stop(self) -> None:
        self._stop.set()


class _CheckpointRelay(threading.Thread):
    """Ships the task's checkpoint files while it runs: each new
    `ckpt/step-*.json` is uploaded, registered as a part, and committed as a
    single-part manifest — so a crash a moment later still leaves a valid,
    resumable checkpoint on the coordinator. Best-effort by design: a failed
    ship just means an older resume point; the run itself is never blocked."""

    def __init__(self, client: CoordinatorClient, lease: Lease, ckpt_dir: Path, prefix: str):
        super().__init__(daemon=True)
        self._client = client
        self._lease = lease
        self._ckpt_dir = ckpt_dir
        self._prefix = prefix
        self._halt = threading.Event()
        self._shipped: set[str] = set()

    def run(self) -> None:
        while not self._halt.wait(0.3):
            self._ship_new()

    def finish(self) -> None:
        """Stop scanning and do one final sweep — a dying attempt's last
        checkpoint must be shipped even if the process died mid-interval."""
        self._halt.set()
        self.join(timeout=10)
        self._ship_new()

    def _ship_new(self) -> None:
        if not self._ckpt_dir.is_dir():
            return
        for path in sorted(self._ckpt_dir.glob("step-*.json")):
            if path.name in self._shipped:
                continue
            try:
                step = int(path.stem.split("-")[1])
                key = f"{self._prefix}ckpt/{path.name}"
                sha = self._client.upload_artifact(path, key)
                part = {"key": key, "sha256": sha, "size_bytes": path.stat().st_size}
                self._client.checkpoint_register_part(
                    self._lease.job_id, self._lease.task_id, self._lease.lease_id, step, part
                )
                self._client.checkpoint_commit(
                    self._lease.job_id, self._lease.task_id, self._lease.lease_id,
                    step, [part], f"artifact://{self._prefix}ckpt/{step}/",
                )
                self._shipped.add(path.name)
            except Exception as exc:  # best effort — older resume point, not a dead run
                log.warning(_jlog("checkpoint ship failed", file=path.name, error=str(exc)))


class ExecutorLoop:
    def __init__(
        self,
        client: CoordinatorClient,
        node_id: str,
        runner: SubprocessRunner | None = None,
        poll_seconds: float = 1.0,
        node_heartbeat_seconds: float = 5.0,
        workdir_base: Path | None = None,
        registration=None,  # NodeRegistration; enables re-register after coordinator restart
        max_unpacked_bytes: int = DEFAULT_MAX_BYTES,
        max_unpacked_members: int = DEFAULT_MAX_MEMBERS,
        health_check=None,
        max_consecutive_failures: int = 0,
        sampler_factory=None,
    ):
        self.client = client
        self.node_id = node_id
        self.runner = runner or SubprocessRunner()
        self.poll_seconds = poll_seconds
        self.node_heartbeat_seconds = node_heartbeat_seconds
        # Where per-task tempdirs are created. Matters for the docker tier on
        # macOS: colima/Docker Desktop only share $HOME, so the default
        # system tmp (/var/folders/…) bind-mounts as an empty dir in the VM.
        self.workdir_base = Path(workdir_base) if workdir_base else None
        self.registration = registration
        # The host owner's ceiling on what one task's inputs may cost them
        # in disk and inodes — a submitter cannot raise it from the payload.
        self.max_unpacked_bytes = max_unpacked_bytes
        self.max_unpacked_members = max_unpacked_members
        self.stop_event = threading.Event()
        self.tasks_accepted = 0
        self._last_node_hb = 0.0
        # Host-facing state. Plain attributes on purpose: flashnode/status.py
        # reads them from a separate thread on a timer, and putting a lock in
        # the claim path to protect a counter would be trading a correctness
        # risk for a cosmetic one.
        self.tasks_failed = 0
        self.consecutive_failures = 0
        self.current_task: str | None = None
        self.current_attempt: int | None = None
        self.current_task_started: float | None = None
        self.quarantined = False
        self.health_report: list | None = None
        # INJECTED, never imported. `doctor.py` imports
        # `flashnode.executor.hardening`, which initialises this package,
        # whose __init__ imports this module — loop -> doctor -> executor ->
        # loop is a cycle that resolves or explodes by import order, which is
        # the worst kind of bug to ship to machines we cannot reach.
        #
        # CONTRACT: health_check() returns the BLOCKING problems; an empty
        # list means healthy. This loop never inspects a `.status`, because
        # the doctor's GPU check reports "info" and never fails — a loop
        # testing `!= "ok"` itself would quarantine every CPU-only volunteer
        # on their third unlucky job.
        self.health_check = health_check
        self.max_consecutive_failures = max_consecutive_failures
        # Builds the per-task utilisation sampler. A factory, not an
        # instance: a Thread runs once, so each task needs its own — and a
        # factory is the seam that lets the suite drive the probes instead of
        # the host's real psutil and nvidia-smi.
        self.sampler_factory = sampler_factory or ResourceSampler

    # -- inputs --------------------------------------------------------------

    def _staged_directory(self, name: str, key: str, workdir: Path) -> Path:
        """Download an archive input and hand back the *directory* it
        unpacks to, at ``workdir/inputs/<name>/``.

        That path is the contract the coordinator compiles argv against
        (``python /work/inputs/code/train.py``), so the unpacked tree has to
        land there exactly — which is why the archive is unpacked in a
        staging area first and then moved into place in one rename, rather
        than extracted over the final path. Two reasons: a refused archive
        never appears at the path the task will look at, even momentarily,
        and GitHub's wrapper directory (``owner-repo-<sha>/``) is stripped
        by moving the extractor's content root, not by shuffling files.

        The downloaded archive itself is deleted afterwards and never sits
        under ``inputs/``: the task should see its code, not a second copy
        of its own bytes it could be confused by (or fill the disk with).
        """
        if not _SAFE_INPUT_NAME.match(name) or name in (".", ".."):
            raise TaskExecutionError(
                f"refusing to unpack input with unsafe name {name!r}"
            )
        stage = workdir / ".staging" / name
        dest = workdir / "inputs" / name
        if dest.exists():
            raise TaskExecutionError(f"input {name!r} collides with an existing path")
        try:
            stage.mkdir(parents=True, exist_ok=True)
            # A fixed local filename: the artifact key is submitter-chosen
            # and has no business naming a file on this machine. Nothing
            # downstream reads the name anyway — the extractor detects the
            # container format from the bytes.
            archive = self.client.download_artifact(key, stage / "archive.bin")
            root = extract_archive_safely(
                Path(archive), stage / "unpacked",
                self.max_unpacked_bytes, self.max_unpacked_members,
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(root, dest)
        except ArchiveError as exc:
            # A hostile archive fails this task; it never kills the agent.
            log.warning(_jlog("refused unsafe input archive", input=name, error=str(exc)))
            raise TaskExecutionError(f"input {name!r}: {exc}") from None
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        log.info(_jlog("unpacked input", input=name, path=str(dest)))
        return dest

    # -- one task ------------------------------------------------------------

    def execute_one(self, lease: Lease) -> bool:
        """Run one lease, and record what its outcome says about this HOST.

        Three different things return False here and only one implicates the
        machine:

            TaskExecutionError  — could not run it here            -> counts
            LeaseLost           — someone else has the work        -> does not
            accepted=False      — coordinator declined the result  -> does not

        That last one is HTTP 200. Counting it would punish a healthy host
        for losing a commit race — the same trap the contributions ledger hit
        by crediting on 2xx.
        """
        self.current_task = lease.task_id
        self.current_attempt = lease.attempt_number
        self.current_task_started = time.monotonic()
        try:
            accepted = self._execute_inner(lease)
        except TaskExecutionError:
            # _execute_inner already reported fail() and logged the cause.
            self.tasks_failed += 1
            self.consecutive_failures += 1
            return False
        except LeaseLost:
            # _execute_inner normally swallows this and returns False; the
            # belt-and-braces handler makes the contract above true wherever
            # it is raised from, and costs a healthy host nothing.
            return False
        else:
            if accepted:
                self.consecutive_failures = 0
            return accepted
        finally:
            self.current_task = None
            self.current_attempt = None
            self.current_task_started = None

    def _execute_inner(self, lease: Lease) -> bool:
        """Run a claimed lease end-to-end. Returns True if the commit was
        accepted. Never raises for task-level problems — they are reported."""
        payload = lease.payload
        hb = _AttemptHeartbeat(self.client, lease)
        hb.start()
        relay: _CheckpointRelay | None = None
        try:
            if self.workdir_base:
                self.workdir_base.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f"flashnode-{lease.task_id}-",
                dir=str(self.workdir_base) if self.workdir_base else None,
            ) as tmp:
                workdir = Path(tmp)
                # Explicit, never inferred: an input is unpacked only if the
                # payload names it in `unpack_inputs`. Sniffing the file
                # extension would hand the decision to whoever chose the
                # artifact key — i.e. to the submitter — and "it ended in
                # .tar.gz so we ran an extractor over it" is exactly the
                # kind of implicit behaviour that turns a hostile upload
                # into a host-owner problem.
                unpack = payload.get("unpack_inputs") or []
                if not isinstance(unpack, list) or not all(isinstance(n, str) for n in unpack):
                    raise TaskExecutionError(
                        "payload 'unpack_inputs' must be a list of input names"
                    )
                unpack_set = set(unpack)

                inputs: dict[str, Path] = {}
                for name, uri in (payload.get("inputs") or {}).items():
                    key = str(uri).removeprefix("artifact://")
                    if name in unpack_set:
                        inputs[name] = self._staged_directory(name, key, workdir)
                    else:
                        inputs[name] = self.client.download_artifact(
                            key, workdir / "inputs" / Path(key).name
                        )
                unknown = unpack_set - set(payload.get("inputs") or {})
                if unknown:
                    # A named-but-absent input means the payload and the job
                    # disagree; failing loudly beats silently running a
                    # command whose code was never staged.
                    raise TaskExecutionError(
                        f"unpack_inputs names inputs that do not exist: {sorted(unknown)}"
                    )

                prefix = payload.get("output_prefix", f"jobs/{lease.job_id}/{lease.task_id}/")
                if payload.get("checkpoint") is not None:
                    # resume from the task's latest valid checkpoint, wherever
                    # the previous attempt ran
                    manifest = self.client.checkpoint_latest(lease.job_id, lease.task_id)
                    if manifest and manifest.get("parts"):
                        inputs["resume"] = self.client.download_artifact(
                            manifest["parts"][0]["key"], workdir / "inputs" / "resume.json"
                        )
                        log.info(_jlog("resuming from checkpoint",
                                       task=lease.task_id, step=manifest.get("step")))
                    relay = _CheckpointRelay(self.client, lease, workdir / "out" / "ckpt", prefix)
                    relay.start()

                # Execution evidence is measured around the RUNNER and
                # nothing else. Folding the artifact upload into the wall
                # clock would inflate every reading by however slow the
                # coordinator was that minute — and inflation is the
                # direction a liar wants, since the coordinator's own
                # claim-to-commit elapsed is the number this is cross-checked
                # against.
                sampler = self.sampler_factory()
                sampler.start()
                started = time.monotonic()
                try:
                    outdir = self.runner.run(payload, workdir, inputs)
                finally:
                    wall_seconds = time.monotonic() - started
                    cpu_mean, gpu_mean = sampler.stop()
                    if relay is not None:
                        relay.finish()  # ship the dying attempt's last checkpoint too

                if hb.lost:
                    log.warning(_jlog("lease lost during run — discarding result",
                                      task=lease.task_id))
                    return False

                prefix = payload.get("output_prefix", f"jobs/{lease.job_id}/{lease.task_id}/")
                metrics_sha = ""
                # rglob, not iterdir: a job that writes nested output (e.g.
                # out/checkpoints/model.pt) must not have it silently
                # dropped just because ArgvDockerRunner's size cap already
                # walks the tree recursively (argv_runner.py's rglob) while
                # this used to upload only the top level.
                for path in sorted(outdir.rglob("*")):
                    if path.is_file():
                        rel = path.relative_to(outdir)
                        sha = self.client.upload_artifact(path, f"{prefix}{rel.as_posix()}")
                        # metrics.json is the commit key: only the file AT
                        # the output root counts, never a same-named file
                        # nested in a subdirectory.
                        if rel == Path("metrics.json"):
                            metrics_sha = sha
                # Read off the runner rather than inferred: only the runner
                # knows whether a container ran at all. `getattr` with an
                # absent default because a runner is an interface anyone may
                # implement — one that measures nothing reports absence, and
                # absence is a fine answer. Guessing would not be.
                evidence = ExecutionEvidence(
                    wall_seconds=wall_seconds,
                    cpu_percent_mean=cpu_mean,
                    gpu_util_percent_mean=gpu_mean,
                    image_digest=getattr(self.runner, "last_image_digest", "") or "",
                    exit_code=getattr(self.runner, "last_exit_code", None),
                )
                accepted = self.client.complete(
                    lease.lease_id, metrics_sha or "0" * 64, evidence=evidence
                )
                if accepted:
                    self.tasks_accepted += 1
                log.info(_jlog("task finished", task=lease.task_id,
                               attempt=lease.attempt_number, accepted=accepted))
                return accepted
        except TaskExecutionError as exc:
            log.warning(_jlog("task failed", task=lease.task_id, error=str(exc)))
            try:
                self.client.fail(lease.lease_id, str(exc)[:500])
            except Exception:
                pass  # lease will expire on its own — same outcome, slower
            # Re-raised, not returned: execute_one counts THIS outcome
            # against the host and the other two against nobody. Caught one
            # frame up, so callers still see False.
            raise
        except LeaseLost:
            log.warning(_jlog("lease lost", task=lease.task_id))
            return False
        finally:
            hb.stop()

    # -- the loop ------------------------------------------------------------

    def _maybe_node_heartbeat(self) -> None:
        if time.monotonic() - self._last_node_hb >= self.node_heartbeat_seconds:
            ok = self.client.node_heartbeat(self.node_id)
            if not ok and self.registration is not None:
                # A refused heartbeat usually means the coordinator restarted
                # and forgot us — re-register instead of starving forever.
                log.info(_jlog("heartbeat refused — re-registering", node=self.node_id))
                self.client.register(self.registration)
            self._last_node_hb = time.monotonic()

    def _should_stop_volunteering(self) -> bool:
        """After a streak of host-side failures, ask whether it is the HOST.

        A counter alone would guess. This measures: re-run the same checks
        `flashnode doctor` runs, and let the answer decide.

        - nothing blocking -> this machine is fine and the JOBS are failing.
          Say so, reset, keep working. A host that stops because of someone
          else's broken job is a host that stops for no reason.
        - blocking problems -> stop claiming. Continuing means burning this
          job's retries on a machine that cannot run anything.
        """
        if self.health_check is None or self.max_consecutive_failures <= 0:
            return False
        if self.consecutive_failures < self.max_consecutive_failures:
            return False
        unhealthy = self.health_check()
        if not unhealthy:
            log.info(_jlog(
                "consecutive task failures, but this host passes its own "
                "checks — the jobs are failing, not the machine",
                failures=self.consecutive_failures))
            self.consecutive_failures = 0
            return False
        self.quarantined = True
        self.health_report = unhealthy
        log.error(_jlog("stopping: this host can no longer run tasks",
                        failures=self.consecutive_failures,
                        failed_checks=[getattr(r, "name", "?") for r in unhealthy]))
        return True

    def run(self, max_tasks: int | None = None, idle_exit: bool = False) -> int:
        """Claim-and-execute until stopped, `max_tasks` accepted, or — with
        `idle_exit` — the queue drains (drain mode for tests/one-shot runs).
        Returns the number of accepted tasks."""
        backoff = 1.0
        while not self.stop_event.is_set():
            if max_tasks is not None and self.tasks_accepted >= max_tasks:
                break
            try:
                self._maybe_node_heartbeat()
                lease = self.client.claim(self.node_id)
                backoff = 1.0
            except Exception as exc:
                log.warning(_jlog("coordinator unreachable", error=str(exc), backoff_s=backoff))
                if self.stop_event.wait(backoff):
                    break
                backoff = min(backoff * 2, 30)
                continue
            if lease is None:
                if idle_exit:
                    break
                if self.stop_event.wait(self.poll_seconds):
                    break
                continue
            log.info(_jlog("claimed", task=lease.task_id, attempt=lease.attempt_number))
            self.execute_one(lease)
            if self._should_stop_volunteering():
                break
        return self.tasks_accepted


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
