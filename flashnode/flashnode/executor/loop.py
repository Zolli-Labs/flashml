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

from flashruntime.protocol.v1alpha1 import Lease

from flashnode.executor.archives import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_MEMBERS,
    ArchiveError,
    extract_archive_safely,
)
from flashnode.executor.client import CoordinatorClient, LeaseLost
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

                try:
                    outdir = self.runner.run(payload, workdir, inputs)
                finally:
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
                accepted = self.client.complete(lease.lease_id, metrics_sha or "0" * 64)
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
            return False
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
        return self.tasks_accepted


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
