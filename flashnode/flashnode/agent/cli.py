"""Command-line entry point for the FlashNode agent.

Target surface (see docs/SYSTEM_OVERVIEW.md §10):

    flashnode join --code <one-time-code>
    flashnode status
    flashnode leave
"""

from __future__ import annotations

import sys

from flashnode import __version__

USAGE = """\
flashnode {version} — FlashML open host agent (pre-release scaffold)

usage: flashnode <command>

commands:
  agent     run the node agent loop (register with FlashML Cloud + heartbeat)
  work      register with a FlashRuntime coordinator and execute leased tasks
            (--coordinator URL | FLASHNODE_COORDINATOR_URL; --max-tasks N)
  login     enrol this machine — prints a code to approve in a browser
            (--coordinator URL; --token TOKEN to skip the browser step)
  logout    remove the saved bearer token for a FlashRuntime coordinator
            (--coordinator URL)
  doctor    check this machine can run tasks (docker engine, images, mounts)
  join      connect this machine to a FlashML control plane (not yet implemented)
  status    show node identity, capabilities, and active leases (not yet implemented)
  leave     drain and disconnect (not yet implemented)
"""


def _login(args: list[str]) -> int:
    """Enrol this machine.

    The default path is device-code: print a short code, wait for a
    signed-in human to approve it in a browser, save the token that comes
    back. That is what the console tells volunteers to run, and until now it
    could not work — `--token` was REQUIRED, and the enrolment flow issues
    no token for anyone to paste. The API half (/v1alpha1/device/code,
    /v1alpha1/device/token) had been built and simply had no client.

    `--token` stays supported for a credential you already hold: CI, a
    self-hosted coordinator, or re-pointing a machine with no browser to
    hand.
    """
    import argparse

    from flashnode.identity.credentials import save_token

    parser = argparse.ArgumentParser(
        prog="flashnode login",
        description="Enrol this machine with FlashML.",
    )
    parser.add_argument(
        "--coordinator",
        required=True,
        help="FlashML Cloud API base URL (e.g. https://flashml-api.onrender.com)",
    )
    parser.add_argument(
        "--token",
        help="skip the browser step and save a token you already have",
    )
    opts = parser.parse_args(args)

    if opts.token:
        path = save_token(opts.coordinator, opts.token)
        print(f"flashnode login: credential saved to {path}", file=sys.stderr)
        return 0

    from flashnode.identity.enrol import (
        EnrolmentError,
        describe_this_machine,
        poll_for_token,
        request_device_code,
    )
    from flashnode.identity.store import load_or_create_node_id

    try:
        node_id = load_or_create_node_id()
    except OSError as exc:
        print(
            f"flashnode login: cannot write this machine's identity: {exc}\n"
            "Set FLASHNODE_STATE_DIR to a directory you can write to.",
            file=sys.stderr,
        )
        return 1

    hostname, platform_name = describe_this_machine()

    try:
        start = request_device_code(
            opts.coordinator, node_id, hostname, platform_name
        )
    except EnrolmentError as exc:
        print(f"flashnode login: {exc}", file=sys.stderr)
        return 1

    # stdout, not stderr: this is the output the person is here for, and it
    # should survive being piped.
    #
    # flush=True is load-bearing. Python block-buffers stdout when it is not
    # a terminal, so piping `flashnode login` anywhere — tee, a log, a setup
    # script — showed nothing at all while the process sat waiting for an
    # approval of a code it had never displayed.
    print(flush=True)
    print(f"  Your code:  {start.user_code}", flush=True)
    print(f"  Approve at: {start.verification_uri}", flush=True)
    print(flush=True)
    print(
        "Open that on any device you're signed in on — your phone is fine.",
        flush=True,
    )
    print("Waiting for approval… (Ctrl-C to cancel)", flush=True)

    try:
        token = poll_for_token(
            opts.coordinator, start.device_code, interval=start.interval
        )
    except EnrolmentError as exc:
        print(f"\nflashnode login: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Cancelling is a normal act, not a crash. The code expires unused.
        print("\nflashnode login: cancelled.", file=sys.stderr)
        return 130

    path = save_token(opts.coordinator, token)
    print(f"\nApproved. This machine is enrolled — credential saved to {path}.")
    print("Start contributing with:  flashnode work --runner docker")
    return 0


def _logout(args: list[str]) -> int:
    import argparse

    from flashnode.identity.credentials import clear_token, credentials_path

    parser = argparse.ArgumentParser(prog="flashnode logout")
    parser.add_argument("--coordinator", required=True, help="FlashRuntime coordinator base URL")
    opts = parser.parse_args(args)

    removed = clear_token(opts.coordinator)
    if removed:
        print(f"flashnode logout: credential removed from {credentials_path()}", file=sys.stderr)
    else:
        print(f"flashnode logout: no saved credential for {opts.coordinator}", file=sys.stderr)
    return 0


def _work(args: list[str]) -> int:
    import argparse
    import logging
    import os
    import shutil
    import signal

    from flashnode.executor import CoordinatorClient, ExecutorLoop
    from flashnode.identity.store import load_or_create_node_id
    from flashnode.inventory.capabilities import discover

    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"flashnode","msg":%(message)s}',
    )
    parser = argparse.ArgumentParser(prog="flashnode work")
    parser.add_argument(
        "--coordinator",
        default=os.environ.get("FLASHNODE_COORDINATOR_URL", "http://localhost:8100"),
        help="FlashRuntime coordinator base URL",
    )
    parser.add_argument(
        "--runner",
        choices=["subprocess", "docker", "argv"],
        default=os.environ.get("FLASHNODE_RUNNER", "subprocess"),
        help="task execution tier (docker/argv need the docker CLI on PATH)",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    opts = parser.parse_args(args)

    runner = None
    if opts.runner in ("docker", "argv"):
        # OPTIONAL, and additive. The built-in namespace allowlist
        # (executor/images.py DEFAULT_ALLOWED_IMAGE_PREFIXES) is what a
        # volunteer runs on, so this env var is no longer required and an
        # empty value is no longer a refusal to start.
        #
        # It used to be mandatory, which quietly capped the project at the
        # number of machines whose owners would hand-maintain a list of image
        # references: every image we published stranded every host until its
        # owner edited the variable, so security fixes would reach a fraction
        # of the fleet. Setting it now means "also allow these", for
        # self-hosting the stack or for integration tests.
        images = frozenset(
            i.strip() for i in os.environ.get("FLASHNODE_ALLOWED_IMAGES", "").split(",") if i.strip()
        )
        # Both sandboxed tiers shell out to the `docker` binary directly
        # (subprocess.run(["docker", ...])); if it isn't installed that call
        # raises FileNotFoundError deep inside a task attempt. Check for it
        # here, at startup, rather than let the agent die on the first task.
        if shutil.which("docker") is None:
            print(
                f"flashnode work: --runner {opts.runner} requires the `docker` CLI "
                "on PATH — refusing to start without it",
                file=sys.stderr,
            )
            return 2
        if opts.runner == "docker":
            from flashnode.executor.docker_runner import DockerRunner

            runner = DockerRunner(allowed_images=images)
        else:
            from flashnode.executor.argv_runner import ArgvDockerRunner

            runner = ArgvDockerRunner(
                allowed_images=images,
                cpus=float(os.environ.get("FLASHNODE_MAX_CPUS", "2.0")),
                memory_gb=float(os.environ.get("FLASHNODE_MAX_MEMORY_GB", "2.0")),
                timeout_seconds=float(os.environ.get("FLASHNODE_TASK_TIMEOUT_S", "3600")),
                max_output_bytes=int(
                    os.environ.get("FLASHNODE_MAX_OUTPUT_BYTES", str(2 * 1024**3))
                ),
            )

    workdir_base = os.environ.get("FLASHNODE_WORKDIR") or None

    from flashnode.identity.credentials import load_token

    node_id = load_or_create_node_id()
    client = CoordinatorClient(
        opts.coordinator,
        join_code=os.environ.get("FLASHNODE_JOIN_CODE") or None,
        token=load_token(opts.coordinator),
    )
    registration = discover(
        node_id, kubernetes_node="", node_meta=None,
        argv_capable=(opts.runner == "argv"),
        # An argv-only volunteer has no module runner behind it: advertise
        # module_capable=False so the coordinator's placement gate stops
        # routing "python -m <module>" tasks here (F1) — otherwise those
        # tasks burn every attempt against ArgvDockerRunner's payload
        # rejection before the job ever fails for real.
        module_capable=(opts.runner != "argv"),
    )
    client.register(registration)
    loop = ExecutorLoop(
        client, node_id, runner=runner,
        poll_seconds=opts.poll_seconds, workdir_base=workdir_base,
        registration=registration,  # survives coordinator restarts
    )

    def _stop(signum, frame):  # noqa: ARG001
        loop.stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    accepted = loop.run(max_tasks=opts.max_tasks)
    print(f"flashnode work: {accepted} task(s) accepted", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "agent":
        from flashnode.agent.daemon import main as agent_main

        return agent_main()
    if args and args[0] == "work":
        return _work(args[1:])
    if args and args[0] == "doctor":
        from flashnode.doctor import doctor_main

        return doctor_main(args[1:])
    if args and args[0] == "login":
        return _login(args[1:])
    if args and args[0] == "logout":
        return _logout(args[1:])
    print(USAGE.format(version=__version__), end="")
    if args and args[0] in {"join", "status", "leave"}:
        print(f"\nerror: '{args[0]}' is not implemented yet in this scaffold.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
