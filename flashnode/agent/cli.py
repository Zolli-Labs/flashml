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
  join      connect this machine to a FlashML control plane (not yet implemented)
  status    show node identity, capabilities, and active leases (not yet implemented)
  leave     drain and disconnect (not yet implemented)
"""


def _work(args: list[str]) -> int:
    import argparse
    import logging
    import os
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
        help="task execution tier (docker/argv need FLASHNODE_ALLOWED_IMAGES)",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    opts = parser.parse_args(args)

    runner = None
    if opts.runner in ("docker", "argv"):
        # Shared by both sandboxed tiers: neither may start against an empty
        # allowlist, so the check is hoisted here rather than duplicated per
        # branch.
        images = frozenset(
            i.strip() for i in os.environ.get("FLASHNODE_ALLOWED_IMAGES", "").split(",") if i.strip()
        )
        if not images:
            print(
                f"flashnode work: --runner {opts.runner} requires FLASHNODE_ALLOWED_IMAGES "
                "(comma-separated image references) — refusing to start with an "
                "empty allowlist",
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

    node_id = load_or_create_node_id()
    client = CoordinatorClient(
        opts.coordinator, join_code=os.environ.get("FLASHNODE_JOIN_CODE") or None
    )
    registration = discover(
        node_id, kubernetes_node="", node_meta=None,
        argv_capable=(opts.runner == "argv"),
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
    print(USAGE.format(version=__version__), end="")
    if args and args[0] in {"join", "status", "leave"}:
        print(f"\nerror: '{args[0]}' is not implemented yet in this scaffold.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
