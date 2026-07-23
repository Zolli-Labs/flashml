"""Thin CLI over the FlashRuntime API — plus the offline planner and the
local "bring your own code" front door.

  flashruntime plan path/to/plan.yaml [--json]   # no API/cluster needed
  flashruntime submit "python train.py" [--source DIR] [--task-params JSON] \
      [--max-restarts N] [--output-dir DIR] [--watch|--no-watch]  # local, no API
  flashruntime submit-spec path/to/job.yaml [--api URL]  # POST a JobSpec to the coordinator
  flashruntime status <job-id>
  flashruntime events <job-id>
  flashruntime logs <job-id>
  flashruntime cancel <job-id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _api(args) -> str:
    return args.api or os.environ.get("FLASHML_RUNTIME_API", "http://localhost:8100")


def _plan(args) -> int:
    """Run the offline strategy planner: file → PlanRequest → PlanReport."""
    from flashruntime.planner import plan, render
    from flashruntime.protocol.plan_v1alpha1 import PlanRequest

    if args.request_file.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            print("pyyaml is required for YAML input: pip install pyyaml", file=sys.stderr)
            return 2
        with open(args.request_file) as f:
            raw = yaml.safe_load(f)
    else:
        with open(args.request_file) as f:
            raw = json.load(f)

    try:
        request = PlanRequest.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError — show it plainly
        print(f"invalid PlanRequest: {exc}", file=sys.stderr)
        return 1

    report = plan(request)
    if args.json:
        print(report.model_dump_json(indent=2, exclude_none=True))
    else:
        print(render(report))
    return 0 if report.selected is not None else 3


def _submit(args) -> int:
    """Build a CommandWorkload and run it locally via the SDK, then print a
    human summary. Exit 0 iff SUCCEEDED. The SDK is imported here (not at
    module top) so cli.py stays cheap on coordinator-only installs where the
    other subcommands only need httpx."""
    from flashruntime.sdk import submit
    from flashruntime.workloads.command import CommandWorkload, Source

    try:
        task_params = json.loads(args.task_params) if args.task_params else None
    except json.JSONDecodeError as exc:  # a bad --task-params must not traceback
        print(f"--task-params is not valid JSON: {exc}", file=sys.stderr)
        return 2

    workload = CommandWorkload(
        command=args.cmd,
        source=Source(path=args.source),
        task_params=task_params,
    )

    # --watch defaults on only at a real terminal: in CI/pipes we must never
    # block on or open a viewer. Until Task 7 ships the viewer this flag just
    # prints one honest line and proceeds — remove this placeholder in T7.
    watch = args.watch if args.watch is not None else sys.stdout.isatty()
    if watch:
        print("viewer lands with the next feature; running without it")

    run = submit(workload, output_dir=args.output_dir, max_restarts=args.max_restarts)

    print(f"state:     {run.state.value}")
    print(f"trials:    {len(run.trials)}")
    if workload.outputs.primary_metric:
        best = run.best_trial()
        print(f"best:      {best}" if best else "best:      (no trial reported the metric)")
    print(f"output:    {run.output_dir}")
    return 0 if run.state.value == "SUCCEEDED" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flashruntime")
    parser.add_argument("--api", help="FlashRuntime API base URL")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="evaluate a PlanRequest offline and print the strategy")
    p_plan.add_argument("request_file", help="PlanRequest as .yaml or .json")
    p_plan.add_argument("--json", action="store_true", help="emit the full PlanReport as JSON")

    p_submit = sub.add_parser("submit", help="run a command workload locally (no API needed)")
    # dest is `cmd`, not `command`: the subparsers dest is already `command`
    # (the subcommand name), and a positional named `command` would clobber it.
    p_submit.add_argument("cmd", metavar="CMD", help="the command to run, e.g. 'python train.py --lr {lr}'")
    p_submit.add_argument("--source", default=".", help="directory holding the user's code")
    p_submit.add_argument("--task-params", help="JSON list of param dicts for Mode A fan-out")
    p_submit.add_argument("--max-restarts", type=int, default=0, help="automatic recovery budget")
    p_submit.add_argument("--output-dir", help="where run.json and artifacts land (default: temp dir)")
    p_submit.add_argument(
        "--watch",
        action=argparse.BooleanOptionalAction,
        default=None,  # None ⇒ decide by TTY in _submit (never block/open a viewer in CI)
        help="open the live viewer (default: on at a terminal, off in pipes/CI)",
    )

    p_submit_spec = sub.add_parser(
        "submit-spec",
        help=(
            "POST a JobSpec YAML to the coordinator — was `submit` before 0.1.0; "
            "renamed when `submit` became the local-workload front door"
        ),
    )
    p_submit_spec.add_argument("spec_file")
    for name in ("status", "events", "logs", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("job_id")

    args = parser.parse_args(argv)

    if args.command == "plan":
        return _plan(args)
    if args.command == "submit":
        return _submit(args)

    import httpx

    base = _api(args)
    try:
        if args.command == "submit-spec":
            import yaml

            with open(args.spec_file) as f:
                spec = yaml.safe_load(f)
            r = httpx.post(f"{base}/v1alpha1/jobs", json=spec, timeout=60)
        elif args.command == "cancel":
            r = httpx.post(f"{base}/v1alpha1/jobs/{args.job_id}/cancel", timeout=60)
        elif args.command == "status":
            r = httpx.get(f"{base}/v1alpha1/jobs/{args.job_id}", timeout=30)
        else:
            r = httpx.get(f"{base}/v1alpha1/jobs/{args.job_id}/{args.command}", timeout=30)
    except httpx.ConnectError as exc:
        print(f"cannot reach FlashRuntime API at {base}: {exc}", file=sys.stderr)
        return 2

    if r.status_code >= 400:
        print(f"error {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    print(json.dumps(r.json(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
