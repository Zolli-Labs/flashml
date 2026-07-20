"""Thin CLI over the FlashRuntime API — plus the offline planner.

  flashruntime plan path/to/plan.yaml [--json]   # no API/cluster needed
  flashruntime submit path/to/job.yaml [--api URL]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flashruntime")
    parser.add_argument("--api", help="FlashRuntime API base URL")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="evaluate a PlanRequest offline and print the strategy")
    p_plan.add_argument("request_file", help="PlanRequest as .yaml or .json")
    p_plan.add_argument("--json", action="store_true", help="emit the full PlanReport as JSON")

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("spec_file")
    for name in ("status", "events", "logs", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("job_id")

    args = parser.parse_args(argv)

    if args.command == "plan":
        return _plan(args)

    import httpx

    base = _api(args)
    try:
        if args.command == "submit":
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
