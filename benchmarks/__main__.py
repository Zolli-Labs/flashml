"""CLI: run scenarios, write a ``bench_v1`` result file, print the table.

    python -m benchmarks run --all --repeats 5          # the real baseline
    python -m benchmarks run --scenario hpo_sweep        # one scenario
    python -m benchmarks run --all --smoke               # 1 repeat, labelled

A ``bench_v1`` document is ``{schema, host, rows}`` where ``host`` records the
machine (os/cpu/cores/ram/python/torch/flashruntime) so a result file is
self-describing — a number is only meaningful next to the box that produced it.
Scenarios whose hard dependency is missing (e.g. no torch) are skipped with a
message and emit no row; a missing *comparator* is a note ON a row instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks import SCHEMA, registry, report
from benchmarks._util import ScenarioUnavailable, ensure_venv_on_path, host_info

RESULTS = Path(__file__).resolve().parent / "results"


def _run(names: list[str], repeats: int) -> list:
    ensure_venv_on_path()
    rows = []
    for name in names:
        try:
            row = registry.SCENARIOS[name].run(repeats)
        except ScenarioUnavailable as exc:
            print(f"skip {name}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        print(f"ran {name}: median={row.median} {row.unit} ({row.repeats} repeats)", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks", description="FlashRuntime honest evaluation suite")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run scenarios and write a bench_v1 result file")
    target = run_p.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="run every scenario")
    target.add_argument("--scenario", choices=sorted(registry.SCENARIOS), help="run one scenario")
    run_p.add_argument("--repeats", type=int, default=5, help="samples per scenario (default 5)")
    run_p.add_argument("--smoke", action="store_true",
                       help="1 repeat; the printed table is labelled non-representative")
    run_p.add_argument("--out", type=Path, default=None,
                       help="output path (default: results/<timestamp>.json)")
    args = parser.parse_args(argv)

    names = sorted(registry.SCENARIOS) if args.all else [args.scenario]
    repeats = 1 if args.smoke else args.repeats
    rows = _run(names, repeats)

    doc = {"schema": SCHEMA, "host": host_info(), "rows": [r.model_dump() for r in rows]}
    out = args.out or (RESULTS / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"\nwrote {out}\n")
    print(report.render_markdown(rows, smoke=args.smoke) if rows else "(no scenarios ran)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
