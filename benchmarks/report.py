"""Render measured results as Markdown — for the terminal and for the docs.

Honesty mechanics live here, not in the scenarios:
  * a row with ``repeats < 3`` is REFUSED outside ``--smoke`` (too few samples
    to publish); ``--smoke`` renders it but stamps the table
    "smoke run — not representative";
  * the table shows exactly what the JSON holds — no rounding-away of an
    unfavorable number, and every row's ``notes`` are printed verbatim.

``scripts/build_docs.py`` calls :func:`render_document` at site-build time so
``docs/site/benchmarks.md`` shows the machine's own measured numbers with the
methodology and repro command beside them.
"""

from __future__ import annotations

from typing import Iterable

MIN_REPEATS = 3
SMOKE_LABEL = "_smoke run — not representative (1 repeat; run without `--smoke` for the real baseline)_"
REPRO = "python -m benchmarks run --all --repeats 5"


def _as_dict(row) -> dict:
    return row if isinstance(row, dict) else row.model_dump()


def _check_repeats(rows: list[dict], smoke: bool) -> None:
    if smoke:
        return
    thin = [r["scenario"] for r in rows if r["repeats"] < MIN_REPEATS]
    if thin:
        raise ValueError(
            f"refusing to render rows with repeats < {MIN_REPEATS}: {', '.join(thin)} "
            "(use smoke=True / --smoke to render a labelled non-representative table)"
        )


def _fmt(x: float) -> str:
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return f"{x:.3g}" if isinstance(x, (int, float)) else str(x)


def render_markdown(rows: Iterable, smoke: bool = False) -> str:
    """Compact summary table (one line per scenario). Refuses thin rows unless
    ``smoke``; a smoke table is captioned as non-representative."""
    rows = [_as_dict(r) for r in rows]
    _check_repeats(rows, smoke)
    lines = []
    if smoke:
        lines.append(SMOKE_LABEL)
        lines.append("")
    lines.append("| scenario | median | unit | p10 | p90 | repeats |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {_fmt(r['median'])} | {r['unit']} "
            f"| {_fmt(r['p10'])} | {_fmt(r['p90'])} | {r['repeats']} |"
        )
    return "\n".join(lines)


def _host_table(host: dict) -> str:
    keys = ["os", "cpu", "cores", "ram_gb", "python", "torch", "flashruntime"]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    values = "| " + " | ".join(str(host.get(k, "")) for k in keys) + " |"
    return "\n".join([header, sep, values])


def _detail(row: dict, hypotheses: dict[str, str]) -> str:
    lines = [f"### {row['scenario']}", ""]
    hyp = hypotheses.get(row["scenario"])
    if hyp:
        lines += [f"**Hypothesis:** {hyp}", ""]
    lines.append(f"**Measured:** {_fmt(row['median'])} {row['unit']} "
                 f"(p10 {_fmt(row['p10'])}, p90 {_fmt(row['p90'])}, {row['repeats']} repeats)")
    lines.append("")
    if row.get("comparators"):
        lines.append("| comparator | value |")
        lines.append("| --- | --- |")
        for k, v in row["comparators"].items():
            lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")
    for note in row.get("notes", []):
        lines.append(f"- _{note}_")
    if row.get("notes"):
        lines.append("")
    return "\n".join(lines)


def render_document(bench: dict, smoke: bool = False) -> str:
    """Full Markdown body for the docs page: the host block, the summary table,
    per-scenario detail (hypothesis + comparators + notes), and the repro
    command. ``bench`` is a ``bench_v1`` document (``schema``/``host``/``rows``)."""
    try:  # hypotheses come from the scenario modules; optional so a bare JSON still renders
        from benchmarks.registry import SCENARIOS

        hypotheses = {name: s.hypothesis for name, s in SCENARIOS.items()}
    except Exception:  # noqa: BLE001
        hypotheses = {}

    rows = [_as_dict(r) for r in bench.get("rows", [])]
    parts = []
    if smoke:
        parts += [SMOKE_LABEL, ""]
    parts += ["**Measured on:**", "", _host_table(bench.get("host", {})), ""]
    parts += ["Reproduce every number below with:", "", "```bash", REPRO, "```", ""]
    parts += ["## Summary", "", render_markdown(rows, smoke=smoke), ""]
    for row in rows:
        parts.append(_detail(row, hypotheses))
    return "\n".join(parts).rstrip() + "\n"
