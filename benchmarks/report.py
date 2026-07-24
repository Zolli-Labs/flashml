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


# Canonical section order for the grouped docs render: the wall-clock
# "Performance" tables first, then the fault-tolerance "Resilience" tables. Any
# section not named here falls after these in first-appearance order (so a
# future section never silently disappears).
SECTION_ORDER = ("performance", "resilience")


def _table_body(rows: list[dict]) -> str:
    """The header + one line per row — no smoke caption (callers add it once)."""
    lines = ["| scenario | median | unit | p10 | p90 | repeats |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {_fmt(r['median'])} | {r['unit']} "
            f"| {_fmt(r['p10'])} | {_fmt(r['p90'])} | {r['repeats']} |"
        )
    return "\n".join(lines)


def _group_by_section(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group rows by ``section`` (default "performance"), row order preserved
    within each. Sections come out in ``SECTION_ORDER`` first, then any unknown
    section by first appearance — so a doc with only one section yields exactly
    one group (never an empty table for the absent section)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("section", "performance"), []).append(r)
    ordered = [s for s in SECTION_ORDER if s in groups]
    ordered += [s for s in groups if s not in SECTION_ORDER]
    return [(s, groups[s]) for s in ordered]


def render_markdown(rows: Iterable, smoke: bool = False) -> str:
    """Compact summary table (one line per scenario). Refuses thin rows unless
    ``smoke``; a smoke table is captioned as non-representative."""
    rows = [_as_dict(r) for r in rows]
    _check_repeats(rows, smoke)
    lines = []
    if smoke:
        lines.append(SMOKE_LABEL)
        lines.append("")
    lines.append(_table_body(rows))
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
    """Full Markdown body for the docs page: the host block, the grouped summary
    tables (one per ``section`` — "Performance", "Resilience"), per-scenario
    detail (hypothesis + comparators + notes), and the repro command. ``bench``
    is a ``bench_v1`` document (``schema``/``host``/``rows``). Refuses thin rows
    (``repeats`` < 3) unless ``smoke``, exactly as :func:`render_markdown`."""
    try:  # hypotheses come from the scenario modules; optional so a bare JSON still renders
        from benchmarks.registry import SCENARIOS

        hypotheses = {name: s.hypothesis for name, s in SCENARIOS.items()}
    except Exception:  # noqa: BLE001
        hypotheses = {}

    rows = [_as_dict(r) for r in bench.get("rows", [])]
    _check_repeats(rows, smoke)  # refuse thin rows once, across every section
    grouped = _group_by_section(rows)
    parts = []
    if smoke:
        parts += [SMOKE_LABEL, ""]
    parts += ["**Measured on:**", "", _host_table(bench.get("host", {})), ""]
    parts += ["Reproduce every number below with:", "", "```bash", REPRO, "```", ""]
    # One headed summary table per present section — a resilience COUNT never
    # shares a table with a wall-clock median. Sections with no rows are omitted.
    parts += ["## Summary", ""]
    for section, section_rows in grouped:
        parts += [f"### {section.title()}", "", _table_body(section_rows), ""]
    # Per-scenario detail follows the same grouping so the page reads section by
    # section top to bottom.
    for _, section_rows in grouped:
        for row in section_rows:
            parts.append(_detail(row, hypotheses))
    return "\n".join(parts).rstrip() + "\n"
