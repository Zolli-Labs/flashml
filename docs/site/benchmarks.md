# Benchmarks

Every number on this page is **measured**, never asserted. The tables below are
rendered at docs-build time straight from the committed baseline JSON
(`benchmarks/results/baseline-<host>.json`) — the docs cannot show a figure the
suite did not produce. Each scenario states its hypothesis and its measurement
method in its own source file (`benchmarks/scenarios/`), so the methodology is
auditable from the code alone, and every caveat and skip is printed verbatim in
the notes under each table.

Where a comparator (ray, accelerate) is not installed on the baseline machine,
its row says so and its setup code is *counted*, not run, from the cited
fixtures in `benchmarks/scenarios/snippets/` — an honest line count, never a
fabricated timing. Some figures here are small, zero, or negative: that is the
suite working as intended. On a tiny CPU model, process-startup dominates
wall-clock and a checkpoint write or a 40-step recompute falls below the
run-to-run noise floor — the notes say so, and the size-independent guarantees
(e.g. `steps_not_recomputed`) are reported alongside. The value shows up at real
model scale; the honesty shows up here.

Reproduce the whole baseline yourself:

```bash
python -m benchmarks run --all --repeats 5
```

Run a single scenario, or a fast labelled smoke:

```bash
python -m benchmarks run --scenario recovery_economics
python -m benchmarks run --all --smoke
```

<!-- BENCH_TABLES -->
