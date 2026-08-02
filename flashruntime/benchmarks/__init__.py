"""benchmarks — FlashRuntime's honest evaluation suite.

Every number in here is MEASURED on the machine that runs it, never asserted.
Each scenario file carries its hypothesis at the top and its measurement method
in the module docstring, so the methodology is auditable from the file alone.
Where a comparator (ray, accelerate) can't run on this host, the row says so in
`notes` — a missing comparator is a documented skip, never a fabricated figure.

Run it:

    python -m benchmarks run --all --repeats 5      # the real baseline
    python -m benchmarks run --scenario hpo_sweep   # one scenario
    python -m benchmarks run --all --smoke          # 1 repeat, labelled

Add a scenario = one file in `scenarios/` + one line in `registry.SCENARIOS`.
"""

SCHEMA = "bench_v1"
