"""Demo 1/3 — sklearn hyperparameter sweep: FlashRuntime operating code that
does not import it.

    python examples/demo_sklearn_sweep.py
    python examples/demo_sklearn_sweep.py --watch          # + live run page

The script being operated (`user_sklearn/train.py`) has NO flashruntime
import anywhere. The whole contract is convention: CLI flags in, a flat
`metrics.json` out. The adapter builds one independent task per point in the
grid, so the sweep is Mode A shaped (retriable, independently placeable) even
though it runs locally here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # sibling _preflight, not a package

import flashruntime as flash  # noqa: E402
from _preflight import hold_viewer, require_sklearn  # noqa: E402
from flashruntime.integrations import sklearn as fr_sklearn  # noqa: E402

EXAMPLES = Path(__file__).parent

# One task per combination: 2 models × 3 C values = 6 trials. Every dict the
# adapter builds carries every key, because the CLI flags come from the union.
GRID = {"model": ["logreg", "rf"], "C": [0.1, 1.0, 10.0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watch", action="store_true", help="open the live run page")
    args = parser.parse_args()

    require_sklearn()

    run = flash.submit(
        fr_sklearn.hpo("train.py", GRID, source=str(EXAMPLES / "user_sklearn")),
        watch=True if args.watch else False,
    )

    print(f"\nstate: {run.state.value}   trials: {len(run.trials)}")
    for trial in run.trials:
        print(f"  model={trial.get('model'):<7} C={trial.get('C'):<5} "
              f"accuracy_mean={trial.get('accuracy_mean')}")
    print(f"\nbest: {run.best_trial()}")

    expected = len(GRID["model"]) * len(GRID["C"])
    ok = run.state.value == "SUCCEEDED" and len(run.trials) == expected
    print("\nRESULT:", "PASS" if ok else f"FAIL (expected {expected} succeeded trials)")

    if args.watch:
        hold_viewer(run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
