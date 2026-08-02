"""Demo 2/3 — PyTorch DDP under FlashRuntime: N processes, gloo on CPU.

    python examples/demo_pytorch_ddp.py
    python examples/demo_pytorch_ddp.py --nproc 2 --steps 200 --watch
    python examples/demo_pytorch_ddp.py --vanilla     # a script with NO ft import

Two scripts, one launcher. `user_pytorch/train.py` uses the three
`flashruntime.torch` verbs (prepare / checkpoint / log_metrics), so its loss
stream and verified checkpoints show up in the run. `--vanilla` points the
same launcher at `user_pytorch_vanilla/train.py`, which wires
`init_process_group` and `DistributedDataParallel` by hand and imports
nothing from flashruntime — proving the launcher operates unmodified code,
and showing exactly what you give up without the helper (no metric stream,
no manifests, so nothing to resume from).

Requires torchrun on PATH — see `_preflight.ACTIVATE_HINT`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # sibling _preflight, not a package

import flashruntime as flash  # noqa: E402
from _preflight import hold_viewer, require_torchrun  # noqa: E402
from flashruntime.integrations import pytorch as fr_torch  # noqa: E402

EXAMPLES = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nproc", type=int, default=2, help="processes (ranks) to launch")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--vanilla", action="store_true",
                        help="operate the no-flashruntime-import script instead")
    parser.add_argument("--watch", action="store_true", help="open the live run page")
    args = parser.parse_args()

    require_torchrun()

    # The vanilla script takes no flags — it is deliberately untouched code.
    source = EXAMPLES / ("user_pytorch_vanilla" if args.vanilla else "user_pytorch")
    script_args = (
        "" if args.vanilla
        else f"--steps {args.steps} --checkpoint-every {args.checkpoint_every}"
    )

    print(f"operating {source.name}/train.py on {args.nproc} ranks (gloo/CPU)")
    run = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(source),
            nproc_per_node=args.nproc,   # nnodes>1 raises NotImplementedError by design
            script_args=script_args,
        ),
        watch=True if args.watch else False,
    )

    print(f"\nstate: {run.state.value}")
    for trial in run.trials:
        print(f"  {trial}")
    if not args.vanilla and run.trials:
        t = run.trials[0]
        print(f"\ndevice={t.get('device')}  backend={t.get('backend')}  steps={t.get('steps')}")
    if args.vanilla:
        print("\n(no loss stream / no checkpoint manifests — that is what the "
              "flashruntime.torch verbs buy you)")

    ok = run.state.value == "SUCCEEDED"
    print("\nRESULT:", "PASS" if ok else "FAIL")

    if args.watch:
        hold_viewer(run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
