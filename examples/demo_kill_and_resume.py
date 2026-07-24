"""Demo 3/3 — crash mid-training, recover in one call.

    python examples/demo_kill_and_resume.py
    python examples/demo_kill_and_resume.py --kill-at-step 400 --steps 800 --watch

ONE submit with `max_restarts=1`. The training process dies at
`--kill-at-step`; the SDK turns the failed attempt into `FailureSignals`,
runs `recovery.classify()` then `recovery.decide()`, and — because a lost
rank is a transient worker crash, not a deterministic app bug — relaunches
the group from the last VALID checkpoint manifest. The decision is visible
as FAILURE_CLASSIFIED / RECOVERY_ACTION_SELECTED events on the run.

The point to check is `resumed_from` in the final trial: recovery, not a
restart from zero.

Requires torchrun on PATH — see `_preflight.ACTIVATE_HINT`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # sibling _preflight, not a package

import flashruntime as flash  # noqa: E402
from _preflight import hold_viewer, require, require_torchrun  # noqa: E402
from flashruntime.integrations import pytorch as fr_torch  # noqa: E402

EXAMPLES = Path(__file__).parent

# user_pytorch/train.py trains on 512 samples at batch 32, split across ranks.
SAMPLES, BATCH = 512, 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--kill-at-step", type=int, default=400)
    parser.add_argument("--watch", action="store_true", help="open the live run page")
    args = parser.parse_args()

    require_torchrun()

    # Bit-exact resume needs the resumed step to land on an epoch boundary:
    # on resume the `for` loop restarts the dataloader at batch 0, so a step
    # mid-epoch would replay a different sample order and the loss would
    # drift. Catch it here rather than let the demo quietly prove less.
    align = SAMPLES // (BATCH * args.nproc)
    for name, value in (("--checkpoint-every", args.checkpoint_every),
                        ("--kill-at-step", args.kill_at_step)):
        require(
            value % align == 0,
            f"{name}={value} is not a multiple of {align} (batches per rank per epoch\n"
            f"at --nproc {args.nproc}). Resume would restart the dataloader mid-epoch\n"
            f"and the loss would no longer match the uninterrupted run.",
        )
    require(args.kill_at_step < args.steps,
            f"--kill-at-step ({args.kill_at_step}) must be below --steps ({args.steps})")

    script_args = (
        f"--steps {args.steps} --checkpoint-every {args.checkpoint_every} "
        f"--kill-at-step {args.kill_at_step}"
    )
    print(f"training to step {args.steps}, dying at step {args.kill_at_step}, "
          f"checkpointing every {args.checkpoint_every}")

    run = flash.submit(
        fr_torch.ddp("train.py", source=str(EXAMPLES / "user_pytorch"),
                     nproc_per_node=args.nproc, script_args=script_args),
        max_restarts=1,                      # the recovery budget — 0 would just FAIL
        watch=True if args.watch else False,
    )

    print(f"\nstate: {run.state.value}   attempts: {len(run.attempts)}")
    for attempt in run.attempts:
        print(f"  {attempt['attempt_id']:<16} {attempt['state']}")

    print("\nrecovery decision:")
    for event in run.events:
        if event["type"] in ("FAILURE_CLASSIFIED", "RECOVERY_ACTION_SELECTED"):
            print(f"  {event['type']:<26} {event.get('message', '')}")

    resumed_from = run.trials[0].get("resumed_from") if run.trials else None
    print(f"\nfinal trial: {run.trials[0] if run.trials else None}")

    ok = (run.state.value == "SUCCEEDED"
          and len(run.attempts) == 2
          and resumed_from == args.kill_at_step)
    print("\nRESULT:", "PASS" if ok else
          f"FAIL (expected SUCCEEDED, 2 attempts, resumed_from={args.kill_at_step}; "
          f"got resumed_from={resumed_from})")

    if args.watch:
        hold_viewer(run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
