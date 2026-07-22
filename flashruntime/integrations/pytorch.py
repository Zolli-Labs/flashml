"""PyTorch adapter: launch conventions only. torchrun starts N processes
and hands each RANK/WORLD_SIZE — the user's code (or
flashruntime.torch.prepare) wires DDP from there. No torch import here.
"""

from __future__ import annotations

import shlex

from flashruntime.protocol.plan_v1alpha1 import CheckpointPolicy
from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source


def ddp(
    script: str,
    *,
    source: str = ".",
    nproc_per_node: int = 2,
    nnodes: int = 1,
    script_args: str = "",
    env: dict[str, str] | None = None,
) -> CommandWorkload:
    if nnodes > 1:
        raise NotImplementedError(
            "multi-node rendezvous is a launcher concern — later slice (spec §10); "
            "--standalone below is single-node by definition"
        )
    command = [
        "torchrun",
        f"--nproc-per-node={nproc_per_node}",
        f"--nnodes={nnodes}",
        "--standalone",
        script,
        *shlex.split(script_args),
    ]
    return CommandWorkload(
        command=command,
        source=Source(path=source),
        env=env or {},
        mode="coordinated",
        checkpoint=CheckpointPolicy(
            backend="local_manifest",
            note="flashruntime.torch.checkpoint: parts-first/manifest-last under FLASHML_CKPT_DIR",
        ),
        outputs=OutputSpec(collect=["metrics.json"]),
    )
