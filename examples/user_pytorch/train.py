"""A PyTorch script whose ONLY FlashRuntime coupling is flashruntime.torch.

The model, loss, data, and loop are ordinary PyTorch. The same file runs
three ways:

    python train.py --steps 200                          # single process
    torchrun --nproc-per-node=2 --standalone train.py    # DDP by hand
    flash.submit(integrations.pytorch.ddp(...))          # operated by FlashRuntime

Deterministic on CPU (fixed seeds; under DDP, ft.prepare swaps in a seed-0
DistributedSampler whose order repeats every epoch) so a killed-and-resumed
run reproduces the uninterrupted result — recovery must not change the math.

Bit-exact resume has one alignment constraint: on resume the `for` loop
restarts the dataloader at batch 0, so the resumed step must sit on an
epoch boundary — a multiple of batches-per-rank-per-epoch (here
512 samples / 32 batch / 2 ranks = 8). Keep --checkpoint-every and
--kill-at-step multiples of 8 (×2 ranks) or the 1e-6 loss comparison in
tests/test_examples_e2e.py will drift.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader, TensorDataset

import flashruntime.torch as ft


def make_data(n: int = 512, d: int = 16, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    y = ((x @ w).squeeze(1) > 0).long()
    return TensorDataset(x, y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--kill-at-step",
        type=int,
        default=None,
        help="simulate a crash (fresh runs only; resumed retries finish)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loader = DataLoader(make_data(), batch_size=32, shuffle=False)

    model, optimizer, loader = ft.prepare(model, optimizer, loader)
    start = ft.start_step()
    # ft.prepare put the model on ft.device() ("cpu" or this rank's "cuda:N");
    # ordinary PyTorch then owns moving each batch to the SAME device. On CPU
    # this is a no-op, so the CPU path is byte-for-byte unchanged; on a GPU box
    # it is what makes the run actually train on CUDA instead of crashing with a
    # cpu-vs-cuda device mismatch (caught by tests/test_gpu_e2e.py on real GPUs).
    device = ft.device()

    step = start
    loss = torch.tensor(0.0)
    while step < args.steps:
        for x, y in loader:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            ft.checkpoint(model, optimizer, step=step, every=args.checkpoint_every)
            ft.log_metrics({"step": step, "loss": round(loss.item(), 6)})
            if args.kill_at_step and start == 0 and step >= args.kill_at_step:
                raise SystemExit(3)  # fresh run only — the retry resumes past this

    ft.checkpoint(model, optimizer, step=step)  # final checkpoint
    if ft.is_main():
        metrics = {
            "steps": step,
            "resumed_from": start,
            "final_loss": round(loss.item(), 6),
            # where/how this rank actually trained — "cpu"/"cuda:0" and the
            # torch.distributed backend ("gloo" on CPU, "nccl" on GPU, None
            # single-process). Lets a GPU run PROVE the CUDA/nccl path that a
            # CPU box can't (tests/test_gpu_e2e.py). Additive: CPU e2e ignores
            # these keys.
            "device": ft.device(),
            "backend": ft.backend(),
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        print(metrics)


if __name__ == "__main__":
    main()
