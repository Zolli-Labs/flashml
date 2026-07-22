"""A PyTorch script whose ONLY FlashRuntime coupling is flashruntime.torch.

The model, loss, data, and loop are ordinary PyTorch. The same file runs
three ways:

    python train.py --steps 200                          # single process
    torchrun --nproc-per-node=2 --standalone train.py    # DDP by hand
    flash.submit(integrations.pytorch.ddp(...))          # operated by FlashRuntime

Deterministic on CPU (fixed seeds, no shuffle) so a killed-and-resumed run
reproduces the uninterrupted result — recovery must not change the math.
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

    step = start
    loss = torch.tensor(0.0)
    while step < args.steps:
        for x, y in loader:
            if step >= args.steps:
                break
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
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        print(metrics)


if __name__ == "__main__":
    main()
