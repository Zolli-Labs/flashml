# Tutorial: make a ConvNet fault-tolerant

This is the flagship walkthrough. You start with an ordinary PyTorch ConvNet
and an ordinary training loop, and end with the *same* code — unchanged in its
math — running as a 2-process DDP job that survives a crash and resumes from
its last verified checkpoint, with a live page you can watch.

The promise throughout: **FlashRuntime operates your code; it never rewrites
it.** The model below is used verbatim, including one unusual thing it does,
because "your code, unmodified" is the whole point.

If you have not installed FlashRuntime and PyTorch yet, do the
[Get started](../get-started.md) page first — you need `pip install
flashruntime` and a CPU build of `torch`. No GPU, no cluster.

---

## 1. The plain script (no FlashRuntime)

Here is the model and a plain single-process training loop. It reads its
hyperparameters from `argparse` and writes a `metrics.json` at the end — the
one convention FlashRuntime asks of any script. There is **no FlashRuntime
import** yet.

```python
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class OurConvNet(torch.nn.Module):
    def __init__(self, num_outputs=20):
        super(OurConvNet, self).__init__()
        self.Conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.Sigma = nn.Sigmoid()
        self.Avg = nn.AvgPool2d(2, stride=2, padding=0)
        self.Conv2 = nn.Conv2d(64, 128, 5, 1)
        self.Fl = nn.Flatten()
        self.Linear = nn.Linear(128 * 5 * 5, num_outputs)

    def forward(self, x):
        out = self.Sigma(x)   # NOTE: the Sigmoid is applied to the INPUT first.
        out = self.Conv1(out)
        out = self.Avg(out)
        out = self.Conv2(out)
        out = self.Avg(out)
        out = self.Fl(out)
        out = self.Linear(out)
        return out


def make_data(n=256, num_outputs=20, seed=0):
    # Synthetic data with a fixed seed keeps the run deterministic on CPU.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 28, 28, generator=g)
    y = torch.randint(0, num_outputs, (n,), generator=g)
    return TensorDataset(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = OurConvNet(num_outputs=20)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loader = DataLoader(make_data(), batch_size=32, shuffle=False)

    step = 0
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

    with open("metrics.json", "w") as f:
        json.dump({"steps": step, "final_loss": round(loss.item(), 6)}, f)
    print("done", step, loss.item())


if __name__ == "__main__":
    main()
```

Run it like any script:

```bash
python train.py --steps 30
```

### The shape math (so `128 * 5 * 5` is not a magic number)

The input is `3 × 28 × 28`. Follow one image through `forward`, and every
`nn.Linear(128 * 5 * 5, ...)` factor falls out of the convolution arithmetic:

```
 28  --Conv1(k5, s1, p2)-->  28   (padding 2 keeps the size)
     --Avg(2, 2)---------->  14
     --Conv2(k5, s1, p0)-->  10   (no padding: 14 - 5 + 1 = 10)
     --Avg(2, 2)---------->   5
  =>  Flatten = 128 channels · 5 · 5 = 3200 = nn.Linear(128 * 5 * 5, num_outputs)
```

So `3 × 28 × 28` is the input size that makes the flattened feature map land
**exactly** on `128 * 5 * 5`. Feed a different size and the `Flatten → Linear`
handoff mismatches; keep 28 and it fits.

### One honest note about this model

`forward` applies `self.Sigma` (a `Sigmoid`) to the **raw input** before the
first convolution — an unusual ordering (a sigmoid is normally an activation
*between* layers, not a preprocessing step on the pixels). We keep it exactly
as written. FlashRuntime's promise is to operate *your* code, so we do not
"fix" the model to match convention — we run what you wrote.

---

## 2. Make it resumable: `import flashruntime.torch as ft`

The plain script has no checkpoints, so a crash starts over from step 0. One
import — `flashruntime.torch` (aliased `ft`) — gives you launch-anywhere DDP
and fault-tolerant checkpointing without rebuilding any framework machinery.
Here is the whole change, as a diff:

```diff
 import argparse
 import json

 import torch
 import torch.nn as nn
 from torch.utils.data import DataLoader, TensorDataset

+import flashruntime.torch as ft
+

 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--steps", type=int, default=40)
     parser.add_argument("--lr", type=float, default=0.05)
+    parser.add_argument("--checkpoint-every", type=int, default=8)
+    parser.add_argument("--kill-at-step", type=int, default=None,
+                        help="simulate a crash (fresh runs only; a resumed retry finishes)")
     args = parser.parse_args()

     torch.manual_seed(0)
     model = OurConvNet(num_outputs=20)
     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
     loader = DataLoader(make_data(), batch_size=32, shuffle=False)

-    step = 0
+    model, optimizer, loader = ft.prepare(model, optimizer, loader)
+    start = ft.start_step()
+
+    step = start
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
+            ft.checkpoint(model, optimizer, step=step, every=args.checkpoint_every)
+            ft.log_metrics({"step": step, "loss": round(loss.item(), 6)})
+            if args.kill_at_step and start == 0 and step >= args.kill_at_step:
+                raise SystemExit(3)  # fresh run only — the retry resumes past this

-    with open("metrics.json", "w") as f:
-        json.dump({"steps": step, "final_loss": round(loss.item(), 6)}, f)
-    print("done", step, loss.item())
+    ft.checkpoint(model, optimizer, step=step)  # final checkpoint
+    if ft.is_main():
+        metrics = {"steps": step, "resumed_from": start, "final_loss": round(loss.item(), 6)}
+        with open("metrics.json", "w") as f:
+            json.dump(metrics, f)
+        print(metrics)
```

What each added call does — and *nothing more* (this is the whole surface):

- **`ft.prepare(model, optimizer, loader)`** — launched distributed
  (`WORLD_SIZE > 1`), it initializes torch's own process group (`gloo` on CPU,
  `nccl` on GPU), wraps the model in `DistributedDataParallel`, and swaps the
  DataLoader's sampler for a seed-0 `DistributedSampler` so each rank sees a
  disjoint shard. It then restores the newest **valid** checkpoint if one
  exists. Launched as plain `python train.py`, it is a no-op passthrough.
- **`ft.start_step()`** — `0` on a fresh run, `>0` after a resume (it is the
  step `prepare` restored to). The loop starts from it.
- **`ft.checkpoint(model, optimizer, step=step, every=8)`** — rank 0 writes a
  checkpoint under the parts-first / manifest-last contract; the manifest is
  written **last**, so a half-written checkpoint is never treated as valid.
- **`ft.log_metrics({...})`** — rank 0 appends one JSON line to
  `metrics.jsonl`; this is the streaming series the live page draws as a loss
  curve. It never raises.

The full surface is three verbs plus read-only launch-fact accessors — see
the [torch helper reference](../reference/torch-helper.md). There are no FSDP
policies, no autocast, no DeepSpeed config here: that is deliberate (ADR-0003 —
we do not rebuild Accelerate). A script that wants those uses the real
framework features; the launcher still launches it correctly.

The finished, copy-paste-runnable version of this script is in
[section 5](#5-the-final-script-copy-paste-runnable) below.

---

## 3. Operate it: `flash.submit(fr_torch.ddp(...))`

Now hand the script to FlashRuntime. The `integrations.pytorch` adapter builds
the `torchrun` command that starts N worker processes; `flash.submit()` runs
it, waits, collects the artifacts, and hands back a `Run`:

```python
import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch

run = flash.submit(fr_torch.ddp(
    "train.py",
    source=".",              # directory holding train.py
    nproc_per_node=2,        # two worker processes on this host — gloo/CPU, no GPU
    script_args="--steps 16",
))
print(run.state.value, run.trials)
# SUCCEEDED [{'steps': 16, 'resumed_from': 0, 'final_loss': 2.98...}]
```

`ddp(script, *, source=".", nproc_per_node=2, nnodes=1, script_args="",
env=None)` emits `torchrun --nproc-per-node=2 --nnodes=1 --standalone
--local-addr=127.0.0.1 train.py --steps 16`. Two processes rendezvous on
loopback; each gets its `RANK` / `WORLD_SIZE` / `LOCAL_RANK`, and
`ft.prepare()` wires DDP from there. `nnodes > 1` raises `NotImplementedError`
today — multi-node rendezvous is a later slice.

> **`torchrun` must be on your `PATH`.** It ships with `torch`; if you
> installed torch into a virtualenv, run from that environment so its
> `torchrun` is found.

---

## 4. Crash it, and let it recover

Now the payoff. Point a run at a fixed `output_dir` so its checkpoint tree
persists, tell the script to crash mid-way with `--kill-at-step`, and give
`submit()` a restart budget with `max_restarts=1`. That is the **only** change
— one keyword argument — and recovery is automatic:

```python
import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch

run = flash.submit(
    fr_torch.ddp(
        "train.py",
        source=".",
        nproc_per_node=1,
        script_args="--steps 24 --checkpoint-every 8 --kill-at-step 8",
    ),
    output_dir="out/convnet",   # a FIXED dir — the checkpoint tree lives here
    max_restarts=1,             # one automatic recovery attempt
)
print(run.state.value, run.trials)
# SUCCEEDED [{'steps': 24, 'resumed_from': 8, 'final_loss': 2.99...}]
```

`resumed_from: 8` is the proof: the first attempt crashed at step 8, and the
retry **resumed from the step-8 checkpoint** rather than restarting at zero.
The run's event log tells the whole story:

```
LAUNCH_STARTED          task-000 launched (pid ...)
FAILURE_CLASSIFIED      task-000: worker_crash (exit 1)
RECOVERY_ACTION_SELECTED task-000: restart_group — a lost rank stops the group —
                        restart all workers from latest valid checkpoint
LAUNCH_STARTED          task-000-r1 launched (pid ...)
```

That decision is not a guess. FlashRuntime turned the crash into failure
signals, `classify()`'d them (a `torchrun`-wrapped worker death is a
non-deterministic `worker_crash`, not a code bug), and looked the class up in a
versioned, deterministic policy table. Same failure + same policy version ⇒
same action, every time — no LLM in the loop. The mechanics are the subject of
the [fault-tolerance tutorial](fault-tolerance.md).

One honest constraint on bit-exact resume: on resume the `for` loop restarts
the dataloader at batch 0, so the resumed step must land on an **epoch
boundary**. With 256 samples / batch 32 = 8 batches per epoch single-process,
keep `--checkpoint-every` (and `--kill-at-step`) multiples of that. Off a
boundary, the resumed run is still correct training — it just will not match an
uninterrupted run byte-for-byte.

---

## Watch it live

Pass `watch=True` (or just run at an interactive terminal, where it is the
default) and `flash.submit()` opens a live run page in your browser and records
its URL on `run.viewer_url`:

```python
import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch

run = flash.submit(
    fr_torch.ddp("train.py", source=".", nproc_per_node=2, script_args="--steps 16"),
    output_dir="out/convnet",
    watch=True,
)
print(run.viewer_url)   # http://127.0.0.1:<port>
```

The page polls a loopback server every couple of seconds and draws the run with
**zero external assets** (no CDN, no web font, no remote image) — it renders
with the network cut. Top to bottom, the panels are:

- **Header** — a colored state badge (RUNNING cyan, SUCCEEDED green, FAILED
  red), the exact command, the execution `mode`, `restarts` used out of your
  budget, and the total `attempts` count.
- **Topology** — a single machine box labeled `127.0.0.1 · localhost` with one
  node per attempt/rank, colored by state; a RUNNING node softly pulses. A
  2-process DDP run shows two nodes.
- **Loss** — an autoscaled curve built from the `metrics.jsonl` your
  `ft.log_metrics(...)` calls stream, with the latest value labeled. (No
  `log_metrics` calls ⇒ no curve — this panel is fed by that stream.)
- **Checkpoints** — one violet marker per manifest: its step, a
  **hash-verified** or **invalid** badge (re-verified at read time, not taken
  on trust), a ★ **latest** tag on the one recovery would restore, and the part
  count plus age.
- **Events** — newest first, the same log shown above;
  `FAILURE_CLASSIFIED` is amber and `RECOVERY_ACTION_SELECTED` is cyan, each
  carrying the failure class and the policy's human-readable reason.
- **Logs** — a collapsible tail of each attempt's captured stdout+stderr.

These docs are served from that same viewer at `/docs`, so the page you watch
and the page you are reading are one product.

---

## 5. The final script (copy-paste runnable)

The complete `train.py` from sections 2–4. It runs three ways from one file —
`python train.py`, `torchrun ... train.py`, or `flash.submit(fr_torch.ddp(
...))` — and needs only a CPU:

```python
"""ConvNet tutorial — your model, unmodified, made fault-tolerant.

    python train.py --steps 40                          # single process
    torchrun --nproc-per-node=2 --standalone train.py   # DDP by hand
    flash.submit(fr_torch.ddp("train.py", ...))         # operated by FlashRuntime

Shapes: input 3x28x28 -> Conv1(k5,p2) 28 -> Avg 14 -> Conv2(k5) 10 -> Avg 5,
so Flatten = 128 * 5 * 5 = 3200 = nn.Linear(128*5*5, num_outputs).
"""
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flashruntime.torch as ft


class OurConvNet(torch.nn.Module):
    def __init__(self, num_outputs=20):
        super(OurConvNet, self).__init__()
        self.Conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.Sigma = nn.Sigmoid()
        self.Avg = nn.AvgPool2d(2, stride=2, padding=0)
        self.Conv2 = nn.Conv2d(64, 128, 5, 1)
        self.Fl = nn.Flatten()
        self.Linear = nn.Linear(128 * 5 * 5, num_outputs)

    def forward(self, x):
        out = self.Sigma(x)   # NOTE: Sigmoid applied to the INPUT first — kept verbatim.
        out = self.Conv1(out)
        out = self.Avg(out)
        out = self.Conv2(out)
        out = self.Avg(out)
        out = self.Fl(out)
        out = self.Linear(out)
        return out


def make_data(n=256, num_outputs=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 28, 28, generator=g)
    y = torch.randint(0, num_outputs, (n,), generator=g)
    return TensorDataset(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--kill-at-step", type=int, default=None,
                        help="simulate a crash (fresh runs only; a resumed retry finishes)")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = OurConvNet(num_outputs=20)
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
        metrics = {"steps": step, "resumed_from": start, "final_loss": round(loss.item(), 6)}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        print(metrics)


if __name__ == "__main__":
    main()
```

---

## Where to go next

- **[Fault tolerance, in depth](fault-tolerance.md)** — how a crash becomes
  signals, a class, and a typed recovery action.
- **[PyTorch guide](../guides/pytorch.md)** — the two launch paths (already-DDP
  scripts vs. the `ft.` helper) and every caveat.
- **[Architecture](../concepts/architecture.md)** — the four axes, leases,
  manifests, and recovery that make the above work.
