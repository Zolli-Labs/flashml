# FlashRuntime

**FlashRuntime operates your training job — it never rewrites it.** You keep
the model, the training loop, the loss, the data, and the framework you
already have. FlashRuntime wraps the reliability and reproducibility layer
around them: it launches your command, injects the environment it promised,
tracks your metrics, validates your checkpoints, retries on failure, and
collects your artifacts.

| You own | FlashRuntime operates |
|---|---|
| model, training loop, loss, data, framework | launch, environment, metric tracking, checkpoint validity, recovery, artifact collection |

The contract at the boundary is deliberately thin: **arguments in,
`metrics.json` out.** A script that already reads its hyperparameters from
`argparse` and writes a small JSON file of results needs *zero* FlashRuntime
imports to be operated. This is [ADR-0003](https://github.com/Zolli-Labs/flashruntime/blob/main/docs/adr/0003-reliability-runtime-first-planner-second.md)'s
fourth axis in practice: recipes integrate user code; the distributed math is
always done by your framework (PyTorch DDP, `torchrun`, Hugging Face, sklearn).

## The 60-second demo

Install it, point it at a script, and watch it run — recovering across crashes
on the way:

```python
import flashruntime as flash

run = flash.submit(
    flash.CommandWorkload(
        command="python train.py --epochs 5",
        source=flash.Source(path="~/my-project"),
        outputs=flash.OutputSpec(collect=["metrics.json"]),
    ),
    max_restarts=2,   # a crashed attempt is classified, then relaunched from
                      # the last VALID checkpoint — up to twice
    watch=True,       # opens the live run page and prints its URL
)

print(run.state.value)   # "SUCCEEDED" (or "FAILED")
print(run.artifacts)     # [PosixPath('.../metrics.json'), ...]
print(run.viewer_url)    # http://127.0.0.1:<port> — the live page (also /docs)
```

`flash.submit()` compiles that description into a launch spec, runs it as a
real subprocess, waits, and hands back a `Run`. `command` is `shlex`-split
(there is no shell — for a pipe, pass `command="bash -c '...'"`), and `source`
is a `flash.Source`, so `~` is expanded for you.

- **`max_restarts`** is the automatic fault-tolerance budget. On a FAILED
  attempt FlashRuntime turns the exit into failure signals, classifies them,
  and consults a *versioned, deterministic* recovery policy: a deterministic
  application bug fails fast (a retry only re-hits it); anything else relaunches
  the same spec from the job-scoped checkpoint, up to the budget. Same failure
  + same policy version ⇒ same action, every time — no LLM in the loop.
- **`watch=True`** opens the live run page in a browser (and records its URL on
  `run.viewer_url`). It draws the run's topology, its loss curve, its verified
  checkpoints, and every recovery decision, polling a loopback server with
  **zero external assets**. `watch` defaults to auto: on at an interactive
  terminal, off in a pipe or CI. The same page is served at `/docs`.

## What FlashRuntime does around your job

- **Launch** — starts your command as a subprocess (locally today; leased to a
  remote node through the coordinator when you compile it to a JobSpec).
- **Environment** — injects the env vars a run promises, so the same command is
  reproducible across machines.
- **Tracking** — reads the `metrics.json` your script writes and records it as a
  trial; a fan-out sweep merges each trial's parameters.
- **Checkpoints** — the parts-first / manifest-last contract means a
  half-written checkpoint can never look valid; recovery restores only a
  verified, topology-compatible manifest.
- **Recovery** — typed, deterministic, logged. Every retry emits a
  `FAILURE_CLASSIFIED` and a `RECOVERY_ACTION_SELECTED` event carrying the
  failure class and the policy's human-readable reason.
- **Artifacts** — copies your `outputs.collect` globs out of the run before the
  next trial can overwrite them.

## Next

Head to **[Get started](get-started.md)** to install FlashRuntime, run your
first job, and launch your first 2-process DDP run on CPU — no cluster, no GPU
required.
