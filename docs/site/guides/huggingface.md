# Hugging Face guide

FlashRuntime **operates** your Hugging Face job — it never rewrites your
`Trainer`. HF Trainer already wraps DDP/FSDP internally when it is launched by
`torchrun`, so *launching* an HF job is just the [PyTorch path](pytorch.md).
What `integrations.huggingface` adds is the **callback seam** that commits
Trainer checkpoints as verified manifests and relays Trainer metrics.

`transformers` is imported only inside your training process — never in
FlashRuntime's core.

---

## Launching

`trainer(script, *, source=".", nproc_per_node=1, script_args="")` is a thin
wrapper over the PyTorch `ddp()` adapter, so everything in the
[PyTorch guide](pytorch.md) about launching and multi-process applies
unchanged:

```python
import flashruntime as flash
from flashruntime.integrations import huggingface as fr_hf

run = flash.submit(fr_hf.trainer(
    "train_hf.py",
    source="~/hf-project",
    nproc_per_node=1,
    script_args="--model_name_or_path bert-base-uncased",
))
```

---

## The callback seam

Inside your training script, wire the callback and the resume in the usual HF
way:

```python
from flashruntime.integrations import huggingface as fr_hf

trainer.add_callback(fr_hf.flashruntime_callback())    # on_save -> manifest, on_log -> metrics

resume = fr_hf.latest_checkpoint(training_args.output_dir)   # newest VALID checkpoint dir, or None
trainer.train(resume_from_checkpoint=resume)
```

- **`flashruntime_callback()`** builds a `TrainerCallback` whose `on_save`
  writes a verified manifest for `checkpoint-<step>/` (rank 0 only) and whose
  `on_log` relays metrics through `flashruntime.torch.log_metrics`. The
  `transformers` import is paid inside this factory, in your process.
- **`latest_checkpoint(output_dir)`** returns the storage prefix of the newest
  checkpoint dir with a *valid* manifest (`None` means fresh start) — pass it
  straight to `resume_from_checkpoint`.

So a Trainer run gets the same verified, parts-first / manifest-last checkpoint
guarantee as a hand-written loop: a half-written checkpoint is never selected
for resume, because the manifest is written last.

---

## Adding another framework

`integrations.huggingface` is the smallest possible adapter: `trainer(...)`
delegates straight to `ddp(...)`, and the only HF-specific code is the callback
that maps `on_save`/`on_log` onto FlashRuntime's manifest and metric contracts.
That is the extensibility pattern — teach FlashRuntime a new framework by
writing a small adapter under `flashruntime/integrations/` that (a) returns a
`CommandWorkload` describing what to launch, and (b), if the framework has its
own callback/hook system, maps those hooks onto `write_manifest` /
`log_metrics`. No core change is needed. See the
[PyTorch adapter](pytorch.md#adding-another-framework) for the base case.
