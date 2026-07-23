# Examples

Runnable entry points for the two public workflows. Each uses only the
public `flashruntime` API and asserts (or prints) a meaningful result.

- `bring_your_code_demo.py` — FlashRuntime operating **your** code: an
  sklearn hyperparameter sweep, an optional 2-process CPU DDP run (when
  torch is installed), and the kill-and-resume story (crash mid-training,
  resubmit, resume from the last valid checkpoint manifest). The user code
  it runs lives in `user_sklearn/`, `user_pytorch/`, and
  `user_pytorch_vanilla/`.
- `plan_quickstart.py` — the strategy planner over three workload kinds
  (transformer fine-tune, PyTorch training, classical ML); no cluster
  required.
- `plan-qwen7b-lora.yaml`, `job-kmeans.yaml` — spec files for the
  `flashruntime plan` / job CLIs.

Run any script from the repository root, e.g.:

```bash
.venv/bin/python examples/bring_your_code_demo.py
.venv/bin/python examples/plan_quickstart.py
```

Future examples belong here only after their implementation exists and the
example asserts a meaningful result. Planned APIs stay in the design docs
rather than executable-looking placeholder scripts.
