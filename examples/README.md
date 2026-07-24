# Examples

Runnable entry points for the two public workflows. Each uses only the
public `flashruntime` API and asserts (or prints) a meaningful result.

## One story per script

Each demo runs exactly one thing, takes flags, and prints PASS/FAIL. Run
them in order, or run only the one you care about. `--watch` opens the live
run page and holds it open (the viewer server is a daemon thread of the
demo's own process, so it dies when the script returns).

- `demo_sklearn_sweep.py` — a 6-point hyperparameter grid over
  `user_sklearn/train.py`, a script with **no flashruntime import at all**.
  The contract is convention only: CLI flags in, `metrics.json` out.
- `demo_pytorch_ddp.py` — 2-process CPU DDP (gloo) over
  `user_pytorch/train.py`, which uses the three `flashruntime.torch` verbs.
  `--vanilla` points the same launcher at `user_pytorch_vanilla/train.py`
  (hand-wired DDP, no flashruntime import) to show what the verbs buy.
- `demo_kill_and_resume.py` — one `submit(max_restarts=1)`: the run dies
  mid-training, `recovery.classify()`/`decide()` pick `restart_group`, and
  it resumes from the last valid checkpoint manifest. Checks `resumed_from`.
- `bring_your_code_demo.py` — all three of the above back to back, the
  original single-shot tour.
- `plan_quickstart.py` — the strategy planner over three workload kinds
  (transformer fine-tune, PyTorch training, classical ML); no cluster
  required.
- `plan-qwen7b-lora.yaml`, `job-kmeans.yaml` — spec files for the
  `flashruntime plan` / job CLIs.

## Running them

**Activate the venv — do not call its interpreter by path.** The PyTorch
adapter launches a bare `torchrun`, resolved through `PATH`;
`.venv/bin/python examples/...` never puts `.venv/bin` on `PATH`, so the
launch fails. The single-purpose demos above exit non-zero with an
explanation, but `bring_your_code_demo.py` silently skips both PyTorch acts.

```bash
source .venv/bin/activate       # from the repository root
which torchrun                  # sanity check: .../.venv/bin/torchrun

python examples/demo_sklearn_sweep.py
python examples/demo_pytorch_ddp.py --steps 200 --watch
python examples/demo_kill_and_resume.py --steps 800 --kill-at-step 400
python examples/plan_quickstart.py          # planner only, no torch needed
```

Future examples belong here only after their implementation exists and the
example asserts a meaningful result. Planned APIs stay in the design docs
rather than executable-looking placeholder scripts.
