# The FlashRuntime Strategy Planner — architecture and code walkthrough

The planner answers one question: **given what you want to train, what you
have, and what you care about — how should it run, and why?** You give it a
`PlanRequest` (workload + resources + objective); it returns a `PlanReport`
containing a selected `StrategyPlan` (which distributed method, from which
libraries, with which knobs) and a verdict — with arithmetic — for **every**
candidate it considered, including the rejected ones.

```bash
flashruntime plan examples/plan-qwen7b-lora.yaml     # no cluster required
python examples/plan_quickstart.py
```

```python
import flashruntime as flash

report = flash.plan(flash.PlanRequest(
    workload=flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="lora",
                                       train_tokens_m=25),
    resources=flash.Resources(gpus=4, gpu_type="RTX4090",
                              hourly_cost_usd_per_gpu=0.44),
    objective=flash.Objective(mode="balanced", max_cost_usd=20,
                              deadline_minutes=240),
))
print(flash.render(report))
```

Design authority: ADR-0003 (`docs/adr/0003-reliability-runtime-first-planner-second.md`).
The planner is deliberately a **deterministic, explainable feasibility
filter over a curated strategy menu** — not an auto-parallelizing compiler.
Same inputs (and planner version) ⇒ same plan, always.

---

## 1. The pipeline

```
PlanRequest
   │ resolve.py      fill blanks: params from catalog, shape, per-GPU VRAM
   ▼
candidates.py        walk the curated menu for this workload class,
   │                 evaluate each candidate:
   │                   memory.py    → fits? (with safety bands)
   │                   comm.py      → network good enough? (scaling efficiency)
   │                   timecost.py  → minutes + dollars (when inputs allow)
   │                 apply hard constraints (deadline, budget, allow_* flags)
   ▼
selector.py          rank survivors by objective mode; mint the frozen
   │                 StrategyPlan from the winner; nearest-miss hint if none
   ▼
PlanReport ──▶ explain.py renders it for humans
```

Hard rule enforced throughout: **no ML framework is imported anywhere in
`planner/`**. The planner reasons about PyTorch, DeepSpeed, Ray, and
Transformers *by name and by number*. That keeps it importable anywhere
(the CLI plans on a laptop with nothing installed) and keeps the
StrategyPlan backend-neutral — strategy compilers translate it into real
torchrun/DeepSpeed configuration at execution time (future work).

## 2. The contract — `protocol/plan_v1alpha1.py`

Wire-visible models, versioned like everything else in `protocol/`:

- **Workloads** (a discriminated union on `kind`) — classification is
  *structural*, never label-based; "LoRA" says which parameters train, not
  which strategy to use:
  - `TransformerFineTune` — full / LoRA / QLoRA; the flagship. Carries what
    the estimators need: params (or a catalog model name), precision,
    optimizer, seq_len, micro-batch, optional shape and token count.
  - `PyTorchTraining` — generic deep learning; activation memory is
    user-supplied because arbitrary models have no derivable shape.
  - `ClassicalML` — sklearn/XGBoost; RAM-bound, CPU-first.
  - `IndependentTasks` — HPO / batch inference / evaluation; Mode A.
- **`Resources`** — GPU count/type/VRAM, hosts, interconnect class, host
  RAM (offload must fit *somewhere*), optional hourly price.
- **`Objective`** — mode (`cheapest|fastest|balanced|reliable`), budget,
  deadline, and the permission flags (`allow_quantization`,
  `allow_cpu_offload`, `allow_nvme_offload`).
- **`StrategyPlan`** — the answer: workload mode, strategy family + knobs,
  launcher, topology, the `libraries` list (name + role + purpose — the
  "which library does what" answer), checkpoint policy, estimates, and
  `selected_because`. `plan_id` is a content hash: plans are frozen and
  auditable.
- **`PlanReport`** — the selected plan plus a `CandidateVerdict` for every
  candidate (status ∈ selected / feasible / infeasible / rejected_policy /
  rejected_dominated, each with its reasons), warnings, and — when nothing
  survived — `no_valid_strategy_hint` with the minimal unlocking relaxation.
- **`Estimate`** — a number that admits where it came from:
  `basis: static | profiled | ledger`. Today everything is `static`;
  profiling runs and ledger history are designed to *replace* these values,
  not decorate them.

## 3. Reference data — `planner/catalog.py`

Three small tables, all conservative, all meant to be overridden by
measurement over time:

- `MODEL_CATALOG`: ~14 known models → parameters, hidden size, layers
  ("Qwen/Qwen2.5-7B" resolves forgivingly; unknown models must state
  `parameters_b`). `derive_transformer_shape()` guesses a typical shape from
  size class when nothing better exists — and the estimate says so.
- `GPU_CATALOG`: VRAM + *conservative dense* bf16 TFLOPs per GPU class
  (never the sparsity marketing number). Unknown GPU ⇒ user must give
  `vram_gb`; time/cost estimates are then omitted, not invented.
- `INTERCONNECT_GBPS`: effective bandwidth + latency floor per link class,
  NVLink → PCIe → IB → Ethernet → WAN. FlashNode's network benchmark
  replaces these with per-pool measurements later.
- The policy constants: `ASSUMED_MFU = 0.30`, `FLAT_OVERHEAD_GB = 2.5`, and
  the safety bands `AUTO_OK_FRACTION = 0.80` / `HARD_LIMIT_FRACTION = 0.95`.

**Adding a model or GPU = adding a dict entry.** That is the supported
extension path; no code changes.

## 4. Memory — `planner/memory.py`

The component model (evaluation §D). Baseline for mixed-precision AdamW:

```
bytes/param = 2 (bf16 weights) + 2 (grads) + 4 (fp32 master) + 4 (Adam m) + 4 (Adam v) = 16
```

PEFT changes *which* parameters pay the 16: frozen weights keep only their
2 bytes (or ~0.55 quantized NF4 for QLoRA), while gradients + optimizer
apply to the small trainable set (LoRA adapters ≈ `8·hidden·rank·layers`
params on the attention projections). This is why the same 7B model is
122 GB of training state for full fine-tuning but ~16 GB for LoRA.

`ShardingConfig` encodes what each strategy family divides across N GPUs:

| family | weights | grads | optimizer | note |
|---|---|---|---|---|
| single_gpu / ddp | – | – | – | full replica per GPU |
| fsdp2 | ÷N | ÷N | ÷N | + transient per-layer all-gather peak (`2 bytes × params/layers`) |
| zero3_cpu_offload | ÷N | ÷N | → host RAM | host RAM is checked too — planners that only check VRAM kill hosts |

Activations are the least predictable component: `layers × seq × batch ×
hidden × 18 bytes` [assumption, flash-attention era, pessimistic]; with
activation checkpointing only layer inputs (2 bytes/elem) plus one live
layer survive, at ~+30% step time. A flat 2.5 GB covers CUDA context + NCCL
buffers + workspace.

**The safety bands, not the formulas, make launches safe**: total ≤ 80% of
VRAM ⇒ auto-OK; 80–95% ⇒ `profiling_required` (the plan says so); > 95% ⇒
infeasible. An OOM in a launched job is treated as a planner defect.

## 5. Communication — `planner/comm.py`

First-order per-step traffic:

- **DDP**: one gradient all-reduce per optimizer step, ring cost
  `≈ 2·(N−1)/N · grad_bytes`. For LoRA the gradient is only the adapters'
  (~26 MB for r=16 on 7B) — which is why LoRA+DDP tolerates weak links that
  full fine-tuning (~15 GB/step) cannot. The math makes this emergent; no
  special case needed.
- **FSDP2/ZeRO-3**: ≈ 3× parameter bytes per step across ~3·layers
  latency-sensitive collectives — NVLink/IB territory.

Verdict: scaling efficiency `E = t_compute / (t_compute + exposed_comm)`
with half the compute assumed overlappable. `E < 0.5` ⇒ infeasible
("communication dominates"), `0.5–0.7` ⇒ feasible with a warning. A group
that fits on one host uses the host-local link; only groups spanning hosts
pay the cluster interconnect.

## 6. Time and cost — `planner/timecost.py`

The standard FLOPs identity: forward+backward ≈ `6·params` FLOPs/token, so
`tokens/s/GPU ≈ MFU · TFLOPs / (6·params)`. Fleet rate = × GPUs × scaling
efficiency, then multiplicative penalties for slow knobs (activation
checkpointing +30%, QLoRA +35%, CPU offload ×4 — the last is
order-of-magnitude and says "profile before trusting"). Note LoRA does
**not** reduce the 6·params: activation gradients still flow through every
frozen layer; PEFT saves memory, not backward FLOPs.

Cost = wall-clock × GPUs × the user's hourly rate. Missing inputs (no token
count, unknown GPU throughput, no price) ⇒ the estimate is **omitted**,
never guessed — deadline/budget checks then emit a warning instead of a
false verdict.

## 7. Candidates — `planner/candidates.py`

The curated menu per workload class:

| workload | menu |
|---|---|
| transformer_finetune | `single_gpu`, `ddp`, `fsdp2`, `zero3_cpu_offload` × world sizes {1, 2, 4, …, all} × QLoRA variants (when LoRA + quantization allowed) |
| pytorch_training | `single_gpu`, `ddp`, `fsdp2` |
| classical_ml | `local_process`; `sharded_partial_fit` (Mode A) when the estimator supports it |
| independent_tasks | `lease_tasks` (Mode A) vs `local_sequential` baseline |

Notable policies, each encoding an evaluation decision:

- **Activation checkpointing auto-decide**: when the user didn't specify,
  try without; enable it only if that is what makes the candidate fit —
  the candidate is renamed (`…+ackpt`) and the reason recorded.
- **QLoRA is a generated variant**, not a user obligation: a LoRA request
  with `allow_quantization` spawns `qlora_*` candidates that compete on the
  numbers (they usually win on memory and lose on time).
- **`fsdp2 × qlora` is skipped** — quantized frozen weights + full param
  sharding is an unsupported combo in v1.
- **`zero3_cpu_offload` is DeepSpeed's slot** (the one thing FSDP2 lacks in
  our menu); it is gated on `allow_cpu_offload` and its offloaded bytes are
  checked against host RAM.
- **Independent tasks prefer the lease runtime** structurally: per-task
  failure isolation ("a lost node costs one task retry, never the job") —
  the same Mode A machinery the rebuild plan builds in Stages 0–3.

Every candidate ends as a `CandidateVerdict` + (if it survived) the fields
needed to mint a StrategyPlan; every elimination keeps its arithmetic.

## 8. Selection — `planner/selector.py`

Hard constraints were applied during evaluation (`rejected_policy` with the
violated number). The ranker only orders survivors, lexicographically:

- `cheapest` → (cost, time, workers) — unknown values sort last, so a plan
  can never win *because* its cost is unknown
- `fastest` → (time, cost, workers)
- `balanced` → cost first (the deadline is already a hard gate), time tiebreak
- `reliable` → fewest moving parts: smallest world size, no offload, no
  quantization, most memory headroom

The winner is minted into a `StrategyPlan`: knobs, the library stack with
roles, a checkpoint policy (static 300 s default; Young–Daly
`τ* = √(2·C·MTBF)` once checkpoint duration and pool failure rates are
measured), `selected_because` (including the runner-up comparison), and the
content-hash `plan_id`. When nothing survives, `_nearest_miss()` names the
minimal relaxation — "missed VRAM by 3.1 GB/GPU; allowing quantization may
unlock QLoRA" — because a dead end must still be a useful answer.

## 9. Worked example (the repo's own example file)

Qwen2.5-7B (7.62B params, from the catalog), LoRA r=16, 4 × RTX4090
(24 GB), 25M tokens, $0.44/GPU-h, balanced, ≤$20, ≤240 min:

- **LoRA single GPU**: weights 15.27 GB (7.62B × 2 bytes + adapters) +
  grads 0.03 + optimizer 0.15 (adapters only!) + activations 3.7 +
  overhead 2.5 = **21.64 GB** → inside the 80–95% caution band ⇒ feasible,
  `profiling_required`. But ~385 min ⇒ rejected by deadline at 1 GPU.
- **DDP ×4**: same 21.64 GB replica per GPU; all-reduce is only the 26 MB
  adapter gradient ⇒ efficiency 1.00 even on PCIe; ~96 min, $2.82 ⇒
  **selected** (cheapest-fastest survivor).
- **FSDP2 ×4**: 11.15 GB/GPU (sharded) but 0.71 efficiency (per-layer
  all-gathers on PCIe) ⇒ feasible, slower, loses the ranking.
- **QLoRA ×4**: 10.6 GB/GPU, +35% step time ⇒ feasible, loses on time.
- **ZeRO-3 offload**: fits easily, ×4 step time ⇒ rejected by deadline —
  with the number shown.

Run it: `flashruntime plan examples/plan-qwen7b-lora.yaml`.

## 10. What the planner does *not* do yet (and how it will)

| Gap | Today | Planned replacement |
|---|---|---|
| Activation constant, MFU, offload penalty | labeled `[assumption]` constants | **profiling stage**: 3 warmup + 20 measured steps in the real container, feeding `basis: profiled` |
| Pool failure rates, checkpoint duration | static 300 s checkpoint default | **ledger feedback**: measured runs → `basis: ledger`, Young–Daly intervals |
| Execution of the plan | plan only | strategy compilers (`strategies/`) translate StrategyPlan → torchrun args / DeepSpeed config; the lease runtime executes Mode A (rebuild Stages 0–3) |
| Tensor/pipeline parallelism, MoE, RL, pretraining | honestly out of envelope — the report says so | reserved names in the schema; month-3+ per the roadmap |
| Micro-batch / grad-accum tuning | taken as given | knob search *within* a family once profiling exists |

## 11. Extending the planner

- **New model/GPU/link**: add a `catalog.py` entry.
- **New strategy family**: add a `ShardingConfig` mapping + a branch in
  `_training_stack()` (its libraries) + include it in the family list in
  `candidates.py`. The estimators need no changes unless the family moves
  memory somewhere new.
- **New workload kind**: add the pydantic model to `plan_v1alpha1.py`
  (additive — fine within v1alpha1), a `*_candidates()` function, and a
  dispatch line in `generate()`. Follow the honesty contract: omit what you
  can't estimate; label what you assume.
- **Tests are the spec**: `tests/test_planner.py` pins the arithmetic
  (16 B/param, QLoRA ~4× weight shrink, host-RAM offload gate, WAN
  rejection, determinism). Change a formula ⇒ justify it against
  `FLASHRUNTIME_EVALUATION.md` and update the pinned numbers.
