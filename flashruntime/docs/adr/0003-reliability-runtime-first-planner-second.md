# ADR-0003: Reliability runtime first; planner as an explainable feasibility filter

Date: 2026-07-19 · Status: accepted

## Context

FlashRuntime's product hypothesis ("distributed workload compiler and
reliability runtime") bundles two different products:

1. A **reliability runtime** — job/task/attempt/lease state, checkpoint
   catalog, failure taxonomy, recovery orchestration, one execution contract
   across backends.
2. A **workload compiler** — automatic distributed-strategy selection from
   constraints via memory/communication/cost models.

A principal-engineer evaluation (July 2026, workspace-root
`FLASHRUNTIME_EVALUATION.md`) found (1) sound, under-served, and buildable
incrementally, and (2) research-grade: Alpa — the best-known
auto-parallelization system — was archived in October 2024; DeepSpeed's
Autotuner is engine-specific, profiling-bound, and cannot tune offloading;
static activation-memory estimation carries tens-of-percent error bars.

## Decision

1. **The reliability runtime is the spine and is built first.** Mode A
   (leases/heartbeats/idempotent commit) before Mode B (coordinated
   training), because the lease protocol is the layer no existing library
   provides.
2. **The planner ships as a deterministic, explainable feasibility filter
   over a curated strategy menu** — closed-form arithmetic kills infeasible
   plans, an optional short profiling run tightens estimates, and every plan
   carries its rejections with the numbers. It is *not* marketed or built as
   an optimizing compiler.
3. **The planner consumes the runtime's ledger**: measured memory,
   throughput, checkpoint durations, and failure rates from real runs feed
   back into estimates (`basis: static | profiled | ledger` on every
   number). The runtime is the planner's dataset.
4. **The planner emits a backend-neutral, versioned, frozen `StrategyPlan`**
   (hashable; recorded on the job attempt) that strategy compilers translate
   into torchrun/DeepSpeed/Ray configuration. The planner package never
   imports framework code.
5. **Four orthogonal internal axes**: providers (get machines), launchers
   (start processes), strategies (configure execution), recipes (integrate
   user code). Hugging Face Transformers/PEFT are recipes — workload layer,
   not execution backends.
6. **Library stances**: build on torchrun/Elastic, DDP, FSDP2 (skip
   deprecated FSDP1), PyTorch Distributed Checkpoint, Ray Core; DeepSpeed
   later only for capabilities FSDP2 lacks (NVMe offload, MoE). Do not build
   on HF Accelerate (overlapping ownership of launch/strategy decisions) or
   Ray Train (V1→V2 migration in progress). Reserve a Mode C
   (`recovery_model: per_step_elastic`, torchft-class) in the schema without
   building it.
7. **Execution modes**: Mode 0 local single-process is first-class (the
   planner must be able to recommend *not* distributing); Mode A independent
   tasks; Mode B coordinated training with whole-group restart; Mode C
   reserved.

## Consequences

- Module build order follows vertical slices (workspace-root
  `PLAN_2WEEKS.md`): `leases/` → executor contract → `checkpoint/` →
  `recovery/` → `planner/` (+ `strategies/`, `launchers/`, `recipes/`).
  No empty scaffold packages in the public repo.
- The first planner deliverable is a standalone `flash plan` CLI for
  single-node fine-tuning (feasibility + ranked explained plans), useful
  with no cluster attached.
- "No valid strategy" is a first-class planner output with nearest-miss
  analysis, and an OOM in a launched job is treated as a planner defect
  (estimate vs actual logged as a regression case).
- Checkpoint validity is by construction: shard parts upload first, the
  manifest is written last after hash verification — no manifest, no
  checkpoint.
