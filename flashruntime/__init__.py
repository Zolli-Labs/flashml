"""FlashRuntime — open fault-tolerant distributed ML runtime.

A clean, pure-Python core (`pip install flashruntime` brings only pydantic):

1. **Strategy planner** — decide *how* a job should run, with the math shown:

       import flashruntime as flash

       report = flash.plan(flash.PlanRequest(
           workload=flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="lora"),
           resources=flash.Resources(gpus=4, gpu_type="RTX4090"),
           objective=flash.Objective(mode="balanced", deadline_minutes=240),
       ))
       print(flash.render(report))

2. **Lease manager** (`flashruntime.leases`) — the Mode A reliability core:
   claim → heartbeat → expiry → requeue → idempotent commit, as a plain
   state machine anyone can embed (the FlashML coordinator is one consumer).

3. **Checkpoint catalog** (`flashruntime.checkpoint`) — manifests with
   parts-first / manifest-last commit: a partial checkpoint can never look
   valid, and recovery selects only verified, topology-compatible state.

4. **Recovery policy** (`flashruntime.recovery`) — the typed failure
   taxonomy and a versioned, deterministic action table. No agents, no
   guesswork: same failure + same policy version ⇒ same action.

Infrastructure integrations are opt-in extras, never core imports:
`[service]` FastAPI coordinator + CLI job commands · `[k8s]` KubeRay
backend · `[artifacts]`/`[oss]` MinIO / Alibaba OSS stores ·
`[prototype]`/`[sklearn]` the numpy-based local training engine.
"""

from typing import Any

from flashruntime.planner import PLANNER_VERSION, plan, render
from flashruntime.protocol.plan_v1alpha1 import (
    ClassicalML,
    IndependentTasks,
    Objective,
    PlanReport,
    PlanRequest,
    PyTorchTraining,
    Resources,
    StrategyPlan,
    TransformerFineTune,
)

__all__ = [
    # planner
    "plan",
    "render",
    "PLANNER_VERSION",
    "PlanRequest",
    "PlanReport",
    "StrategyPlan",
    "TransformerFineTune",
    "PyTorchTraining",
    "ClassicalML",
    "IndependentTasks",
    "Resources",
    "Objective",
    # prototype engine (lazy — needs the [prototype] extra)
    "Cluster",
    "Job",
    "JobEvent",
    "algorithms",
    "registered_providers",
]

# The pre-K8s prototype engine needs numpy; keep it importable exactly as
# before (flashruntime.Cluster, flashruntime.algorithms, ...) but resolve it
# lazily (PEP 562) so the pydantic-only core never pays for it.
_PROTOTYPE_EXPORTS = {"Cluster", "Job", "JobEvent", "algorithms", "registered_providers"}


def __getattr__(name: str) -> Any:
    if name in _PROTOTYPE_EXPORTS:
        # importlib, not `from flashruntime import ...`: an attribute-style
        # import would re-enter this __getattr__ and recurse.
        import importlib

        try:
            algorithms_mod = importlib.import_module("flashruntime.algorithms")
            adapters_mod = importlib.import_module("flashruntime.adapters")
            engine_mod = importlib.import_module("flashruntime.engine")
        except ImportError as exc:  # numpy missing
            raise ImportError(
                f"flashruntime.{name} is part of the prototype engine and needs "
                'the optional dependencies: pip install "flashruntime[prototype]"'
            ) from exc
        exports = {
            "Cluster": engine_mod.Cluster,
            "Job": engine_mod.Job,
            "JobEvent": engine_mod.JobEvent,
            "algorithms": algorithms_mod,
            "registered_providers": adapters_mod.registered_providers,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module 'flashruntime' has no attribute {name!r}")
