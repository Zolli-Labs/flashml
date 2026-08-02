"""Contract tests for the future-work interfaces — written first (TDD).

These pin three things about every abstract seam: (1) it imports from its
final home, (2) ABCs refuse direct instantiation (the contract is real,
not decorative), (3) a minimal conforming implementation satisfies it —
so the first real implementation starts from a green example, and
(4) where an interface encodes *current* behavior (FIFO placement), the
semantics match what the live system does today.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# strategies: StrategyPlan → executable launch configuration
# ---------------------------------------------------------------------------


def test_strategy_compiler_is_abstract_and_implementable():
    from flashruntime.strategies import LaunchSpec, StrategyCompiler

    with pytest.raises(TypeError):
        StrategyCompiler()  # abstract — cannot instantiate

    class DummyCompiler(StrategyCompiler):
        family = "dummy"

        def supports(self, plan) -> bool:
            return plan.strategy_family == "dummy"

        def compile(self, plan) -> LaunchSpec:
            return LaunchSpec(
                argv=["python", "-m", "x"], env={"WORLD_SIZE": str(plan.workers)}
            )

    spec = DummyCompiler().compile(_plan(strategy_family="dummy", workers=4))
    assert spec.env["WORLD_SIZE"] == "4"
    assert spec.files == {}  # optional config files default empty


def test_compiler_registry_dispatches_by_support():
    from flashruntime.strategies import LaunchSpec, StrategyCompiler, compiler_for, register_compiler

    class OnlyDdp(StrategyCompiler):
        family = "ddp"

        def supports(self, plan):
            return plan.strategy_family == "ddp"

        def compile(self, plan):
            return LaunchSpec(argv=["torchrun"])

    register_compiler(OnlyDdp())
    assert compiler_for(_plan(strategy_family="ddp")).family == "ddp"
    with pytest.raises(LookupError, match="no strategy compiler"):
        compiler_for(_plan(strategy_family="pipeline_parallel"))


# ---------------------------------------------------------------------------
# launchers: start (and watch) process groups
# ---------------------------------------------------------------------------


def test_launcher_contract():
    from flashruntime.launchers import Launcher, LaunchHandle, LaunchState

    with pytest.raises(TypeError):
        Launcher()

    class DummyHandle(LaunchHandle):
        def poll(self):
            return LaunchState.SUCCEEDED

        def cancel(self):
            pass

    class DummyLauncher(Launcher):
        name = "dummy"

        def launch(self, spec, job_id, attempt_id):
            return DummyHandle()

    from flashruntime.strategies import LaunchSpec

    handle = DummyLauncher().launch(LaunchSpec(argv=["true"]), "j1", "a1")
    assert handle.poll() is LaunchState.SUCCEEDED


# ---------------------------------------------------------------------------
# recipes: workload integration behind one interface
# ---------------------------------------------------------------------------


def test_workload_recipe_contract_matches_executor_task_contract():
    from flashruntime.recipes import WorkloadRecipe

    with pytest.raises(TypeError):
        WorkloadRecipe()

    class Dummy(WorkloadRecipe):
        kind = "dummy_search"
        task_module = "flashml_workloads.sklearn_trial"

        def expand(self, job_id, spec):
            return []

        def validate_output(self, metrics):
            if "accuracy_mean" not in metrics:
                raise ValueError("missing accuracy_mean")

    recipe = Dummy()
    assert recipe.checkpointable() is False  # default: stateless tasks
    with pytest.raises(ValueError):
        recipe.validate_output({})


# ---------------------------------------------------------------------------
# scheduler: placement — the FIFO concrete must equal today's behavior
# ---------------------------------------------------------------------------


def test_fifo_placement_preserves_order_and_is_total():
    from flashruntime.scheduler import FifoPlacement, PlacementPolicy

    assert issubclass(FifoPlacement, PlacementPolicy)
    policy = FifoPlacement()
    tasks = [_task("t1"), _task("t2"), _task("t3")]
    node = {"node_id": "n1", "capabilities": {}}
    eligible = [t for t in tasks if policy.eligible(t, node)]
    assert eligible == tasks  # FIFO: everything eligible, order untouched
    assert policy.choose(tasks, node) is tasks[0]


def test_placement_policy_is_abstract():
    from flashruntime.scheduler import PlacementPolicy

    with pytest.raises(TypeError):
        PlacementPolicy()


# ---------------------------------------------------------------------------
# checkpoint persistence seam (R1)
# ---------------------------------------------------------------------------


def test_manifest_store_protocol_and_memory_reference_impl():
    from flashruntime.checkpoint.store import InMemoryManifestStore

    store = InMemoryManifestStore()
    manifest = _manifest()
    store.add(manifest)
    assert store.all("j1::t1")[0].manifest_id == manifest.manifest_id
    manifest.checkpoint_duration_s = 1.5
    store.save(manifest)  # persistence hook — no-op in memory
    assert store.all("j1::t1")[0].checkpoint_duration_s == 1.5


# ---------------------------------------------------------------------------
# profiling (planner tier 2)
# ---------------------------------------------------------------------------


def test_profiler_contract_and_result_shape():
    from flashruntime.profiling import Profiler, ProfileResult

    with pytest.raises(TypeError):
        Profiler()

    result = ProfileResult(
        peak_vram_gb=18.2, step_time_s=1.9, measured_steps=20, warmup_steps=3
    )
    assert result.basis == "profiled"  # results are born measured


# ---------------------------------------------------------------------------
# providers (get machines)
# ---------------------------------------------------------------------------


def test_resource_provider_contract():
    from flashruntime.providers import Offer, ResourceProvider

    with pytest.raises(TypeError):
        ResourceProvider()

    offer = Offer(provider="dummy", gpu_type="RTX4090", gpus=2, hourly_usd=0.88)
    assert offer.region is None  # optional


# ---------------------------------------------------------------------------
# flash.run — the planner→execution bridge, honestly unimplemented
# ---------------------------------------------------------------------------


def test_flash_run_raises_not_implemented_with_guidance():
    import flashruntime as flash

    report = flash.plan(
        flash.PlanRequest(
            workload=flash.IndependentTasks(task_count=2, est_minutes_per_task=1),
            resources=flash.Resources(hosts=1, cpu_cores=4),
        )
    )
    with pytest.raises(NotImplementedError, match="JobSpec"):
        flash.run(report.selected)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _plan(**over):
    from flashruntime.protocol.plan_v1alpha1 import StrategyPlan

    base = dict(
        plan_id="pl_test",
        planner_version="0.0.0",
        workload_mode="coordinated_training",
        strategy_family="ddp",
        launcher="torchrun",
        workers=1,
    )
    base.update(over)
    return StrategyPlan(**base)


def _task(task_id):
    from flashruntime.protocol.v1alpha1 import TaskSpec

    return TaskSpec(task_id=task_id, job_id="j1", commit_key=f"j1/{task_id}")


def _manifest():
    from flashruntime.protocol.v1alpha1 import (
        CheckpointManifest,
        CheckpointPart,
        CheckpointValidation,
    )

    return CheckpointManifest(
        manifest_id="ck-test",
        job_id="j1::t1",
        attempt_id="a1",
        step=10,
        storage_prefix="artifact://x/",
        parts=[CheckpointPart(key="w.json", sha256="a" * 64, size_bytes=1)],
        validation=CheckpointValidation.HASH_VERIFIED,
    )
