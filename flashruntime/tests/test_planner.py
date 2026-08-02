"""Planner tests: memory arithmetic, candidate policy, selection, honesty.

The numeric assertions pin the *component model* (16 bytes/param full
fine-tuning, LoRA's frozen-weights dominance, QLoRA's ~0.55 B/param frozen
weights, ZeRO-3 sharding + offload) — if a formula changes, these fail
loudly and the change must be justified against the evaluation doc.
"""

from __future__ import annotations

import flashruntime as flash
from flashruntime.planner import plan, render
from flashruntime.service import cli


def _req(workload, gpus=4, gpu_type="RTX4090", **kw):
    resources = flash.Resources(
        gpus=gpus,
        gpu_type=gpu_type,
        cpu_ram_gb=kw.pop("cpu_ram_gb", 128),
        hosts=kw.pop("hosts", 1),
        interconnect=kw.pop("interconnect", "same_host_pcie"),
        hourly_cost_usd_per_gpu=kw.pop("hourly_cost_usd_per_gpu", None),
    )
    return flash.PlanRequest(
        workload=workload, resources=resources, objective=flash.Objective(**kw)
    )


def _verdicts(report, name=None, family=None):
    out = report.candidates
    if name:
        out = [c for c in out if c.name == name]
    if family:
        out = [c for c in out if c.strategy_family == family]
    return out


# ---------------------------------------------------------------------------
# Memory arithmetic
# ---------------------------------------------------------------------------


def test_full_finetune_7b_never_fits_24gb_unsharded():
    """16 B/param ⇒ 7.6B params ≈ 122 GB training state: single-GPU and DDP
    on 24 GB must be infeasible; only offloaded sharding can survive."""
    report = plan(
        _req(flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="full"))
    )
    for c in _verdicts(report, family="single_gpu") + _verdicts(report, family="ddp"):
        assert c.status == "infeasible", c.name
        assert c.memory.total_gb > 100
    assert report.selected is not None
    assert report.selected.strategy_family == "zero3_cpu_offload"
    assert report.selected.memory.cpu_offload_gb > 10  # optimizer state in host RAM


def test_full_finetune_offload_respects_host_ram():
    """A planner that only checks VRAM kills hosts: with tiny host RAM the
    offload candidates must die too, leaving no valid strategy."""
    report = plan(
        _req(flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="full"), cpu_ram_gb=8)
    )
    assert report.selected is None
    assert "VRAM" in (report.no_valid_strategy_hint or "")


def test_lora_7b_fits_where_full_cannot():
    """LoRA freezes the 16-B/param state down to the adapters: 7B bf16
    weights ≈ 15 GB dominate, and a single 24 GB GPU becomes viable."""
    report = plan(_req(flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="lora")))
    single = _verdicts(report, family="single_gpu")
    assert any(c.status in ("feasible", "selected") for c in single if not c.name.startswith("qlora"))
    lora_single = next(c for c in single if not c.name.startswith("qlora"))
    assert 14 < lora_single.memory.weights_gb < 17
    assert lora_single.memory.optimizer_gb < 1  # adapters only


def test_qlora_shrinks_weights_about_4x():
    report = plan(_req(flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="qlora")))
    q = _verdicts(report, family="single_gpu")[0]
    assert 3.5 < q.memory.weights_gb < 6  # ~0.55 B/param + bf16 adapters


def test_activation_checkpointing_auto_enabled_to_fit():
    """When act=False overflows but act=True fits, the planner flips it on,
    renames the candidate, and says why."""
    w = flash.TransformerFineTune(
        model="Qwen/Qwen2.5-7B", method="lora", seq_len=8192, micro_batch_per_gpu=4
    )
    report = plan(_req(w, gpus=1))
    ackpt = [c for c in report.candidates if c.name.endswith("+ackpt")]
    assert ackpt, "expected an auto-activation-checkpointing candidate"
    assert any("activation checkpointing enabled to fit" in r for r in ackpt[0].reasons)


# ---------------------------------------------------------------------------
# Communication policy
# ---------------------------------------------------------------------------


def test_wan_kills_full_finetune_ddp_but_not_single_gpu():
    """Full fine-tuning all-reduces the full gradient every step — WAN links
    must reject it; the single-GPU candidate is unaffected."""
    w = flash.TransformerFineTune(model="Llama-3.2-1B", method="full", parameters_b=1.24)
    report = plan(_req(w, gpus=2, gpu_type="A100-80GB", hosts=2, interconnect="wan"))
    ddp = _verdicts(report, family="ddp")
    assert ddp and all(c.status == "infeasible" for c in ddp)
    assert any("communication dominates" in r for c in ddp for r in c.reasons)
    assert report.selected is not None and report.selected.workers == 1


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------


def _lora_priced(**kw):
    return _req(
        flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="lora", train_tokens_m=25),
        hourly_cost_usd_per_gpu=0.44,
        **kw,
    )


def test_reliable_mode_prefers_fewest_moving_parts():
    report = plan(_lora_priced(mode="reliable"))
    assert report.selected.workers == 1
    assert report.selected.offload == "none"


def test_fastest_mode_prefers_largest_feasible_world():
    report = plan(_lora_priced(mode="fastest"))
    assert report.selected.workers == 4


def test_deadline_is_a_hard_gate_with_reason():
    report = plan(_lora_priced(mode="balanced", deadline_minutes=10))
    assert report.selected is None
    rejected = [c for c in report.candidates if c.status == "rejected_policy"]
    assert rejected and any("deadline" in r for c in rejected for r in c.reasons)


def test_plan_is_deterministic():
    a = plan(_lora_priced(mode="balanced"))
    b = plan(_lora_priced(mode="balanced"))
    assert a.model_dump() == b.model_dump()
    assert a.request_digest == b.request_digest


# ---------------------------------------------------------------------------
# Other workload kinds
# ---------------------------------------------------------------------------


def test_independent_tasks_get_lease_runtime_and_wave_math():
    report = plan(
        flash.PlanRequest(
            workload=flash.IndependentTasks(task_count=24, est_minutes_per_task=12),
            resources=flash.Resources(hosts=3, cpu_cores=8),
        )
    )
    sel = report.selected
    assert sel.strategy_family == "lease_tasks"
    assert sel.launcher == "flashruntime-leases"
    assert sel.workers == 6  # 3 hosts × (8 cores // 4)
    assert sel.est_time_min.value == 48.0  # 4 waves × 12 min
    assert any(ref.name == "flashruntime leases" for ref in sel.libraries)


def test_classical_ml_local_until_ram_runs_out():
    small = plan(
        flash.PlanRequest(
            workload=flash.ClassicalML(algorithm="kmeans", dataset_mb=800, supports_partial_fit=True),
            resources=flash.Resources(cpu_ram_gb=16),
        )
    )
    assert small.selected.strategy_family == "local_process"
    huge = plan(
        flash.PlanRequest(
            workload=flash.ClassicalML(algorithm="kmeans", dataset_mb=60000, supports_partial_fit=True),
            resources=flash.Resources(cpu_ram_gb=16),
        )
    )
    assert huge.selected.strategy_family == "sharded_partial_fit"
    local = next(c for c in huge.candidates if c.strategy_family == "local_process")
    assert local.status == "infeasible"


def test_pytorch_generic_training_plans():
    report = plan(
        _req(
            flash.PyTorchTraining(parameters_m=350, precision="fp32", est_train_gpu_hours=6),
            gpus=2,
            hourly_cost_usd_per_gpu=1.0,
        )
    )
    assert report.selected is not None
    assert report.selected.strategy_family in ("single_gpu", "ddp", "fsdp2")


def test_unknown_model_asks_for_parameters():
    report = plan(_req(flash.TransformerFineTune(model="totally/unknown-model")))
    assert report.selected is None
    assert "parameters_b" in (report.no_valid_strategy_hint or "")


# ---------------------------------------------------------------------------
# Honesty and surface contracts
# ---------------------------------------------------------------------------


def test_estimates_carry_static_basis_and_report_round_trips():
    report = plan(_lora_priced(mode="balanced"))
    assert report.selected.est_time_min.basis == "static"
    assert report.selected.memory.basis == "static"
    restored = flash.PlanReport.model_validate(report.model_dump())
    assert restored.selected.plan_id == report.selected.plan_id


def test_render_mentions_rejections_and_libraries():
    text = render(plan(_lora_priced(mode="balanced")))
    assert "SELECTED PLAN" in text
    assert "libraries" in text
    assert "torchrun" in text
    assert "INFEASIBLE" in text or "not chosen" in text


def test_cli_plan_yaml(capsys):
    rc = cli.main(["plan", "examples/plan-qwen7b-lora.yaml"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SELECTED PLAN" in out


def test_cli_plan_json_output(capsys):
    rc = cli.main(["plan", "examples/plan-qwen7b-lora.yaml", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"kind": "PlanReport"' in out
