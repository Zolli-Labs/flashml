"""Strategy-planner quickstart: three workload kinds, no cluster required.

    python examples/plan_quickstart.py
"""

import flashruntime as flash

# 1. Transformer fine-tuning: the flagship case. The planner decides
#    between single-GPU, DDP, FSDP2, QLoRA variants, and CPU offload —
#    and shows the arithmetic for every candidate.
lora = flash.PlanRequest(
    workload=flash.TransformerFineTune(
        model="Qwen/Qwen2.5-7B", method="lora", lora_rank=16, train_tokens_m=25
    ),
    resources=flash.Resources(
        gpus=4, gpu_type="RTX4090", cpu_ram_gb=128, hourly_cost_usd_per_gpu=0.44
    ),
    objective=flash.Objective(mode="balanced", max_cost_usd=20, deadline_minutes=240),
)

# 2. Hyperparameter search: independent tasks → the lease runtime (Mode A).
sweep = flash.PlanRequest(
    workload=flash.IndependentTasks(
        task_kind="hyperparameter_search", task_count=24, est_minutes_per_task=12
    ),
    resources=flash.Resources(hosts=3, cpu_cores=8),
)

# 3. Classical ML: local process, unless the dataset outgrows RAM.
kmeans = flash.PlanRequest(
    workload=flash.ClassicalML(
        library="sklearn", algorithm="kmeans", dataset_mb=800, supports_partial_fit=True
    ),
    resources=flash.Resources(cpu_ram_gb=16),
)

for title, request in [("LoRA fine-tune", lora), ("HPO sweep", sweep), ("K-means", kmeans)]:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(flash.render(flash.plan(request)))
