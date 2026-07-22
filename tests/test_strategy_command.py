from __future__ import annotations


def _wl(**over):
    from flashruntime.workloads.command import CommandWorkload

    base = dict(command="python train.py")
    base.update(over)
    return CommandWorkload(**base)


def test_compile_is_pure_and_deterministic():
    from flashruntime.strategies.command import compile_workload

    wl = _wl(command="python train.py --lr {lr}", env={"TAG": "run-{lr}"})
    a = compile_workload(wl, {"lr": 0.1})
    b = compile_workload(wl, {"lr": 0.1})
    assert a == b
    assert a.argv == ["python", "train.py", "--lr", "0.1"]
    assert a.env == {"TAG": "run-0.1"}


def test_workdir_hint_carries_source_path():
    from flashruntime.strategies.command import compile_workload
    from flashruntime.workloads.command import Source

    spec = compile_workload(_wl(source=Source(path="/home/me/proj")))
    assert spec.workdir_hint == "/home/me/proj"


def test_torchrun_world_size_extracted():
    from flashruntime.strategies.command import compile_workload

    spec = compile_workload(_wl(command="torchrun --nproc-per-node=4 --standalone train.py"))
    assert spec.world_size == 4
    assert spec.argv[0] == "torchrun"
