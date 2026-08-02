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


def test_torchrun_world_size_space_separated():
    from flashruntime.strategies.command import compile_workload

    spec = compile_workload(_wl(command="torchrun --nproc-per-node 4 --standalone train.py"))
    assert spec.world_size == 4
    assert spec.argv[0] == "torchrun"


def test_torchrun_world_size_underscore_spelling():
    from flashruntime.strategies.command import compile_workload

    eq = compile_workload(_wl(command="torchrun --nproc_per_node=4 train.py"))
    assert eq.world_size == 4
    sp = compile_workload(_wl(command="torchrun --nproc_per_node 4 train.py"))
    assert sp.world_size == 4


def test_torchrun_non_integer_world_size_does_not_crash():
    from flashruntime.strategies.command import compile_workload

    spec = compile_workload(_wl(command="torchrun --nproc-per-node=gpu train.py"))
    assert spec.world_size == 1
    assert any("world_size unresolved" in n for n in spec.notes)
    assert any("--nproc-per-node=gpu" in n for n in spec.notes)


def test_non_torchrun_command_defaults_world_size_one():
    from flashruntime.strategies.command import compile_workload

    spec = compile_workload(_wl(command="python train.py --nproc-per-node=8"))
    assert spec.world_size == 1


def test_env_passthrough_untouched_when_params_none():
    from flashruntime.strategies.command import compile_workload

    wl = _wl(env={"CFG": "{unresolved}", "TAG": "static"})
    spec = compile_workload(wl)
    assert spec.env == {"CFG": "{unresolved}", "TAG": "static"}
