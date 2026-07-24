"""flashruntime.torch: single-process behavior + checkpoint/resume. The
2-process gloo path is exercised end-to-end in test_examples_e2e.py."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture()
def ft(monkeypatch, tmp_path):
    import flashruntime.torch as ft_mod

    monkeypatch.setenv("FLASHML_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("FLASHML_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(ft_mod, "_restored_step", 0)
    monkeypatch.setattr(ft_mod, "_last_beat", 0.0)
    monkeypatch.setattr(ft_mod, "_last_step", None)
    return ft_mod


def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


def test_prepare_is_noop_single_process(ft):
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = object()  # must be passed through untouched
    m2, o2, l2 = ft.prepare(model, opt, loader)
    assert m2 is model and o2 is opt and l2 is loader
    assert ft.world_size() == 1 and ft.rank() == 0 and ft.is_main()
    assert ft.start_step() == 0


def test_checkpoint_every_gating(ft, tmp_path):
    model = _model()
    ft.checkpoint(model, step=7, every=5)
    assert not list((tmp_path / "ckpt").glob("step-*"))
    ft.checkpoint(model, step=10, every=5)
    assert (tmp_path / "ckpt" / "step-000010" / "manifest.json").is_file()


def test_device_and_backend_accessors_report_single_process_defaults(ft):
    # read-only launch-fact accessors (T12, reviewer-blessed): before/without
    # a distributed prepare(), a CPU box reports device "cpu" and no backend
    model = _model()
    ft.prepare(model, None, None)
    assert ft.device() == "cpu"
    assert ft.backend() is None


def test_checkpoint_every_zero_is_a_noop_not_a_crash(ft, tmp_path):
    # every<=0 means "no periodic checkpointing" — a fault-tolerance helper
    # must never itself crash training over a config value (used to raise
    # ZeroDivisionError; found by the benchmark suite)
    model = _model()
    ft.checkpoint(model, step=10, every=0)
    ft.checkpoint(model, step=10, every=-3)
    assert not list((tmp_path / "ckpt").glob("step-*"))


def test_checkpoint_then_resume_restores_weights_and_step(ft):
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.no_grad():
        model.weight.fill_(3.14)
    ft.checkpoint(model, opt, step=5)

    import flashruntime.torch as ft_mod

    ft_mod._restored_step = 0
    fresh = torch.nn.Linear(4, 2)
    fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.1)
    m2, _, _ = ft.prepare(fresh, fresh_opt, None)
    assert float(m2.weight.detach()[0, 0]) == pytest.approx(3.14)
    assert ft.start_step() == 5


def test_corrupted_checkpoint_is_never_restored(ft, tmp_path):
    model = _model()
    ft.checkpoint(model, step=5)
    ft.checkpoint(model, step=10)
    (tmp_path / "ckpt" / "step-000010" / "model.pt").write_bytes(b"garbage")

    import flashruntime.torch as ft_mod

    ft_mod._restored_step = 0
    ft.prepare(torch.nn.Linear(4, 2), None, None)
    assert ft.start_step() == 5  # fell back to the older VALID manifest


def test_resolve_device_is_pure_and_explicit():
    from flashruntime.torch import _resolve_device

    assert _resolve_device(world_size=1, cuda_available=False, local_rank=0) == "cpu"
    assert _resolve_device(world_size=2, cuda_available=False, local_rank=1) == "cpu"
    assert _resolve_device(world_size=1, cuda_available=True, local_rank=0) == "cuda:0"
    assert _resolve_device(world_size=4, cuda_available=True, local_rank=3) == "cuda:3"


def test_checkpoint_state_dicts_are_cpu(ft, tmp_path):
    # saved tensors must be CPU regardless of training device, so manifests
    # stay topology- and device-agnostic (restore maps them wherever needed)
    model = _model()
    ft.checkpoint(model, step=5)
    state = torch.load(tmp_path / "ckpt" / "step-000005" / "model.pt", map_location=None)
    assert all(t.device.type == "cpu" for t in state.values())


def test_log_metrics_appends_jsonl_and_never_raises(ft, tmp_path, monkeypatch):
    ft.log_metrics({"loss": 1.0})
    ft.log_metrics({"loss": 0.5})
    lines = (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(l)["loss"] for l in lines] == [1.0, 0.5]
    # unwritable target must not kill training
    monkeypatch.setenv("FLASHML_OUTPUT_DIR", "/dev/null/nope")
    ft.log_metrics({"loss": 0.1})  # must not raise


# ---- per-rank heartbeats (run-monitor telemetry) ---------------------------
# Every rank mirrors its identity + progress to ranks/rank-N.json so the run
# viewer can draw machine → worker → rank with live PIDs and steps. The write
# is best-effort and throttled; it must never be able to crash training.


def _beat_path(tmp_path):
    return tmp_path / "out" / "ranks" / "rank-0.json"


def test_heartbeat_written_and_shaped(ft, tmp_path):
    import os

    ft._write_heartbeat(step=7, force=True)
    beat = json.loads(_beat_path(tmp_path).read_text())
    assert beat["rank"] == 0 and beat["local_rank"] == 0
    assert beat["pid"] == os.getpid()
    assert beat["world_size"] == 1
    assert beat["step"] == 7
    assert beat["device"] == "cpu"  # single-process default before prepare()
    assert isinstance(beat["ts"], float)


def test_heartbeat_throttles_but_force_overrides(ft, tmp_path):
    ft._write_heartbeat(step=1, force=True)
    ft._write_heartbeat(step=2)  # inside the 1 s window — must be skipped
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 1
    ft._write_heartbeat(step=3, force=True)  # force bypasses the throttle
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 3


def test_heartbeat_remembers_last_step_when_not_given(ft, tmp_path):
    ft._write_heartbeat(step=42, force=True)
    ft._write_heartbeat(force=True)  # no step: reuse the last known one
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 42


def test_heartbeat_never_raises_on_unwritable_dir(ft, tmp_path, monkeypatch):
    # FLASHML_OUTPUT_DIR pointing at a *file* makes ranks/ uncreatable
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("FLASHML_OUTPUT_DIR", str(blocker))
    ft._write_heartbeat(step=1, force=True)  # must not raise


def test_log_metrics_refreshes_heartbeat(ft, tmp_path):
    ft.log_metrics({"loss": 0.5, "step": 9})
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 9


def test_gated_checkpoint_still_beats(ft, tmp_path):
    # every-gated checkpoint calls return before touching torch, but the
    # heartbeat (progress signal) must still fire — the loop calls this
    # every iteration and the viewer wants the live step.
    ft.checkpoint(None, step=7, every=5)  # gated: no manifest written
    assert not list((tmp_path / "ckpt").glob("step-*"))
    assert json.loads(_beat_path(tmp_path).read_text())["step"] == 7


def test_prepare_writes_initial_heartbeat(ft, tmp_path):
    model = _model()
    ft.prepare(model, None, None)
    beat = json.loads(_beat_path(tmp_path).read_text())
    assert beat["step"] == 0 and beat["device"] == "cpu"
