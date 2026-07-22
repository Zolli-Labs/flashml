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
    assert float(m2.weight[0, 0]) == pytest.approx(3.14)
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


def test_log_metrics_appends_jsonl_and_never_raises(ft, tmp_path, monkeypatch):
    ft.log_metrics({"loss": 1.0})
    ft.log_metrics({"loss": 0.5})
    lines = (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(l)["loss"] for l in lines] == [1.0, 0.5]
    # unwritable target must not kill training
    monkeypatch.setenv("FLASHML_OUTPUT_DIR", "/dev/null/nope")
    ft.log_metrics({"loss": 0.1})  # must not raise
