"""The `nvidia-smi` probe.

Every test drives a stub runner, so the whole suite runs on a machine with no
driver — the same reason `doctor.py` parameterises its subprocess calls. The
probe's contract is narrow and absolute: it NEVER raises and it NEVER guesses
an index. A `GpuInfo` without a valid `index` cannot exist, and inventing one
would send a malformed device up in `NodeRegistration` — a 422 that takes the
whole host offline rather than costing it one GPU.
"""

from __future__ import annotations

import subprocess

import pytest

from flashnode.inventory.gpu import probe_gpus

TWO_GPUS = (
    b"0, NVIDIA GeForce RTX 4090, 24564, 550.54.15, 8.9\n"
    b"1, NVIDIA GeForce RTX 4090, 24564, 550.54.15, 8.9\n"
)


def _runner(stdout: bytes = b"", returncode: int = 0, stderr: bytes = b""):
    """A stub `subprocess.run` that records every call it was given."""
    calls: list[tuple[tuple, dict]] = []

    def run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(list(argv), returncode, stdout, stderr)

    run.calls = calls
    return run


def test_two_gpus_parse_into_typed_gpuinfo():
    gpus = probe_gpus(run=_runner(TWO_GPUS))
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert gpus[0].memory_total_mb == 24564
    assert gpus[0].driver_version == "550.54.15"
    assert gpus[0].compute_capability == "8.9"


def test_the_query_is_the_documented_one_and_is_bounded_in_time():
    run = _runner(TWO_GPUS)
    probe_gpus(run=run)
    argv, kwargs = run.calls[0]
    assert argv[0] == "nvidia-smi"
    assert "--query-gpu=index,name,memory.total,driver_version,compute_cap" in argv
    assert "--format=csv,noheader,nounits" in argv
    # A hung nvidia-smi must not hang registration.
    assert kwargs["timeout"] == 5


def test_no_driver_yields_no_gpus():
    """The overwhelmingly common case: `nvidia-smi` is not installed."""

    def run(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'nvidia-smi'")

    assert probe_gpus(run=run) == []


def test_a_non_zero_exit_yields_no_gpus():
    assert probe_gpus(run=_runner(b"", returncode=9, stderr=b"No devices were found")) == []


def test_a_timeout_yields_no_gpus():
    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=5)

    assert probe_gpus(run=run) == []


def test_unparseable_output_yields_no_gpus():
    assert probe_gpus(run=_runner(b"Unable to determine the device handle\n")) == []


def test_empty_output_yields_no_gpus():
    assert probe_gpus(run=_runner(b"\n  \n")) == []


def test_an_index_is_never_guessed_and_never_defaulted():
    """A row whose index will not parse is DROPPED, never emitted with a
    guessed 0. One unreadable device must cost one device, not the host."""
    out = (
        b"0, NVIDIA A100, 40960, 550.54.15, 8.0\n"
        b", NVIDIA A100, 40960, 550.54.15, 8.0\n"       # index missing
        b"GPU 2, NVIDIA A100, 40960, 550.54.15, 8.0\n"  # index not a number
        b"-1, NVIDIA A100, 40960, 550.54.15, 8.0\n"     # nonsense index
        b"3, NVIDIA A100, 40960, 550.54.15, 8.0\n"
    )
    assert [g.index for g in probe_gpus(run=_runner(out))] == [0, 3]


def test_a_short_row_keeps_the_gpu_it_can_read():
    """Older nvidia-smi builds do not report compute_cap. That costs the
    field, not the device."""
    gpus = probe_gpus(run=_runner(b"0, NVIDIA GeForce GTX 1080, 8119, 470.223.02\n"))
    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce GTX 1080"
    assert gpus[0].memory_total_mb == 8119
    assert gpus[0].driver_version == "470.223.02"
    assert gpus[0].compute_capability == ""


def test_index_alone_is_enough():
    gpus = probe_gpus(run=_runner(b"0\n"))
    assert len(gpus) == 1
    assert gpus[0].index == 0
    assert gpus[0].name == ""
    assert gpus[0].memory_total_mb is None


def test_driver_placeholders_become_empty_rather_than_the_literal_text():
    out = b"0, NVIDIA A100, [N/A], [Not Supported], N/A\n"
    gpu = probe_gpus(run=_runner(out))[0]
    assert gpu.memory_total_mb is None
    assert gpu.driver_version == ""
    assert gpu.compute_capability == ""


def test_an_old_driver_that_rejects_compute_cap_still_reports_its_gpus():
    """`compute_cap` postdates plenty of drivers still in the wild, and
    nvidia-smi fails the WHOLE query on an unknown field rather than
    omitting it. Retry once without it instead of telling a real GPU host
    they have none."""
    responses = [
        subprocess.CompletedProcess(
            [], 6, b"", b'Field "compute_cap" is not a valid field to query.'
        ),
        subprocess.CompletedProcess(
            [], 0, b"0, Tesla K80, 11441, 418.87.00\n", b""
        ),
    ]
    seen = []

    def run(argv, **kwargs):
        seen.append(tuple(argv))
        return responses[min(len(seen) - 1, len(responses) - 1)]

    gpus = probe_gpus(run=run)
    assert [g.index for g in gpus] == [0]
    assert gpus[0].compute_capability == ""
    assert "--query-gpu=index,name,memory.total,driver_version" in seen[1]


@pytest.mark.parametrize(
    "boom",
    [
        lambda argv, **kw: (_ for _ in ()).throw(RuntimeError("kaboom")),
        lambda argv, **kw: (_ for _ in ()).throw(OSError("permission denied")),
        lambda argv, **kw: (_ for _ in ()).throw(MemoryError()),
        lambda argv, **kw: None,                       # a runner returning junk
        lambda argv, **kw: object(),
    ],
)
def test_the_probe_never_raises(boom):
    """Registration calls this. Anything it raises takes the agent down on a
    machine that was only ever going to run CPU work."""
    assert probe_gpus(run=boom) == []


def test_the_runner_defaults_to_the_real_subprocess_at_call_time(monkeypatch):
    """Resolved when called, never captured as a default at import — the
    mistake `doctor.check_cli_on_path` documents at length."""
    calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda argv, **kw: calls.append(argv)
        or subprocess.CompletedProcess(argv, 0, TWO_GPUS, b""),
    )
    assert len(probe_gpus()) == 2
    assert calls and calls[0][0] == "nvidia-smi"
