"""Contract tests for flashnode's future-work interfaces (TDD-first).

Pins: importability from the final homes, ABC enforcement, conforming
dummy implementations, and — for HostPolicy, which ships concrete —
conservative defaults and a load/roundtrip that fails closed on garbage.
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# benchmark: admission probes
# ---------------------------------------------------------------------------


def test_admission_probe_is_abstract_and_runnable():
    from flashnode.benchmark import AdmissionProbe, ProbeResult, run_admission

    with pytest.raises(TypeError):
        AdmissionProbe()

    class CpuDummy(AdmissionProbe):
        name = "cpu_hash_mbps"

        def run(self, budget_seconds: float) -> ProbeResult:
            return ProbeResult(name=self.name, value=123.4, unit="MB/s", duration_s=0.01)

    results = run_admission([CpuDummy()], total_budget_seconds=5.0)
    assert results["cpu_hash_mbps"].value == 123.4


def test_run_admission_isolates_probe_failures():
    """One broken probe must not sink the node's whole admission — the
    failure is recorded as a result with value None, not raised."""
    from flashnode.benchmark import AdmissionProbe, ProbeResult, run_admission

    class Boom(AdmissionProbe):
        name = "boom"

        def run(self, budget_seconds: float) -> ProbeResult:
            raise RuntimeError("probe exploded")

    results = run_admission([Boom()], total_budget_seconds=1.0)
    assert results["boom"].value is None
    assert "exploded" in results["boom"].detail


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------


def test_telemetry_collector_contract():
    from flashnode.telemetry import TelemetryCollector, TelemetrySample

    with pytest.raises(TypeError):
        TelemetryCollector()

    class Dummy(TelemetryCollector):
        def sample(self) -> TelemetrySample:
            return TelemetrySample(cpu_percent=12.5, memory_used_bytes=1024)

    s = Dummy().sample()
    assert s.gpus == []  # optional, defaults empty
    assert s.timestamp is not None


# ---------------------------------------------------------------------------
# config: host-owner policy (ships concrete — safety must be default)
# ---------------------------------------------------------------------------


def test_host_policy_defaults_are_conservative():
    from flashnode.config import HostPolicy

    policy = HostPolicy()
    assert policy.max_concurrent_tasks == 1
    assert policy.allow_network_tasks is False
    assert policy.allowed_modules is None  # None = defer to built-in allowlist
    assert policy.workdir_quota_gb > 0


def test_host_policy_loads_from_state_dir_and_fails_closed(tmp_path):
    from flashnode.config import HostPolicy, load_host_policy

    # no file → defaults
    assert load_host_policy(tmp_path) == HostPolicy()

    # valid file → honored
    (tmp_path / "policy.json").write_text(json.dumps({"max_concurrent_tasks": 2}))
    assert load_host_policy(tmp_path).max_concurrent_tasks == 2

    # garbage → refuse loudly (never silently fall back to permissive)
    (tmp_path / "policy.json").write_text("{not json")
    with pytest.raises(ValueError, match="policy.json"):
        load_host_policy(tmp_path)
