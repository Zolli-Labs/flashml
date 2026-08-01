"""An operator who exposes the coordinator and forgets to configure tokens
must get a refusal to boot, not a silently open door."""

import pytest

from flashruntime.service.app import create_app


def test_require_node_auth_without_tokens_refuses_to_start(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHML_REQUIRE_NODE_AUTH", "1")
    monkeypatch.delenv("FLASHML_NODE_TOKENS", raising=False)
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "l.db"))
    monkeypatch.setenv("FLASHML_LOCAL_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    with pytest.raises(RuntimeError, match="FLASHML_NODE_TOKENS"):
        create_app()


def test_require_node_auth_with_tokens_starts(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHML_REQUIRE_NODE_AUTH", "1")
    monkeypatch.setenv("FLASHML_NODE_TOKENS", "node-a:tok-a")
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "l.db"))
    monkeypatch.setenv("FLASHML_LOCAL_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    assert create_app() is not None


def test_default_startup_is_open_and_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("FLASHML_REQUIRE_NODE_AUTH", raising=False)
    monkeypatch.delenv("FLASHML_NODE_TOKENS", raising=False)
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "l.db"))
    monkeypatch.setenv("FLASHML_LOCAL_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    assert create_app() is not None
