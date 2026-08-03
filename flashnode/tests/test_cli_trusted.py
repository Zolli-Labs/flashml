"""--runner trusted: the opt-in wiring. The flag is the ONLY path to
unsandboxed_argv_capable=True — discover() must default it False for every
other caller (security fields fail closed)."""

from flashnode.inventory.capabilities import discover


def test_discover_defaults_unsandboxed_argv_capable_false():
    reg = discover("n-1", kubernetes_node="")
    assert reg.unsandboxed_argv_capable is False


def test_discover_can_opt_in():
    reg = discover("n-1", kubernetes_node="", unsandboxed_argv_capable=True)
    assert reg.unsandboxed_argv_capable is True
    assert reg.argv_capable is False  # trusted is NOT the containerised contract


def test_trusted_runner_selected_for_runner_trusted(monkeypatch, tmp_path, capsys):
    """--runner trusted must:
      - build a TrustedArgvRunner (passed to ExecutorLoop)
      - register with unsandboxed_argv_capable=True, argv_capable=False
      - NEVER invoke the docker doctor gate (trusted is outside the
        ("docker", "argv") branch, same as subprocess)
      - print a plain-words warning that pool jobs run unsandboxed

    Mirrors test_agent.py::test_work_cli_sets_module_capable_from_runner_choice.
    """
    import flashnode.inventory.capabilities as capabilities_mod
    from flashnode.agent import cli
    from flashnode.executor import CoordinatorClient, ExecutorLoop
    from flashnode.executor.trusted_runner import TrustedArgvRunner

    monkeypatch.setenv("FLASHNODE_STATE_DIR", str(tmp_path))

    # The docker doctor gate must not run for --runner trusted. Rather than
    # counting calls, make one explode: any invocation fails the test loudly
    # and precisely where it happened.
    def _boom(**kw):
        raise AssertionError("doctor.run_checks must not run for --runner trusted")

    monkeypatch.setattr("flashnode.doctor.run_checks", _boom)

    captured = {}
    real_discover = capabilities_mod.discover

    def spy_discover(*args, **kwargs):
        reg = real_discover(*args, **kwargs)
        captured["registration"] = reg
        return reg

    monkeypatch.setattr(capabilities_mod, "discover", spy_discover)
    monkeypatch.setattr(CoordinatorClient, "register", lambda self, reg: None)

    real_init = ExecutorLoop.__init__

    def spy_init(self, *args, **kwargs):
        captured["runner"] = kwargs.get("runner")
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ExecutorLoop, "__init__", spy_init)
    monkeypatch.setattr(ExecutorLoop, "run", lambda self, max_tasks=None: 0)

    rc = cli.main(["work", "--runner", "trusted", "--coordinator", "http://localhost:1"])

    assert rc == 0
    assert isinstance(captured["runner"], TrustedArgvRunner)

    reg = captured["registration"]
    assert reg.unsandboxed_argv_capable is True
    assert reg.argv_capable is False

    out = capsys.readouterr().out
    assert "unsandboxed" in out.lower()
