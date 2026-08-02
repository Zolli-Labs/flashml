import pytest

from flashruntime.service.auth import (
    AuthConfigError,
    OpenAuthenticator,
    StaticTokenAuthenticator,
    authenticator_from_env,
)


def test_open_authenticator_is_not_enforcing():
    a = OpenAuthenticator()
    assert a.enforcing is False
    assert a.authenticate("anything") is None
    assert a.authenticate(None) is None


def test_static_authenticator_maps_token_to_node():
    a = StaticTokenAuthenticator({"tok-a": "node-a", "tok-b": "node-b"})
    assert a.enforcing is True
    assert a.authenticate("tok-a") == "node-a"
    assert a.authenticate("tok-b") == "node-b"


def test_static_authenticator_denies_unknown_and_missing_tokens():
    a = StaticTokenAuthenticator({"tok-a": "node-a"})
    assert a.authenticate("nope") is None
    assert a.authenticate("") is None
    assert a.authenticate(None) is None


def test_static_authenticator_rejects_an_empty_token_at_construction():
    """An empty token would authenticate every caller sending no token."""
    with pytest.raises(AuthConfigError, match="empty token"):
        StaticTokenAuthenticator({"": "node-a"})


def test_token_comparison_is_constant_time():
    """Guard against a timing oracle on token contents. We cannot measure
    timing reliably in a unit test, so we pin the implementation choice."""
    import inspect

    from flashruntime.service import auth

    assert "compare_digest" in inspect.getsource(auth.StaticTokenAuthenticator)


def test_env_without_tokens_yields_open():
    assert authenticator_from_env({}).enforcing is False


def test_env_with_tokens_yields_enforcing():
    a = authenticator_from_env({"FLASHML_NODE_TOKENS": "node-a:tok-a,node-b:tok-b"})
    assert a.enforcing is True
    assert a.authenticate("tok-a") == "node-a"


def test_env_tolerates_whitespace_and_trailing_commas():
    a = authenticator_from_env({"FLASHML_NODE_TOKENS": " node-a:tok-a , node-b:tok-b ,"})
    assert a.authenticate("tok-a") == "node-a"
    assert a.authenticate("tok-b") == "node-b"


def test_env_rejects_a_malformed_pair():
    with pytest.raises(AuthConfigError, match="node_id:token"):
        authenticator_from_env({"FLASHML_NODE_TOKENS": "garbage"})


def test_env_rejects_a_duplicate_token_across_nodes():
    """Two nodes sharing a token makes attribution — and revocation — a lie."""
    with pytest.raises(AuthConfigError, match="duplicate token"):
        authenticator_from_env({"FLASHML_NODE_TOKENS": "node-a:same,node-b:same"})


def test_open_authenticator_enforcing_is_read_only():
    """The enforcing property must not be assignable on a live instance."""
    a = OpenAuthenticator()
    with pytest.raises(AttributeError):
        a.enforcing = True


def test_static_authenticator_enforcing_is_read_only():
    """The enforcing property must not be assignable on a live instance."""
    a = StaticTokenAuthenticator({"tok-a": "node-a"})
    with pytest.raises(AttributeError):
        a.enforcing = False


# -- 4b-2: a non-ASCII token must deny, not raise --------------------------


def test_a_non_ascii_token_is_denied_not_a_crash():
    """`hmac.compare_digest` raises TypeError on a non-ASCII str. Unhandled,
    that is an unauthenticated remote 500 — and the 500-vs-401 difference is
    an oracle separating "malformed" from "merely wrong"."""
    a = StaticTokenAuthenticator({"tok-a": "node-a"})
    assert a.authenticate("tök-a") is None
    assert a.authenticate("日本語") is None
    assert a.is_operator("tök-a") is False


def test_a_non_str_token_is_denied():
    """Fail closed on type confusion rather than trusting the caller."""
    a = StaticTokenAuthenticator({"tok-a": "node-a"})
    assert a.authenticate(b"tok-a") is None  # type: ignore[arg-type]
    assert a.authenticate(0) is None  # type: ignore[arg-type]
    assert a.is_operator(b"tok-a") is False  # type: ignore[arg-type]


# -- 4b-4: operator tokens -------------------------------------------------


def test_is_operator_is_true_only_for_operator_tokens():
    a = StaticTokenAuthenticator({"tok-a": "node-a"}, {"op-tok": "driver"})
    assert a.is_operator("op-tok") is True
    assert a.is_operator("tok-a") is False
    assert a.is_operator("nope") is False
    assert a.is_operator("") is False
    assert a.is_operator(None) is False


def test_an_operator_token_is_not_a_node():
    """An operator holds no lease, so it must never be able to claim,
    complete, or fail one — it is not a node identity at all."""
    a = StaticTokenAuthenticator({"tok-a": "node-a"}, {"op-tok": "driver"})
    assert a.authenticate("op-tok") is None


def test_an_open_authenticator_has_no_operators():
    assert OpenAuthenticator().is_operator("anything") is False


def test_operator_token_colliding_with_a_node_token_is_rejected():
    """One string that is both lease-scoped and unscoped is an escalation."""
    with pytest.raises(AuthConfigError, match="collides"):
        StaticTokenAuthenticator({"same": "node-a"}, {"same": "driver"})


def test_an_empty_operator_token_is_rejected_at_construction():
    with pytest.raises(AuthConfigError, match="empty token"):
        StaticTokenAuthenticator({"tok-a": "node-a"}, {"": "driver"})


def test_env_reads_operator_tokens():
    a = authenticator_from_env({"FLASHML_NODE_TOKENS": "node-a:tok-a",
                                "FLASHML_OPERATOR_TOKENS": "driver:op-tok"})
    assert a.enforcing is True
    assert a.authenticate("tok-a") == "node-a"
    assert a.is_operator("op-tok") is True
    assert a.is_operator("tok-a") is False


def test_env_rejects_an_operator_token_colliding_with_a_node_token():
    with pytest.raises(AuthConfigError, match="collides"):
        authenticator_from_env({"FLASHML_NODE_TOKENS": "node-a:same",
                                "FLASHML_OPERATOR_TOKENS": "driver:same"})


def test_env_rejects_a_malformed_operator_pair():
    with pytest.raises(AuthConfigError, match="name:token"):
        authenticator_from_env({"FLASHML_OPERATOR_TOKENS": "garbage"})


def test_operator_tokens_alone_still_enforce():
    """Fail closed: ignoring them would leave a coordinator its operator
    believes is credentialed completely open."""
    a = authenticator_from_env({"FLASHML_OPERATOR_TOKENS": "driver:op-tok"})
    assert a.enforcing is True
    assert a.is_operator("op-tok") is True
    assert a.authenticate("op-tok") is None
