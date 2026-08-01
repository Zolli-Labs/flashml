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
