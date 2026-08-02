import json
import stat

import pytest

from flashnode.identity.credentials import (
    clear_token, credentials_path, load_token, save_token,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHNODE_CREDENTIALS", str(tmp_path / "credentials.json"))


def test_no_token_before_login():
    assert load_token("https://c.example") is None


def test_save_then_load_round_trip():
    save_token("https://c.example", "tok-a")
    assert load_token("https://c.example") == "tok-a"


def test_tokens_are_scoped_per_coordinator():
    """One machine may join several pools; a second login must not clobber
    the first."""
    save_token("https://a.example", "tok-a")
    save_token("https://b.example", "tok-b")
    assert load_token("https://a.example") == "tok-a"
    assert load_token("https://b.example") == "tok-b"


def test_credentials_file_is_not_world_readable():
    save_token("https://c.example", "tok-a")
    mode = credentials_path().stat().st_mode
    assert not mode & stat.S_IRGRP
    assert not mode & stat.S_IROTH


def test_clear_removes_only_that_coordinator():
    save_token("https://a.example", "tok-a")
    save_token("https://b.example", "tok-b")
    assert clear_token("https://a.example") is True
    assert load_token("https://a.example") is None
    assert load_token("https://b.example") == "tok-b"


def test_clear_is_idempotent():
    assert clear_token("https://nothing.example") is False


def test_a_corrupt_credentials_file_does_not_crash_the_agent():
    credentials_path().parent.mkdir(parents=True, exist_ok=True)
    credentials_path().write_text("{not json")
    assert load_token("https://c.example") is None


def test_trailing_slashes_do_not_split_the_identity():
    save_token("https://c.example/", "tok-a")
    assert load_token("https://c.example") == "tok-a"


def test_client_sends_bearer_header_when_a_token_is_present():
    from flashnode.executor.client import CoordinatorClient

    c = CoordinatorClient("http://c.example", token="tok-a")
    assert c._headers().get("Authorization") == "Bearer tok-a"


def test_client_sends_no_auth_header_without_a_token():
    from flashnode.executor.client import CoordinatorClient

    c = CoordinatorClient("http://c.example")
    assert "Authorization" not in c._headers()
