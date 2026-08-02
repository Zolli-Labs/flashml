"""Device-code enrolment, the client half.

`flashnode login` is the first command a volunteer runs and the one they
are least equipped to debug, so the cases here are the ones that decide
whether they get in or give up.
"""
from __future__ import annotations

import pytest

from flashnode.identity.enrol import (
    DEFAULT_INTERVAL,
    DeviceCodeStart,
    EnrolmentError,
    describe_this_machine,
    poll_for_token,
    request_device_code,
)

API = "https://api.test"


class FakeHttp:
    """Scripted responses, and a record of what was actually sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self.responses.pop(0)


# --- starting the flow -----------------------------------------------------


def test_request_device_code_returns_what_the_terminal_prints():
    http = FakeHttp(
        (
            200,
            {
                "device_code": "dev-abc",
                "user_code": "WXYZ-1234",
                "verification_uri": "https://console.test/activate",
                "interval": 5,
                "expires_at": "2026-08-01T13:00:00+00:00",
            },
        )
    )
    start = request_device_code(API, "fn-1", "laptop", "macOS-15", http=http)
    assert isinstance(start, DeviceCodeStart)
    assert start.user_code == "WXYZ-1234"
    assert start.verification_uri == "https://console.test/activate"
    assert start.interval == 5

    url, payload = http.calls[0]
    assert url == "https://api.test/v1alpha1/device/code"
    assert payload == {
        "node_id": "fn-1",
        "hostname": "laptop",
        "platform": "macOS-15",
    }


def test_a_trailing_slash_on_the_base_url_does_not_double_up():
    """People paste URLs with trailing slashes. //v1alpha1/... 404s."""
    http = FakeHttp(
        (200, {"device_code": "d", "user_code": "u", "verification_uri": "/activate"})
    )
    request_device_code("https://api.test/", "fn-1", "h", "p", http=http)
    assert http.calls[0][0] == "https://api.test/v1alpha1/device/code"


def test_a_missing_interval_falls_back_rather_than_crashing():
    http = FakeHttp(
        (200, {"device_code": "d", "user_code": "u", "verification_uri": "/activate"})
    )
    start = request_device_code(API, "fn-1", "h", "p", http=http)
    assert start.interval == DEFAULT_INTERVAL


def test_a_rejected_node_id_is_reported_not_swallowed():
    http = FakeHttp((400, {"detail": "invalid node_id"}))
    with pytest.raises(EnrolmentError, match="invalid node_id"):
        request_device_code(API, "bad id", "h", "p", http=http)


def test_a_response_of_the_wrong_shape_names_itself():
    http = FakeHttp((200, {"unexpected": True}))
    with pytest.raises(EnrolmentError, match="expected shape"):
        request_device_code(API, "fn-1", "h", "p", http=http)


# --- polling ---------------------------------------------------------------


def test_polls_until_a_human_approves():
    pending = (400, {"error": "authorization_pending", "interval": 5})
    http = FakeHttp(pending, pending, (200, {"token": "machine-token"}))
    slept: list[float] = []

    token = poll_for_token(
        API, "dev-abc", interval=5, http=http, sleep=slept.append, now=lambda: 0.0
    )

    assert token == "machine-token"
    assert len(http.calls) == 3
    assert slept == [5, 5]
    assert http.calls[0] == (
        "https://api.test/v1alpha1/device/token",
        {"device_code": "dev-abc"},
    )


def test_the_server_can_slow_us_down_mid_flight():
    http = FakeHttp(
        (400, {"error": "authorization_pending", "interval": 20}),
        (200, {"token": "t"}),
    )
    slept: list[float] = []
    poll_for_token(API, "d", interval=5, http=http, sleep=slept.append, now=lambda: 0.0)
    assert slept == [20]


@pytest.mark.parametrize("bad", [0, -5, "", None])
def test_a_nonsense_interval_never_becomes_a_hot_loop(bad):
    """A server bug must not turn every volunteer's laptop into a load
    generator against our own API."""
    http = FakeHttp(
        (400, {"error": "authorization_pending", "interval": bad}),
        (200, {"token": "t"}),
    )
    slept: list[float] = []
    poll_for_token(API, "d", interval=bad, http=http, sleep=slept.append, now=lambda: 0.0)
    assert slept and all(s >= 1.0 for s in slept)


def test_an_absurdly_long_interval_is_capped():
    http = FakeHttp(
        (400, {"error": "authorization_pending", "interval": 99999}),
        (200, {"token": "t"}),
    )
    slept: list[float] = []
    poll_for_token(API, "d", http=http, sleep=slept.append, now=lambda: 0.0)
    assert slept == [30.0]


def test_gives_up_eventually_and_says_what_to_do():
    """The server answers unapproved and expired identically, on purpose, so
    the timeout message is the only place the likely causes can be named."""
    clock = iter([0.0, 0.0, 999.0, 999.0])
    http = FakeHttp(
        (400, {"error": "authorization_pending"}),
        (400, {"error": "authorization_pending"}),
    )
    with pytest.raises(EnrolmentError, match="timed out"):
        poll_for_token(
            API,
            "d",
            http=http,
            sleep=lambda _s: None,
            now=lambda: next(clock),
            max_seconds=60,
        )


def test_an_unexpected_status_stops_immediately():
    """Polling a 500 forever would hide an outage behind a spinner."""
    http = FakeHttp((500, {"detail": "database is down"}))
    with pytest.raises(EnrolmentError, match="database is down"):
        poll_for_token(API, "d", http=http, sleep=lambda _s: None, now=lambda: 0.0)


# --- machine description ---------------------------------------------------


def test_describe_this_machine_is_bounded_and_non_empty():
    """Shown on the approval screen so a human can tell which laptop this
    is. Bounded because it is written to a database column."""
    hostname, platform_name = describe_this_machine()
    assert hostname and len(hostname) <= 120
    assert platform_name and len(platform_name) <= 120
