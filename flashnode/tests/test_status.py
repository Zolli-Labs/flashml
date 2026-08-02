"""The status block a volunteer actually reads.

Rendering is a pure function of the loop's counters plus a clock, so it is
tested without a terminal, a thread, or a coordinator.
"""

from __future__ import annotations

import io
import threading
import time

from flashnode.status import StatusView, format_status, human_duration


class _Loop:
    def __init__(self, **kw):
        self.tasks_accepted = 0
        self.tasks_failed = 0
        self.consecutive_failures = 0
        self.current_task = None
        self.current_attempt = None
        self.current_task_started = None
        self.quarantined = False
        self._last_node_hb = 0.0
        self.__dict__.update(kw)


def test_running_shows_the_task_and_how_long_it_has_been_going():
    text = format_status(
        _Loop(current_task="fed-2e2d4d6ab57f", current_attempt=1,
              current_task_started=100.0, tasks_accepted=12, _last_node_hb=138.0),
        coordinator="flashml-api.onrender.com", version="0.3.2", now=140.0,
    )
    assert "fed-2e2d4d6ab57f" in text
    assert "attempt 1" in text
    assert "40s" in text
    assert "12 accepted" in text
    assert "2s ago" in text


def test_idle_says_no_work_queued_is_normal():
    """The single most common worry a volunteer has, answered in the UI
    instead of in a support message."""
    text = format_status(_Loop(_last_node_hb=99.0), coordinator="c",
                         version="0.3.2", now=100.0)
    assert "no work queued" in text
    assert "normal" in text


def test_failures_are_shown_only_when_there_are_some():
    clean = format_status(_Loop(), coordinator="c", version="0.3.2", now=1.0)
    assert "0 failed" in clean
    dirty = format_status(_Loop(tasks_failed=3), coordinator="c",
                          version="0.3.2", now=1.0)
    assert "3 failed" in dirty


def test_the_coordinator_and_version_are_visible():
    text = format_status(_Loop(), coordinator="flashml-api.onrender.com",
                         version="0.3.2", now=1.0)
    assert "flashml-api.onrender.com" in text
    assert "0.3.2" in text


def test_human_duration_is_readable_at_every_scale():
    assert human_duration(9) == "9s"
    assert human_duration(75) == "1m15s"
    assert human_duration(8064) == "2h14m"


def test_a_quarantined_loop_says_so():
    text = format_status(_Loop(quarantined=True), coordinator="c",
                         version="0.3.2", now=1.0)
    assert "stopped" in text.lower()


# -------------------------------------------------------------- the renderer


def test_the_view_redraws_and_stops_cleanly():
    loop = _Loop(tasks_accepted=4)
    out = io.StringIO()
    out.isatty = lambda: True
    view = StatusView(loop, coordinator="c", version="0.3.2", stream=out,
                      interval=0.01)
    view.start()
    time.sleep(0.05)
    view.stop()
    assert "4 accepted" in out.getvalue()
    assert not any(t.name == "flashnode-status" and t.is_alive()
                   for t in threading.enumerate())


def test_a_renderer_that_raises_never_takes_the_work_loop_with_it():
    """The view is cosmetic. A bug in it must not stop a machine from
    contributing — so the thread swallows its own errors and exits.

    Note the loop stand-in raises on ATTRIBUTE ACCESS rather than subclassing
    _Loop with a property: _Loop.__init__ assigns self.tasks_accepted, and
    assigning over a class-level property raises in the constructor, so the
    test would blow up before the thread ever started.
    """
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    out = io.StringIO()
    out.isatty = lambda: True
    view = StatusView(Exploding(), coordinator="c", version="0.3.2",
                      stream=out, interval=0.01)
    view.start()
    time.sleep(0.05)
    view.stop()  # must not raise
    assert not view._thread.is_alive()


def test_the_thread_is_a_daemon_so_it_cannot_block_shutdown():
    out = io.StringIO()
    out.isatty = lambda: True
    view = StatusView(_Loop(), coordinator="c", version="0.3.2", stream=out,
                      interval=0.01)
    view.start()
    assert view._thread.daemon is True
    view.stop()
