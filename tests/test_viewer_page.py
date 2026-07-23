# tests/test_viewer_page.py
"""The live run page (`viewer.page.render()`) and its open-on-submit wiring.

Two things are proven here, both headless (no browser):

  1. render() is ONE self-contained HTML document — the sections the design
     contract names are present (topology, loss, checkpoints, events, the
     /api/state poll target) and it references NO off-host asset (no CDN,
     no remote font/image/script). A strict regex over every src=/href= is
     the guard: a self-contained page must survive with the network cut.
  2. submit(watch=True) opens a viewer: it starts a RunViewerServer bound to
     127.0.0.1, records the URL on Run.viewer_url, calls webbrowser.open
     (monkeypatched here — a test must never actually pop a browser), and
     the server serves a live /api/state for the run while it executes.

The visual tokens live in viewer.page as the single source of truth; the
coordinator dashboard mirrors them, so a palette-alignment check lives here
too (the shared accent must appear where the old ad-hoc hex used to).
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
import urllib.request

from flashruntime.viewer.page import TOKENS, render
from flashruntime.viewer.server import RunViewerServer

# Every src=/href= value the page ships. A self-contained document may point
# only at same-origin paths (/docs/, #anchors, /api/...) — never off-host, so
# no value may carry a scheme (`://`) or be protocol-relative (`//host`).
_ASSET_REF = re.compile(r'(?:src|href)\s*=\s*"([^"]*)"', re.IGNORECASE)


# --------------------------------------------------------------------------
# render(): the self-contained document
# --------------------------------------------------------------------------


def test_render_contains_required_section_markers():
    html = render()
    for marker in ('id="topology"', 'id="loss"', 'id="checkpoints"', 'id="events"'):
        assert marker in html, f"missing section marker {marker}"
    # the page must poll the live snapshot endpoint
    assert "/api/state" in html


def test_render_is_a_single_html_document():
    html = render()
    assert html.lstrip().lower().startswith("<!doctype html>")
    # everything inline: a <script> and a <style>, no external <link rel=stylesheet>
    assert "<script>" in html and "<style>" in html
    assert 'rel="stylesheet"' not in html.lower()


def test_render_has_no_offhost_asset_references():
    html = render()
    for value in _ASSET_REF.findall(html):
        assert "://" not in value, f"off-host asset reference: {value!r}"
        assert not value.startswith("//"), f"protocol-relative asset reference: {value!r}"


def test_render_links_docs_and_polls_named_draw_functions():
    html = render()
    # Docs affordance points at the same-origin docs mount.
    assert "/docs/" in html
    # Drawing is organized by section, per the contract (readable, named JS).
    assert "drawTopology" in html
    assert "drawLoss" in html
    # Battery courtesy: polling pauses when the tab is hidden.
    assert "document.hidden" in html


def test_render_uses_the_shared_tokens():
    html = render()
    # The oklch accents (the single source of truth) are actually wired in.
    assert TOKENS["running"] in html  # cyan
    assert TOKENS["ok"] in html  # green
    assert TOKENS["ckpt"] in html  # violet
    assert TOKENS["bg"] in html  # #0d1117 background family


# --------------------------------------------------------------------------
# palette alignment: the coordinator dashboard mirrors the same tokens
# --------------------------------------------------------------------------


def test_dashboard_palette_uses_shared_tokens():
    """dashboard.py swaps its ad-hoc hex accents for the shared tokens — the
    two surfaces read as one house style (values, not structure, change)."""
    from flashruntime.service.dashboard import _PAGE

    assert TOKENS["running"] in _PAGE  # cyan accent now shared
    assert TOKENS["ckpt"] in _PAGE  # violet accent now shared
    # the old ad-hoc hexes are gone (replaced by the tokens)
    assert "#58a6ff" not in _PAGE
    assert "#d2a8ff" not in _PAGE


# --------------------------------------------------------------------------
# the server serves the real page at /
# --------------------------------------------------------------------------


def test_server_root_serves_the_run_page(tmp_path):
    (tmp_path / "run.json").write_text(
        json.dumps({"contract": "viewer_v1", "state": "PENDING", "attempts": [], "events": []})
    )
    server = RunViewerServer(tmp_path)
    url = server.start()
    try:
        with urllib.request.urlopen(url + "/") as resp:
            body = resp.read().decode()
        assert 'id="topology"' in body
        assert "/api/state" in body
    finally:
        server.stop()


# --------------------------------------------------------------------------
# open-on-submit: watch=True starts + records a live viewer
# --------------------------------------------------------------------------


def _write_script(tmp_path, body: str) -> str:
    src = tmp_path / "userproj"
    src.mkdir(exist_ok=True)
    (src / "train.py").write_text(textwrap.dedent(body))
    return str(src)


def test_submit_watch_true_opens_and_serves_live(tmp_path, monkeypatch):
    import flashruntime as flash

    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda u, *a, **k: opened.append(u) or True)

    # A short-lived run so we can observe /api/state while it is still live.
    source = _write_script(
        tmp_path,
        """
        import json, time
        time.sleep(0.4)
        json.dump({"accuracy": 0.9}, open("metrics.json", "w"))
        """,
    )
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": source}),
        output_dir=tmp_path / "out",
        wait=False,  # return immediately: the server is live during the run
        watch=True,
    )
    try:
        # the viewer opened, bound to loopback, and its URL is on the Run
        assert run.viewer_url is not None
        assert run.viewer_url.startswith("http://127.0.0.1:")
        assert opened == [run.viewer_url]  # webbrowser.open got exactly that URL

        # /api/state serves the LIVE snapshot while the run executes
        with urllib.request.urlopen(run.viewer_url + "/api/state") as resp:
            live = json.loads(resp.read())
        assert live["contract"] == "viewer_v1"
        assert live["state"] in ("PENDING", "RUNNING", "SUCCEEDED")

        # after completion the page still serves the final snapshot
        run.wait(timeout=10)
        with urllib.request.urlopen(run.viewer_url + "/api/state") as resp:
            done = json.loads(resp.read())
        assert done["state"] == "SUCCEEDED"
    finally:
        pass  # server is a daemon thread; it is freed at interpreter exit


def test_submit_watch_false_starts_no_viewer(tmp_path, monkeypatch):
    import flashruntime as flash

    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda u, *a, **k: opened.append(u) or True)

    source = _write_script(
        tmp_path,
        """
        import json
        json.dump({"accuracy": 0.9}, open("metrics.json", "w"))
        """,
    )
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": source}),
        output_dir=tmp_path / "out",
        watch=False,
    )
    assert run.state.value == "SUCCEEDED"
    assert run.viewer_url is None  # no viewer requested → none started
    assert opened == []  # and no browser opened
