"""RunViewerServer — the read-only HTTP surface over one run directory.

A stdlib `ThreadingHTTPServer` a `flash.submit(watch=True)` opens on the run's
output_dir. It serves exactly four things and 404s everything else:

  GET /             → the run page (a minimal placeholder until Task 7)
  GET /api/state    → state.collect(run_dir) as JSON (the live snapshot)
  GET /docs, /docs/…→ static files under the packaged _docs/ dir, else 404
  (anything else)   → 404

Threading, not a single-thread loop: the page polls /api/state every couple
of seconds while a human also clicks around; a blocking handler would stall
one request behind another. Everything is bound to 127.0.0.1 — this is a
personal, local viewer, never a network service.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from flashruntime.viewer.state import collect

# Packaged docs live here once Task 8's builder has run; absent in a fresh
# checkout, which is why /docs degrades to an honest "docs not built" 404.
_DEFAULT_DOCS_DIR = Path(__file__).parent / "_docs"

_PLACEHOLDER_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>flashruntime run</title></head>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem">
<h1>flashruntime run viewer</h1>
<p>The live run page arrives in the next release. Until then:</p>
<ul>
  <li><a href="/api/state">/api/state</a> &mdash; this run's snapshot as JSON</li>
  <li><a href="/docs/">/docs/</a> &mdash; documentation</li>
</ul>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    """Routes GETs for one run. Reads its run_dir/docs_dir off the server
    instance (`self.server`), so a single handler class serves any run."""

    # Silence the default per-request stderr line: the user launched this from
    # the same terminal their training logs to, and access-log spam would bury
    # that output. (The viewer is deliberately quiet.)
    def log_message(self, *args) -> None:  # noqa: D401
        pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _PLACEHOLDER_PAGE.encode())
        elif path == "/api/state":
            # collect() is total (never raises), so this branch cannot 500 on
            # any on-disk state — it returns a snapshot or an {"error": ...}.
            body = json.dumps(self.server.run_state()).encode()
            self._send(200, "application/json; charset=utf-8", body)
        elif path == "/docs" or path.startswith("/docs/"):
            self._serve_docs(path)
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def _serve_docs(self, path: str) -> None:
        docs_dir: Path = self.server.docs_dir
        if not docs_dir.is_dir():
            # The builder (Task 8) has not run in this checkout.
            self._send(404, "text/plain; charset=utf-8", b"docs not built")
            return
        rel = path[len("/docs"):].lstrip("/") or "index.html"  # /docs and /docs/ → index.html
        target = (docs_dir / rel).resolve()
        # PATH-TRAVERSAL GUARD: `rel` comes straight from the request line, so
        # a crafted "/docs/../../etc/passwd" would otherwise resolve OUTSIDE
        # docs_dir. Reject anything whose resolved path is not under the docs
        # root — serving files by request path without this check is a hole.
        if not target.is_relative_to(docs_dir.resolve()) or not target.is_file():
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        try:
            body = target.read_bytes()
        except OSError:
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        self._send(200, ctype, body)

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RunViewerServer:
    """Serve one run directory over HTTP on 127.0.0.1.

    `port=0` (the default) asks the OS for a free port, so opening a viewer
    never collides with another run's. `.start()` binds, spawns a daemon
    serve loop, and returns the URL; `.stop()` shuts the loop down and closes
    the socket (frees the port). `docs_dir` defaults to the packaged _docs/
    but is injectable — the traversal guard and the docs routes are only
    testable against a real directory, and the package's own _docs/ does not
    exist until Task 8 builds it.
    """

    def __init__(self, run_dir: Path, port: int = 0, docs_dir: Path | None = None):
        self.run_dir = Path(run_dir)
        self.docs_dir = Path(docs_dir) if docs_dir is not None else _DEFAULT_DOCS_DIR
        self._requested_port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url: str | None = None

    def start(self) -> str:
        httpd = ThreadingHTTPServer(("127.0.0.1", self._requested_port), _Handler)
        # Hand the handler its per-server config. run_state is a bound method
        # so each request re-reads the run dir fresh (a live snapshot).
        httpd.run_state = lambda: collect(self.run_dir)  # type: ignore[attr-defined]
        httpd.docs_dir = self.docs_dir  # type: ignore[attr-defined]
        httpd.daemon_threads = True  # worker threads never block interpreter exit
        self._httpd = httpd
        port = httpd.server_address[1]  # the OS-assigned port when we asked for 0
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{port}"
        return self.url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()  # stop serve_forever
            self._httpd.server_close()  # close the listening socket → port freed
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
