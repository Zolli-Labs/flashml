"""What a host owner sees while their machine is working.

`flashnode work` used to emit only a JSON log stream, so a volunteer could
not tell contributing from quietly-failing from idle-because-nothing-is-
queued. That last one matters most: an idle agent looks identical to a broken
one, and "no work queued — this is normal" is the answer to the most common
worry a host has.

Rendering is a pure function of the loop's counters plus a clock — no
terminal, no thread, no coordinator needed to test it.
"""

from __future__ import annotations

import threading
import time

__all__ = ["StatusView", "format_status", "human_duration"]


def human_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def format_status(loop, *, coordinator: str, version: str, now: float,
                  started: float = 0.0) -> str:
    lines = [
        f"flashnode {version} · {coordinator} · up {human_duration(now - started)}"
    ]
    if loop.quarantined:
        lines.append("  stopped    this machine can no longer run tasks — see below")
    elif loop.current_task:
        elapsed = human_duration(now - (loop.current_task_started or now))
        lines.append(
            f"  running    {loop.current_task}  ·  "
            f"attempt {loop.current_attempt}  ·  {elapsed}"
        )
    else:
        lines.append("  waiting    no work queued — this is normal")
    lines.append(
        f"  session    {loop.tasks_accepted} accepted   {loop.tasks_failed} failed"
    )
    lines.append(f"  heartbeat  {human_duration(now - loop._last_node_hb)} ago")
    return "\n".join(lines)


class StatusView:
    """Redraws `format_status` in place on a timer, from its own thread.

    It READS the loop and never calls into it. That is the whole design: the
    executor is correctness-critical — leases, heartbeats, the checkpoint
    relay — and a cosmetic feature must not appear anywhere in its control
    flow. The thread is a daemon, swallows its own exceptions, and if it dies
    the machine keeps contributing.
    """

    def __init__(self, loop, *, coordinator: str, version: str, stream,
                 interval: float = 1.0):
        self.loop = loop
        self.coordinator = coordinator
        self.version = version
        self.stream = stream
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.monotonic()
        self._lines_drawn = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="flashnode-status",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._draw()
            except Exception:
                # Cosmetic. Never take the work loop down with the view.
                return
            self._stop.wait(self.interval)

    def _draw(self) -> None:
        text = format_status(self.loop, coordinator=self.coordinator,
                             version=self.version, now=time.monotonic(),
                             started=self._started)
        if self._lines_drawn:
            # Move up and clear, so the block updates in place rather than
            # scrolling. Only ever reached on a TTY — cli.py gates this.
            self.stream.write(f"\033[{self._lines_drawn}A\033[J")
        self.stream.write(text + "\n")
        self.stream.flush()
        self._lines_drawn = len(text.splitlines())
