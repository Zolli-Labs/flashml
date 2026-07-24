"""Shared prerequisite checks for the standalone demos.

Why this exists: the DDP demos need `torchrun` on PATH, and the usual way to
get that wrong is to run `.venv/bin/python examples/...` — which never puts
`.venv/bin` on PATH. `bring_your_code_demo.py` reacts by silently skipping
the PyTorch acts; a single-purpose demo must instead fail loudly, because a
demo that prints nothing it promised is worse than one that errors.
"""

from __future__ import annotations

import shutil
import sys

ACTIVATE_HINT = (
    "Activate the venv rather than calling its interpreter by path:\n"
    "    cd <repo root> && source .venv/bin/activate\n"
    "(`.venv/bin/python examples/...` leaves `.venv/bin` off PATH, so the\n"
    "launcher cannot find torchrun.)"
)


def require(condition: bool, message: str) -> None:
    """Exit with a usable message instead of a traceback or a silent skip."""
    if not condition:
        sys.exit(f"\nPREREQUISITE MISSING\n\n{message}\n")


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401, PLC0415 — presence check only
    except ImportError:
        require(False, "scikit-learn is not installed:\n    uv pip install -e '.[sklearn]'")


def require_torchrun() -> None:
    """torch importable AND torchrun launchable — they fail separately."""
    try:
        import torch  # noqa: F401, PLC0415 — presence check only
    except ImportError:
        require(False, "torch is not installed (a CPU-only build is enough):\n    pip install torch")
    require(
        shutil.which("torchrun") is not None,
        f"torch is installed but `torchrun` is not on PATH.\n\n{ACTIVATE_HINT}",
    )


def hold_viewer(run) -> None:
    """Block so the live viewer stays reachable.

    The viewer server runs on a daemon thread of THIS process (see
    `flashruntime.sdk._open_viewer`), so it dies the moment the script
    returns. Any demo that opened one has to wait here or the page is gone
    before the user can look at it.
    """
    if not run.viewer_url:
        return
    print(f"\nviewer still live at {run.viewer_url}")
    try:
        input("press enter to stop it and exit... ")
    except (EOFError, KeyboardInterrupt):
        print()
