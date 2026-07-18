"""Command-line entry point for the FlashNode agent.

Target surface (see docs/SYSTEM_OVERVIEW.md §10):

    flashnode join --code <one-time-code>
    flashnode status
    flashnode leave
"""

from __future__ import annotations

import sys

from flashnode import __version__

USAGE = """\
flashnode {version} — FlashML open host agent (pre-release scaffold)

usage: flashnode <command>

commands:
  join      connect this machine to a FlashML control plane (not yet implemented)
  status    show node identity, capabilities, and active leases (not yet implemented)
  leave     drain and disconnect (not yet implemented)
"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(USAGE.format(version=__version__), end="")
    if args and args[0] in {"join", "status", "leave"}:
        print(f"\nerror: '{args[0]}' is not implemented yet in this scaffold.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
