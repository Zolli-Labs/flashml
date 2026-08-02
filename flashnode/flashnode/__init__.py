"""FlashNode: the open host agent of the FlashML system.

Installed by resource contributors to safely execute distributed ML tasks.
See README.md and docs/SYSTEM_OVERVIEW.md.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Read from installed metadata rather than hardcoding. A literal here is a
# SECOND source of truth for the version, and it drifted the moment it
# existed: 0.2.0 shipped to PyPI while this file still said 0.1.0, so
# `flashnode --help` reported 0.1.0 and — far worse — every agent registered
# with the coordinator as agent_version="0.1.0" regardless of what was
# actually installed (inventory/capabilities.py). Any version-based decision
# the coordinator makes would have been reading a constant.
try:
    __version__ = _pkg_version("flashnode")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
