"""Task execution: the device profile's pull-based work cycle.

`ExecutorLoop` claims leases from a FlashRuntime coordinator (outbound
only), downloads shared input artifacts, runs the task through a runner,
uploads outputs, and commits — with attempt heartbeats renewing the lease
throughout.

Runner tiers behind one interface:
- Tier 1 `SubprocessRunner` (implemented): allowlisted Python modules in a
  fresh subprocess with a wall-clock timeout. Dev/trusted-pool profile —
  not a security boundary.
- Tier 2 Docker (next): allowlisted images, cpu/memory limits,
  `--network none`, non-root, read-only rootfs. No host Docker socket
  exposure to workloads, no privileged mode. gVisor/Kata tiers later.
"""

from flashnode.executor.client import CoordinatorClient, LeaseLost
from flashnode.executor.loop import ExecutorLoop
from flashnode.executor.runner import SubprocessRunner, TaskExecutionError

__all__ = [
    "CoordinatorClient",
    "ExecutorLoop",
    "LeaseLost",
    "SubprocessRunner",
    "TaskExecutionError",
]
