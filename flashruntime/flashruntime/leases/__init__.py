"""Task lease semantics: assignment, renewal via heartbeat, expiration.

A lease grants one node the right to execute a task attempt for a bounded
period. Missed heartbeats past the deadline expire the lease and the task
is reassigned. Only one attempt may ever commit — late duplicates are
rejected (and recorded).

Embeddable, pure-Python, no I/O:

    from flashruntime.leases import LeaseManager
    from flashruntime.protocol.v1alpha1 import TaskSpec

    mgr = LeaseManager(on_event=print)
    mgr.add_task(TaskSpec(task_id="t1", job_id="j1", commit_key="j1/t1"))
    lease = mgr.claim(node_id="laptop-1")
    ... work ...
    mgr.heartbeat(lease.lease_id)
    mgr.complete(lease.lease_id, output_sha256="...")

The FlashRuntime service exposes this same manager over HTTP; flashnode's
device executor is its remote client.
"""

from flashruntime.leases.manager import LeaseError, LeaseManager
from flashruntime.leases.store import InMemoryLeaseStore, LeaseStore, TaskRecord

__all__ = ["LeaseManager", "LeaseError", "LeaseStore", "InMemoryLeaseStore", "TaskRecord"]
