"""Task lease semantics: assignment, renewal via heartbeat, expiration.

A lease grants one node the right to execute a task attempt for a bounded
period. Missed heartbeats past policy threshold expire the lease and the
task is reassigned. Only one attempt may commit the final artifact.
"""
