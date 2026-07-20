"""Placement/scheduling interfaces — intentionally not built yet.

Today claim order is FIFO over PENDING tasks (see `leases/`): honest and
sufficient while tasks are CPU-sized and homogeneous. This package becomes
real when payloads carry resource requirements (e.g. vram_gb) and claims
must filter by node capability — see research item R9 in the workspace
`HANDBOOK.md` §7. Score-based placement comes only after the ledger holds
enough reliability data to justify it (master report §9: capability,
reliability, and trust stay separate assessments).
"""
