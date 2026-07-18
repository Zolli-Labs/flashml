"""Checkpoint manifests and compatibility scoring.

A checkpoint is a manifest (job, attempt, step, world size, rank objects,
hashes, validation status), not just a path. The control plane restores
only from manifests marked complete and topology-compatible.
"""
