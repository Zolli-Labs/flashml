"""Versioned public schemas shared by FlashNode and FlashML Cloud.

Planned contents (see docs/SYSTEM_OVERVIEW.md §Repository boundaries):
job specifications, node registration and heartbeat messages, lease and
attempt payloads, checkpoint manifests, and the event vocabulary. Every
schema carries an explicit version field; security-relevant fields fail
closed on unknown values.
"""
