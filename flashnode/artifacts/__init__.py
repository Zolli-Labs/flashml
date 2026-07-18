"""Input/output/checkpoint staging.

Downloads task inputs, uploads outputs and checkpoint pieces with content
hashes; commits are idempotent so retries never double-count.
"""
