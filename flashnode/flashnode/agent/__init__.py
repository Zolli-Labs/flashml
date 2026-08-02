"""Control connection and lifecycle.

Maintains the outbound authenticated WebSocket to the control plane (no
inbound ports required), handles registration, lease acceptance, heartbeat
renewal, and graceful drain/shutdown.
"""
