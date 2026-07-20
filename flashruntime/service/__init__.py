"""FlashRuntime API service: accepts JobSpecs, drives an ExecutionBackend,
keeps the job/event/artifact ledger, and serves it to FlashML Cloud (or any
self-hosted client — the runtime is useful without the cloud)."""
