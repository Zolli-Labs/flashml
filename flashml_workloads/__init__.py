"""First-party FlashML example workloads, baked into the workload image.

Entrypoint convention (relied on by the KubeRay backend):
`python -m flashml_workloads.<workload type>` with parameters in the
FLASHML_WORKLOAD_PARAMS env var (JSON) and job identity/artifact wiring in
the other FLASHML_* env vars.
"""
