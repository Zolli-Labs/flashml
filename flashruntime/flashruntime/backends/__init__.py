"""Execution backends: pluggable engines that run a FlashRuntime Job.

FlashRuntime owns the public JobSpec, job state, events, and artifacts; a
backend owns nothing but the translation to (and observation of) one
concrete execution system. The first real backend is KubeRay
(`flashruntime.backends.kuberay`). Documented-but-unimplemented backends
(PAI-DLC, torchrun, Dask, DeepSpeed, Slurm) are described in
docs/adr/0002-pai-dlc-backend.md and must satisfy
`flashruntime.backends.base.ExecutionBackend`.
"""

from flashruntime.backends.base import (
    BackendExecution,
    BackendStatus,
    BackendUnavailableError,
    ExecutionBackend,
    SpecValidationError,
)

__all__ = [
    "BackendExecution",
    "BackendStatus",
    "BackendUnavailableError",
    "ExecutionBackend",
    "SpecValidationError",
]
