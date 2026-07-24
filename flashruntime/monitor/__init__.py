"""Run telemetry: the optional-psutil resource sampler the SDK starts per
launched attempt. Read side: `viewer.state.collect()` tails telemetry.jsonl.
"""

from flashruntime.monitor.sampler import ResourceSampler

__all__ = ["ResourceSampler"]
