# ADR 0001: FlashNode on ACK Edge (future heterogeneous device onboarding)

Status: accepted (design only — not implemented in the POC)
Date: 2026-07-17

## Context

FlashML's long-term goal is contributions from heterogeneous devices —
on-premises machines, x86/Arm boxes, GPU workstations — not just cloud VMs.
Alibaba ACK Edge manages exactly that shape: it connects edge nodes
(on-premises, x86, Arm, GPU, IoT-class hardware) into a cloud-managed
Kubernetes control plane, tolerating weak or intermittent connectivity via
edge autonomy (nodes keep running workloads while disconnected).

The POC runs FlashNode as a DaemonSet on standard Kind/ACK nodes. This ADR
records how the same agent extends to ACK Edge later.

## Decision

1. **FlashNode runs unchanged as a DaemonSet on ACK Edge nodes.** The agent
   already derives node identity from the Kubernetes node plus a stable
   local state file, reads capabilities via the downward/Node API, and
   reports outbound-only — all compatible with edge autonomy.
2. **Environment classification becomes `edge`** via the existing
   `FLASHNODE_ENVIRONMENT` / node-label mechanism
   (`flashml.dev/edge: "true"`), so FlashML Cloud can distinguish pools.
3. **Heartbeat-loss handling must tolerate disconnection.** Cloud-side
   offline detection (currently 30 s) gets a separate, much longer edge
   threshold plus a `disconnected` display state; scheduling treats
   disconnected-but-autonomous nodes as unavailable for *new* work without
   marking their in-flight work failed until the lease actually expires.
4. **Trust tiers stay separate from connectivity.** Joining the cluster
   grants zero workload trust: edge/community pools only receive workloads
   whose owners opted into community execution. Untrusted arbitrary
   workloads are never routed to an edge node merely because it registered.
5. **ACK Edge is not ACK Sandboxed Containers.** Sandboxed containers
   require specific ECS Bare Metal instance families in the cloud pool; an
   edge node is never treated as sandbox-capable, and
   `isolation.tier: sandboxed` never schedules to edge.

## Consequences

- No code changes needed in the POC; the contract above constrains future
  scheduler/lease work (lease expiry vs. heartbeat loss must be modeled
  separately).
- Prerequisites before implementation: an ACK Edge cluster with at least one
  real edge node, lease/attempt protocol in `flashruntime.protocol`, and a
  community-pool trust policy in FlashML Cloud.
- We deliberately do not claim protection from a malicious edge-host owner;
  that requires the (separate) verification/redundancy roadmap.
