# FlashML Roadmap

This is the roadmap index. Detailed requirements, missing work, acceptance
evidence, and exit gates live in one folder per phase under [`phases/`](phases/).

For the live snapshot, start with [`status/README.md`](status/README.md).

| Phase | Outcome | State | Detailed plan |
|---|---|---|---|
| 0 | Provider-agnostic library and local execution | Implemented; baseline commit pending | [`00-foundation`](phases/00-foundation/README.md) |
| 1 | RunPod behind the same provider interface | **Current** | [`01-runpod-connector`](phases/01-runpod-connector/README.md) |
| 2 | PyTorch/FedAvg and embarrassingly parallel jobs | Ready to start in parallel | [`02-model-support`](phases/02-model-support/README.md) |
| 3 | `flashml serve` and dashboard migration | Waiting on Phases 1–2 | [`03-serve-dashboard`](phases/03-serve-dashboard/README.md) |
| 4 | Reusable conformance suite and SSH connector | Waiting on Phase 1 | [`04-conformance-ssh`](phases/04-conformance-ssh/README.md) |
| 5 | v1 API and operational hardening | Planned | [`05-v1-hardening`](phases/05-v1-hardening/README.md) |
| 6 | Additional provider connectors | Planned | [`06-multi-provider`](phases/06-multi-provider/README.md) |
| 7 | Persistent clusters, DDP, resilience, routing | Research | [`07-advanced`](phases/07-advanced/README.md) |

## Rules for advancing a phase

A phase is complete only when all of the following are true:

1. Every required deliverable in its phase document is checked.
2. Its verification commands pass.
3. Its exit criteria have direct evidence, not only implementation intent.
4. The phase document records that evidence.
5. [`status/README.md`](status/README.md), this table, and the root README agree.
6. The implementation and documentation are committed together.

Optional work never blocks a phase. Deferred work must name the phase that owns
it so it cannot silently disappear.
