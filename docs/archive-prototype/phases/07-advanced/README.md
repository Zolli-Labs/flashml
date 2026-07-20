# Phase 7 — Advanced Distributed Systems

**State:** Research; no committed delivery date.

## Outcome under investigation

Determine which advanced workloads require a new persistent-cluster provider
capability instead of forcing them through the serverless task interface.

## Research requirements

- [ ] Specify persistent, mutually reachable worker capabilities.
- [ ] Prototype true DDP/NCCL only on providers satisfying that capability.
- [ ] Design checkpoint/restart across spot interruption and process loss.
- [ ] Evaluate cost-aware provider routing from normalized offers.
- [ ] Evaluate a single job spanning providers, including data movement and consistency costs.
- [ ] Record explicit go/no-go criteria before changing the stable provider contract.

## Exit gate

Research becomes an implementation phase only after a design document proves
the required capability, security boundary, cost model, and compatibility
impact on existing serverless connectors.
