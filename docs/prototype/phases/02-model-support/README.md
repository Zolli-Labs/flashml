# Phase 2 — Generalized Model Support

**State:** Ready; may run in parallel with Phase 1 against `local`.

## Outcome

FlashML supports the three intended distributed shapes instead of being
limited to sklearn: MapReduce, parameter server/FedAvg, and embarrassingly
parallel work.

## Required deliverables

- [ ] Safe PyTorch checkpoint codec (`state_dict` to bytes and back), preferring `safetensors`.
- [ ] `TorchParameterServerAlgorithm` with checkpoint keys in task payloads.
- [ ] Real local mini-batch training and sample-weighted FedAvg reduction.
- [ ] Dotted import paths for model, optimizer, and loss factories.
- [ ] Finalization returns a usable loaded `nn.Module`.
- [ ] `EmbarrassinglyParallelAlgorithm` for independent fits/sweeps.
- [ ] Convergence-focused PyTorch example and tests on `local`.
- [ ] Independent-per-shard example and tests.
- [ ] Update the adapter table in `docs/MODELS.md`.
- [ ] Re-run both adapters on RunPod after Phase 1.

## Constraints

Large model state goes through `Storage`; task messages contain keys and small
metrics. Live objects/cloudpickle are deferred. True DDP/NCCL is not part of
this phase because serverless workers lack the required persistent network.

## Verification

Tests must prove numerical learning, not merely that a loop completed. The
PyTorch result must perform above a documented baseline, and independent jobs
must return one usable result per input unit.

## Exit gate

Both adapters work on `local`, are documented through the public API, and are
re-verified on RunPod when Phase 1 is available.
