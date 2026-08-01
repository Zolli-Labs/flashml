# Federated averaging

Federated averaging (FedAvg) is FlashRuntime's answer to a question the
[PyTorch guide](pytorch.md) and the
[JobSpec & isolation guide](jobspec-and-isolation.md) both run into: what do
you do when the machines that want to help train a model **cannot talk to
each other**? Volunteer nodes run their task containers with `--network
none` (see the repo's `docs/guides/donate-a-machine.md`) — no LAN, no
internet, no way for one container to find another. Coordinated
multi-process training (DDP, FSDP) needs the opposite: every rank must
rendezvous with every other rank over a process group before the first
`all_reduce`. On a volunteer pool those two requirements are irreconcilable,
so FlashRuntime does not attempt coordinated training there at all.

FedAvg sidesteps the rendezvous problem by never requiring it. Instead of
ranks synchronizing gradients mid-step, **rounds** synchronize whole models
between steps:

1. The driver broadcasts the current weights as a plain `artifact://` blob.
2. Each participating node downloads the weights, trains **independently**
   for a fixed number of local steps on its own data shard, and uploads a
   weight **delta** (not the new weights — see below).
3. Once enough deltas have committed, the driver averages them, applies the
   result to the broadcast weights, and starts the next round.

No node ever needs to see another node's IP address, let alone open a
connection to it. Every cross-node interaction is a `PUT`/`GET` against the
coordinator's artifact store, which is exactly the same shape of traffic a
volunteer node already does to pull its task inputs and push its results.
That is why this is a **round loop implemented as a driver chaining
ordinary lease jobs** (`flashml_workloads/fedavg_driver.py`,
`flashml_workloads/fedavg_worker.py`) — the same "pipelines are jobs chained
by a driver, not a new execution mode" pattern as the sharded-k-means POC —
rather than a new backend.

## Why a delta, not the new weights

Each worker's task uploads `delta.json` (the change it made to the weights
it started from) alongside `metrics.json`. The driver averages **deltas**,
not raw weight snapshots, because a delta is a direction that stays
meaningful even if the weights it was computed against are no longer the
newest ones — the exact situation a straggling volunteer produces when it
finally reports in after the round has moved on. Averaging final weights
directly would require every worker to have started from the *same*
snapshot; averaging deltas only requires knowing what each worker started
from, which the driver already does.

## The quorum rule, and why late deltas are discarded

`kmeans_driver` (the other job-chaining driver in this codebase) requires
**every** dispatched shard to report before it aggregates. FedAvg
deliberately does not: `run_fedavg(..., min_participants=N)` aggregates as
soon as `N` of the round's shards have committed, not when all of them have.

This is not a shortcut — it is the correct policy for volunteer compute.
Machines that donate spare cycles are unequal and unreliable by
construction: laptops close, Wi-Fi drops, a slow machine might still be on
local step 3 when a fast one has already finished. Requiring all of them
before a round can proceed would let a single closed laptop stall every
other participant's contribution indefinitely. Quorum aggregation lets the
round move on as soon as it has a statistically meaningful sample.

The corollary is what makes quorum aggregation *safe* rather than merely
convenient: once the driver has read the quorum's deltas and applied them,
**any delta that commits afterward for that round is discarded**, never
folded into a later round. `run_fedavg` freezes the participant set at the
moment quorum is reached and never re-reads that job's artifacts again
(`fedavg_driver.py`, `run_fedavg`). A late delta was computed against
weights that no longer exist by the time it arrives — the model has already
moved past them — and applying it on top of a newer round's weights would
not be "one more contribution," it would silently corrupt the average with
a step that was never actually taken from the current state. Discarding is
the honest behavior; a driver that tried to be more "inclusive" here would
be quietly wrong instead.

`tests/test_fedavg_convergence.py::test_round_completes_on_quorum_when_a_node_never_reports`
pins exactly this: three shards are dispatched but the test's agent pool is
capped to exactly two successful claims and then stops claiming, so the
third shard is never bound to any node and sits PENDING for the life of the
test. The round still aggregates on the two that committed — with an exact
`participants == 2` assertion — rather than hanging until the deadline
waiting for the shard nobody was ever going to serve. (The cap on claims,
not the node count, is what makes the third shard genuinely abandoned:
either registered node can claim either shard, so without the cap both
nodes could sequentially serve all three before the driver's poll notices
quorum.)

## The `flashml.yaml` shape

A federated-averaging round is submitted as an ordinary lease-mode job:

```yaml
apiVersion: flashml.dev/v1alpha1
kind: Job
metadata:
  name: fedavg-r000
spec:
  execution:
    backend: leases
  image:
    repository: local/tier1
    tag: dev
  workload:
    type: federated_averaging
    parameters:
      round: 0
      num_shards: 2
      local_steps: 20
      lr: 0.1
      batch_size: 16
      seed: 0
      in_dim: 8
      hidden: 16
      out_dim: 2
      dataset_size: 256
      # weights: artifact://jobs/<prev-job>/round-000/weights.json
      # (omitted on round 0 — each worker seeds its own model from `seed`)
```

`isolation.tier` is left at its default, `"standard"`, deliberately: unlike
the `argv` runner tier for arbitrary bring-your-code jobs, a
`federated_averaging` task's payload is a fixed, trusted `module` execution
(`flashml_workloads.fedavg_worker`), so it does not need the sandboxed argv
path and its `argv_capable` gate. A node only needs `module_capable`
(fail-open — absent counts as capable) to be eligible. `run_fedavg` builds
this JobSpec once per round and submits it as a new job
(`flashml_workloads/fedavg_driver.py:_round_body`) — the round number is
the only thing that changes between the driver's own resume points.

## What this proves — and what it does not

`tests/test_fedavg_convergence.py` runs this loop against a **real**
coordinator over real HTTP: real job expansion, real leases, real local
artifact storage (`FLASHML_LOCAL_ARTIFACTS_DIR`), and real commit-time
sha256 validation on every uploaded artifact. Two independent worker
"agents" (a few lines of `urllib`, standing in for `flashnode work` — see
the test file's docstring for why an in-repo test cannot import `flashnode`
directly) pull leases, train, and commit without ever talking to each
other. The measured per-round mean loss across four rounds with two
participating nodes:

```
round 0  participants 2/2  mean_loss 0.5361
round 1  participants 2/2  mean_loss 0.3781
round 2  participants 2/2  mean_loss 0.2548
round 3  participants 2/2  mean_loss 0.1757
converged: 0.5361 -> 0.1757 over 4 rounds
```

(`scripts/fedavg_local_demo.py` reproduces this and exits non-zero if the
final round's loss is not below the first — a demo that prints numbers
nobody checks is not evidence.)

Read that number correctly: **this proves collaborative training, not
faster training.** Two nodes did not finish training in half the wall-clock
time of one — they trained *sequentially* through four rounds, each doing
its own local steps, and the loss came down because their independently
computed updates were combined. Nothing here claims a throughput or
speed-up result; DDP/FSDP make that claim, on a coordinated pool that can
rendezvous, and that claim is out of scope for volunteer nodes entirely (the
repo's `docs/guides/donate-a-machine.md` has the full list of things the
volunteer pool does not attempt, including "no coordinated multi-process
training"). What FedAvg proves is that machines which cannot see or trust
each other — and
in the volunteer case, cannot even reach each other over the network — can
still jointly move one model's loss in the right direction, coordinated
entirely through the coordinator's leases and artifact store.
