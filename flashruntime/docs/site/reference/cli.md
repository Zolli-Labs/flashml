# Reference: CLI (`flashruntime`)

The `flashruntime` command ships with the `[service]` extra
(`pip install "flashruntime[service]"`). It is the terminal front door to the
same operations the SDK exposes: plan a job offline, run a command workload
locally, and talk to a coordinator.

The blocks below mirror the real `--help` output.

```
usage: flashruntime [-h] [--api API]
                    {plan,submit,submit-spec,status,events,logs,cancel} ...

positional arguments:
  {plan,submit,submit-spec,status,events,logs,cancel}
    plan                evaluate a PlanRequest offline and print the strategy
    submit              run a command workload locally (no API needed)
    submit-spec         POST a JobSpec YAML to the coordinator — was `submit`
                        before 0.1.0; renamed when `submit` became the local-
                        workload front door

options:
  -h, --help            show this help message and exit
  --api API             FlashRuntime API base URL
```

`--api` names the coordinator base URL for the service-side subcommands
(`submit-spec`, `status`, `events`, `logs`, `cancel`).

---

## `plan` — offline strategy selection

```
usage: flashruntime plan [-h] [--json] request_file

positional arguments:
  request_file  PlanRequest as .yaml or .json

options:
  -h, --help    show this help message and exit
  --json        emit the full PlanReport as JSON
```

Runs `flash.plan()` on a `PlanRequest` file and prints the explained strategy;
`--json` emits the full `PlanReport`. No cluster required.

---

## `submit` — run a command workload locally

```
usage: flashruntime submit [-h] [--source SOURCE] [--task-params TASK_PARAMS]
                           [--max-restarts MAX_RESTARTS]
                           [--output-dir OUTPUT_DIR] [--watch | --no-watch]
                           CMD

positional arguments:
  CMD                   the command to run, e.g. 'python train.py --lr {lr}'

options:
  -h, --help            show this help message and exit
  --source SOURCE       directory holding the user's code
  --task-params TASK_PARAMS
                        JSON list of param dicts for Mode A fan-out
  --max-restarts MAX_RESTARTS
                        automatic recovery budget
  --output-dir OUTPUT_DIR
                        where run.json and artifacts land (default: temp dir)
  --watch, --no-watch   open the live viewer (default: on at a terminal, off
                        in pipes/CI)
```

The terminal equivalent of `flash.submit()`. `--task-params` (a JSON list of
param dicts) fills `{name}` placeholders in `CMD` for a fan-out sweep;
`--max-restarts` is the automatic fault-tolerance budget.

---

## `submit-spec` — POST a JobSpec to the coordinator

```
usage: flashruntime submit-spec [-h] spec_file

positional arguments:
  spec_file

options:
  -h, --help  show this help message and exit
```

POSTs a `JobSpec` YAML/JSON file (e.g. one produced by
`workloads.command.to_jobspec`) to the coordinator named by `--api`. This was
called `submit` before 0.1.0 — it was renamed when local `submit` became the
default front door.

---

## `status` / `events` / `logs` / `cancel` — inspect a coordinator job

```
usage: flashruntime status [-h] job_id
usage: flashruntime events [-h] job_id
usage: flashruntime logs   [-h] job_id
usage: flashruntime cancel [-h] job_id
```

Each takes a `job_id` and queries the coordinator at `--api`: `status` for the
derived job state, `events` for the append-only ledger, `logs` for captured
output, and `cancel` to stop a job.

---

See the [JobSpec & isolation guide](../guides/jobspec-and-isolation.md) for the
wire form these service subcommands operate on, and the [SDK reference](sdk.md)
for the in-process equivalents.
