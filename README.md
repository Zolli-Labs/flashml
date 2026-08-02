# FlashML

Train machine-learning models across ordinary machines — laptops, spare
workstations, whatever is idle — and keep going when one of them disappears.

A machine that vanishes mid-task is the normal case here, not an incident. Work
is never pushed to a machine: a machine *claims* a time-limited lease, proves it
is alive with heartbeats, and only the first valid result is accepted. When a
lease expires, the task returns to the queue and somebody else picks it up from
the last checkpoint.

This repository holds the open parts of FlashML. All of it is Apache-2.0.

## What is in here

| Directory | Package | What it is |
|---|---|---|
| [`flashruntime/`](flashruntime/) | `flashruntime` | The protocol and the fault-tolerant runtime: strategy planning, leases, checkpointing, and recovery. Depends on neither of the others. |
| [`flashnode/`](flashnode/) | `flashnode` | The host agent. Install it on a machine you want to contribute; it registers, claims work, and runs each task in an isolated sandbox with networking disabled. |
| [`examples/federated/`](examples/federated/) | — | A PyTorch model trained across several machines by federated averaging. Includes `simulate.py`, which rehearses the whole loop locally so an encoding mistake fails in seconds instead of after five rounds on other people's laptops. |

`flashnode` depends on `flashruntime`. Nothing else crosses that boundary.

## Contributing a machine

> **Not yet — the packages are not on PyPI at the time of writing.** The install
> becomes a single command with the first release; until then, use the
> instructions shown in the FlashML console, which install from this repository
> directly.

Once released:

```bash
python3 -m venv flashml
flashml/bin/python -m pip install flashnode
flashml/bin/flashnode login --coordinator <your-coordinator-url>
```

The virtual environment is not ceremony. macOS ships no `pip` on `PATH`, and
Homebrew's Python refuses to install into itself under
[PEP 668](https://peps.python.org/pep-0668/). A venv sidesteps both and leaves
the system Python untouched — which matters when you are asking someone to run
code on their own machine.

## Running the runtime yourself

`flashruntime` is usable on its own, with no account and no cloud service. It
runs on SQLite and a local directory, and the test suite needs no
infrastructure:

```bash
cd flashruntime
uv venv .venv && uv pip install -e ".[dev,service]"
.venv/bin/pytest -q
```

The `service` extra is required for the test suite, not optional — eight test
modules import the HTTP service.

## The managed service

Zolli Labs runs a hosted control plane that schedules work across contributed
machines, handles accounts, and shows you what your jobs are doing. It is a
separate, private codebase and is not in this repository. Nothing here depends
on it: the runtime and the agent are usable, testable, and deployable without
it.

## Security

Task code runs inside a sandbox with networking disabled. If you find a way out
of that, or anything else security-relevant, please read
[`flashruntime/SECURITY.md`](flashruntime/SECURITY.md) rather than opening a
public issue.

## License

Apache-2.0. See [`flashruntime/LICENSE`](flashruntime/LICENSE).
