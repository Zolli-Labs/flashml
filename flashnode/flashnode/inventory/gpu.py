"""What GPUs this host has, according to `nvidia-smi`.

WHY A SUBPROCESS AND NOT NVML. `pynvml`/`nvidia-ml-py` would be nicer to
call and is the obvious choice anywhere else. It is the wrong choice here:
this agent installs on volunteers' machines, and AGENTS.md is explicit that
every dependency is attack surface on someone else's computer. The driver
already ships `nvidia-smi`; a machine with a GPU we could use has it, and a
machine without one has nothing for NVML to bind to either.

THE ONE INVARIANT. `GpuInfo.index` is required and is never guessed. A row
this module cannot read an index out of is DROPPED. The alternative —
emitting a device with a defaulted index — puts a malformed entry in
`NodeRegistration`, which the coordinator rejects with a 422, and the host
goes offline entirely. One unreadable device must cost one device.

Everything here is best-effort and silent. `discover()` calls it on the
registration path, so a raise anywhere in this file is an agent that will not
start on a machine that was only ever going to run CPU work.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Callable

from flashruntime.protocol.v1alpha1 import GpuInfo

CommandRunner = Callable[..., subprocess.CompletedProcess]

#: What we ask the driver for, in order. The parser is positional, so this
#: tuple and `GpuInfo`'s fields are one contract — do not reorder it.
QUERY_FIELDS: tuple[str, ...] = (
    "index",
    "name",
    "memory.total",
    "driver_version",
    "compute_cap",
)

#: `compute_cap` postdates plenty of drivers still running on hobbyist
#: hardware, and nvidia-smi fails the WHOLE query on a field it does not
#: know rather than omitting that column. So the fallback is the same query
#: minus the newest field: losing one string beats telling a real GPU host
#: they have no GPU.
FALLBACK_FIELDS: tuple[str, ...] = QUERY_FIELDS[:-1]

#: A hung `nvidia-smi` must not hang registration. Short on purpose: this
#: reads driver state already in memory, it does not do work.
PROBE_TIMEOUT_S = 5

#: nvidia-smi's ways of saying "the driver would not tell me". Kept as
#: values rather than mapped to the literal text, because `driver_version =
#: "[N/A]"` is worse than `driver_version = ""` — it looks like data.
_PLACEHOLDERS = frozenset(
    {"", "n/a", "[n/a]", "not supported", "[not supported]", "unknown", "[unknown]"}
)


def probe_gpus(run: CommandRunner | None = None) -> list[GpuInfo]:
    """Every GPU this host's driver reports. `[]` for anything else.

    `run` is a parameter so the whole suite exercises this with no driver
    present, exactly as `doctor.py` parameterises its subprocess calls, and
    it is resolved at CALL time rather than captured as a default argument —
    `doctor.check_cli_on_path` documents at length what the other form costs.
    """
    run = run or subprocess.run
    for fields in (QUERY_FIELDS, FALLBACK_FIELDS):
        try:
            gpus = _query(run, fields)
        except Exception:  # noqa: BLE001 - see the module docstring
            # Deliberately total. FileNotFoundError (no driver),
            # TimeoutExpired (hung), OSError (no permission), and anything a
            # future runner does wrong all mean the same thing to a caller
            # that is only trying to fill in one optional field. KeyboardInterrupt
            # and SystemExit are not Exceptions and still propagate: a
            # volunteer pressing ^C during registration must still stop it.
            return []
        if gpus:
            return gpus
    return []


def _query(run: CommandRunner, fields: Sequence[str]) -> list[GpuInfo]:
    proc = run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        timeout=PROBE_TIMEOUT_S,
        check=False,
    )
    if getattr(proc, "returncode", 1) != 0:
        return []
    return _parse(_text(getattr(proc, "stdout", None)))


def _text(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode(errors="replace")


def _cell(cells: Sequence[str], position: int) -> str:
    """One CSV column, or `""` — for a short row (an older driver returning
    fewer columns) and for a placeholder alike. Both mean "no value", and
    `GpuInfo` says nothing rather than guessing."""
    if position >= len(cells):
        return ""
    value = cells[position]
    return "" if value.lower() in _PLACEHOLDERS else value


def _megabytes(value: str) -> int | None:
    # `nounits` should make this a bare integer; tolerate "24564 MiB" anyway
    # rather than throwing away the whole reading over a formatting change.
    head = value.split()[0] if value.split() else ""
    try:
        mb = int(head)
    except ValueError:
        return None
    return mb if mb >= 0 else None


def _parse(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        try:
            index = int(cells[0])
        except (ValueError, IndexError):
            # THE invariant: no index, no device. This also swallows
            # nvidia-smi's prose error lines ("No devices were found",
            # "Unable to determine the device handle for GPU ...") without
            # having to pattern-match them.
            continue
        if index < 0:
            # Not something a driver emits. Treat it as garbage rather than
            # advertising a device whose identity we plainly misread.
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=_cell(cells, 1),
                memory_total_mb=_megabytes(_cell(cells, 2)),
                driver_version=_cell(cells, 3),
                compute_capability=_cell(cells, 4),
            )
        )
    return gpus
