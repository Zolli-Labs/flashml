"""Translate a finished local process into recovery `FailureSignals`.

`from_local_launch(exit_code, log_tail)` is the first real caller of the
recovery package: it turns the only evidence a `LocalProcessLauncher` can
observe — the child's OS exit code and the tail of its captured stdout+stderr
— into the typed `FailureSignals` that `classify()` reads. It is a transparent
lookup table, not an inference engine: every rule is one `if` whose comment
names the reason the pattern implies its class, and the *first* matching rule
wins (order encodes priority — deterministic-bug evidence outranks the
transient default).

The four categories it maps, and why each is the honest reading:

- exit 0 / never started → neutral signals. Nothing broke, so classify()
  returns UNKNOWN; the retry loop only consults this on a FAILED launch, so
  this is a guard that keeps the function total, not a live path.

- a deterministic Python bug — a SyntaxError / ImportError / ModuleNotFoundError
  / NameError / AttributeError named in the log, or (barring the torchrun
  wrapper below) any unhandled exception whose "Traceback (most recent call
  last):" reached the log → APPLICATION-ERROR signal (`exit_deterministic`).
  The same import/parse/name failure recurs byte-for-byte on a retry, so the
  policy fails fast (FAIL_JOB) instead of spending a second attempt to reach
  the same certainty. Burning capacity on a bug is the most expensive
  "recovery" (policy.py).

- a signal death / OOM / bare `SystemExit` / any other unexplained nonzero
  exit → WORKER-CRASH signal (the transient default). The OS or an allocator
  killed the process, or the code exited nonzero without an unhandled
  exception — treated as bad luck, worth one fresh attempt. A genuine bug
  that slips through here re-crashes deterministically (printing a traceback)
  and is caught by the rule above on the next go-round.

One empirically necessary carve-out: torchrun / torch.distributed.elastic
wraps *every* worker death — transient crashes included — in a
`ChildFailedError` and prints ITS OWN traceback; the child's real traceback is
not in this log (the summary shows "error_file: <N/A>"). So a bare "Traceback"
line accompanied by that wrapper is the launcher's stack, carrying no evidence
about whether the user's failure was deterministic — it must NOT be read as an
application error, or a resumable crash would fail fast and never retry.

Deliberately narrow: it never fabricates node / accelerator / communication /
storage signals a single local process cannot actually evidence — those
classes belong to the distributed coordinator, not the local launcher.
"""

from __future__ import annotations

from flashruntime.recovery.taxonomy import FailureSignals

# Exception-type names that mark a DETERMINISTIC bug: the same failure recurs
# byte-for-byte on retry, so retrying only re-reaches it. Substring matches on
# the class name as printed in a Python traceback.
_DETERMINISTIC_EXC_MARKERS = (
    "SyntaxError",          # the file will not parse — a retry parses the same file
    "IndentationError",     # a parse error too — same input, same failure
    "ImportError",          # a missing/broken dependency does not heal on retry
    "ModuleNotFoundError",  # ImportError's subclass; its name lacks the substring "ImportError"
    "NameError",            # an undefined name is a code bug, not bad luck
    "AttributeError",       # a code / version-contract bug, deterministic by nature
)

# torchrun / elastic re-raises ChildFailedError (with its own traceback) for
# ANY worker death, transient ones included — see the module docstring. Its
# presence disqualifies the bare-traceback rule below.
_ELASTIC_WRAPPER_MARKER = "ChildFailedError"
_TRACEBACK_MARKER = "Traceback (most recent call last):"


def from_local_launch(exit_code: int | None, log_tail: str) -> FailureSignals:
    """Map a finished local process to the signals `classify()` reads.

    `exit_code` is the child's OS return code (None if it never started);
    `log_tail` is the tail of its captured stdout+stderr. Rules are checked
    top-to-bottom, first match wins — see the module docstring for the full
    table and the reasoning behind each class.
    """
    log = log_tail or ""

    # A clean exit is not a failure — nothing to recover. Guard only: the retry
    # loop never asks on SUCCEEDED, but this keeps from_local_launch total.
    if exit_code == 0 or exit_code is None:
        return FailureSignals(exit_code=exit_code)

    # A named deterministic bug (import / parse / name error) — the same error
    # reappears on a byte-identical retry, so mark it deterministic and let the
    # policy fail fast rather than pay for a re-run to the same failure.
    if any(marker in log for marker in _DETERMINISTIC_EXC_MARKERS):
        return FailureSignals(exit_code=exit_code, exit_deterministic=True)

    # An unhandled exception that printed its own traceback — deterministic by
    # default (unhandled exceptions are code errors). Excluded when it is the
    # torchrun elastic wrapper, whose traceback is the launcher's stack and
    # says nothing about whether the user's failure was deterministic.
    if _TRACEBACK_MARKER in log and _ELASTIC_WRAPPER_MARKER not in log:
        return FailureSignals(exit_code=exit_code, exit_deterministic=True)

    # Everything else: a signal death / OOM (SIGKILL/SIGSEGV, exit 137, a
    # negative POSIX returncode), a bare SystemExit with no traceback, or a
    # torchrun ChildFailedError — all transient. Leave exit_deterministic False
    # so classify() returns WORKER_CRASH and the policy grants a fresh attempt.
    return FailureSignals(exit_code=exit_code)
