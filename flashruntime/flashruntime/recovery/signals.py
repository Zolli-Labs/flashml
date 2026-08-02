"""Translate a finished local process into recovery `FailureSignals`.

`from_local_launch(exit_code, log_tail)` is the first real caller of the
recovery package: it turns the only evidence a `LocalProcessLauncher` can
observe — the child's OS exit code and the tail of its captured stdout+stderr
— into the typed `FailureSignals` that `classify()` reads. It is a transparent
lookup table, not an inference engine: every rule is one `if` whose comment
names the reason the pattern implies its class, and the *first* matching rule
wins (order encodes priority — deterministic-bug evidence outranks the
transient default).

Rule table (checked top-to-bottom, first match wins). One rule, one reason:

  1. exit 0 / never started        → neutral signals (nothing broke; a guard
                                      that keeps the function total).
  2. a named deterministic
     exception on a traceback
     TERMINAL line                 → APPLICATION-ERROR (`exit_deterministic`):
                                      import/parse/name errors recur
                                      byte-for-byte, so fail fast, don't pay
                                      for a re-run to the same certainty.
  3. a bare "Traceback" that is
     NOT the torchrun wrapper       → APPLICATION-ERROR: an unhandled exception
                                      is a code error by default.
  4. everything else (signal death
     / OOM / bare SystemExit /
     torchrun ChildFailedError)    → WORKER-CRASH (the transient default):
                                      bad luck, worth one fresh attempt.

Why rule 2 is anchored to a traceback TERMINAL line, not a bare substring:
CPython prints the failing exception type at the START of the traceback's
final line — `ModuleNotFoundError: ...`, or dotted with its defining module,
`pkg.mod.SomeError: ...` — whereas prose only ever *mentions* an exception
name mid-line. The canonical false positive is the startup warning
`ImportError: flash_attn not available, falling back to eager attention`,
which real loggers emit behind a timestamp/level prefix
(`2026-… WARNING ImportError: …`) so the name never begins the line. A plain
`substring in log` scan read that prose as a deterministic bug and failed the
job fast — and because fail-fast is terminal, that was the worst-direction
false positive: a transient crash that would have resumed instead never
retried. Line-anchoring keeps every genuine traceback terminal (rule 2 still
fires on the real thing) while ignoring the name wherever it appears in prose.

Interaction of rules 2, 3 and the torchrun carve-out (precedence, explicit):
torchrun / torch.distributed.elastic wraps *every* worker death — transient
crashes included — in a `ChildFailedError` and prints ITS OWN traceback; by
default the child's real traceback is not in this log (`error_file: <N/A>`).
That wrapper terminal line is `…errors.ChildFailedError:` — `ChildFailedError`
is not in the deterministic marker set, so rule 2 never fires on the wrapper.
The wrapper's bare "Traceback" line WOULD trip rule 3, so rule 3 is
disqualified whenever `ChildFailedError` is present, letting the death fall
through to the transient default (rule 4) — which is exactly what the
kill-and-resume e2e depends on. Rule 2 is intentionally NOT gated by the
wrapper: a genuine child terminal line (e.g. a real `ModuleNotFoundError:` if
elastic error-files are enabled) is a true deterministic bug and should fail
fast even under torchrun.

Deliberately narrow: it never fabricates node / accelerator / communication /
storage signals a single local process cannot actually evidence — those
classes belong to the distributed coordinator, not the local launcher.
"""

from __future__ import annotations

import re

from flashruntime.recovery.taxonomy import FailureSignals

# Exception-type names that mark a DETERMINISTIC bug: the same failure recurs
# byte-for-byte on retry, so retrying only re-reaches it.
_DETERMINISTIC_EXC_MARKERS = (
    "SyntaxError",          # the file will not parse — a retry parses the same file
    "IndentationError",     # a parse error too — same input, same failure
    "ImportError",          # a missing/broken dependency does not heal on retry
    "ModuleNotFoundError",  # ImportError's subclass; its name lacks the substring "ImportError"
    "NameError",            # an undefined name is a code bug, not bad luck
    "AttributeError",       # a code / version-contract bug, deterministic by nature
)

# A marker only counts when it stands where CPython prints the failing type: at
# the START of the traceback's final line, either bare (`NameError`) or with a
# message (`NameError: ...`), optionally dotted with the defining module
# (`pkg.mod.NameError: ...`). Line-anchoring (see module docstring) is the fix
# for the incidental-substring false positive — exception names in prose don't
# start lines; traceback terminals always do. re.MULTILINE makes ^/$ match at
# every line boundary; the trailing `(?::|$)` boundary stops a marker from
# matching a longer name it merely prefixes (e.g. `NameErrorX:`).
_DETERMINISTIC_TERMINAL_RE = re.compile(
    r"^(?:\w+\.)*(?:" + "|".join(_DETERMINISTIC_EXC_MARKERS) + r")(?::|$)",
    re.MULTILINE,
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

    # Rule 1 — a clean exit is not a failure; nothing to recover. Guard only:
    # the retry loop never asks on SUCCEEDED, but this keeps the function total.
    if exit_code == 0 or exit_code is None:
        return FailureSignals(exit_code=exit_code)

    # Rule 2 — a named deterministic bug (import / parse / name error) standing
    # on a traceback TERMINAL line: the same error reappears on a byte-identical
    # retry, so mark it deterministic and let the policy fail fast. Anchored to
    # the line start (not a bare substring) so an exception name mentioned in a
    # prose log line — e.g. `… WARNING ImportError: flash_attn not available` —
    # is not mistaken for the failure that actually killed the process.
    if _DETERMINISTIC_TERMINAL_RE.search(log):
        return FailureSignals(exit_code=exit_code, exit_deterministic=True)

    # Rule 3 — an unhandled exception that printed its own traceback:
    # deterministic by default (unhandled exceptions are code errors). Excluded
    # when it is the torchrun elastic wrapper, whose traceback is the launcher's
    # stack and says nothing about whether the user's failure was deterministic.
    if _TRACEBACK_MARKER in log and _ELASTIC_WRAPPER_MARKER not in log:
        return FailureSignals(exit_code=exit_code, exit_deterministic=True)

    # Rule 4 — everything else: a signal death / OOM (SIGKILL/SIGSEGV, exit 137,
    # a negative POSIX returncode), a bare SystemExit with no traceback, or a
    # torchrun ChildFailedError — all transient. Leave exit_deterministic False
    # so classify() returns WORKER_CRASH and the policy grants a fresh attempt.
    return FailureSignals(exit_code=exit_code)
