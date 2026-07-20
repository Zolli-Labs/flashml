# Delivery Phases

Each subfolder is a self-contained implementation brief. A contributor should
be able to open one phase folder and learn:

- the user-visible outcome;
- prerequisites and non-goals;
- required code and documentation;
- what exists and what is missing;
- how to verify the work;
- the exact exit gate.

State vocabulary:

| State | Meaning |
|---|---|
| `implemented` | Code exists and its local verification passes. |
| `current` | Highest-priority incomplete phase. |
| `ready` | Can be implemented without waiting for another incomplete phase. |
| `waiting` | Has an explicit dependency on an earlier phase. |
| `planned` | Scoped but not ready or prioritized. |
| `research` | Direction is recorded; interface/design commitment is intentionally absent. |

The authoritative live snapshot is [`../status/README.md`](../status/README.md).
