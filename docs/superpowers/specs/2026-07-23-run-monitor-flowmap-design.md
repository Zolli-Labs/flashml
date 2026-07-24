# Run monitor: process flow map + monitoring dashboard

**Date:** 2026-07-23
**Status:** approved direction (layout A — "mission control"), spec for implementation
**Surfaces:** local run viewer (`flashruntime/viewer/`) first; coordinator dashboard
(`service/dashboard.py`) adopts the shared component in a follow-up phase.

## Goal

Turn the live run page into a real-time **process flow map** — machine → worker
processes → ranks, with PIDs, spawn/finish/respawn shown live — plus a
**monitoring dashboard strip** (elapsed, step progress, throughput, step
latency, CPU/memory/GPU usage, restart budget, checkpoint count). Clicking any
node opens a slide-in detail panel with that node's real-time information.

Decisions already made with the user:

1. **Scope:** one shared flow-map/dashboard component used by BOTH the local
   run viewer and the coordinator dashboard. Viewer ships first.
2. **Resource telemetry:** optional `psutil` (`[monitor]` extra), graceful
   degradation without it (stdlib `os.cpu_count()`, `os.getloadavg()`; missing
   numbers shown as "install flashruntime[monitor]", never faked).
3. **Rank visibility:** layered — process-tree discovery (works for any
   command) + per-rank heartbeat files written by the `flashruntime.torch`
   helper (adds rank↔PID mapping, device, current step when instrumented).
4. **Cost:** OUT of scope for this iteration (no honest data source yet).
5. **Layout:** option A — KPI tiles across the top, full-width flow map below,
   clicking a node slides a detail panel in from the right over the map;
   charts (loss, resource usage) and the existing sections below.

## Non-goals

- No cost estimation (no rate source; a made-up number violates the
  honest-results rule).
- No multi-machine rendezvous work; the map's *data model* supports N
  machines, but the local viewer will only ever see one today. The
  coordinator dashboard phase is where multiple machines appear.
- No framework/CDN/build step — the pages stay stdlib-rendered, fully inline,
  loopback-only (existing viewer contract).
- No change to the `viewer_v1` contract semantics: all new data is **additive**
  (new enrichment files read from disk by `state.collect()`; `run.json` gains
  no required fields).

## Architecture (three components)

```
flashruntime/monitor/          NEW — telemetry collection (no UI)
  sampler.py                   ResourceSampler: daemon thread, ~2 s period
  __init__.py

flashruntime/torch/__init__.py MOD — per-rank heartbeat writes

flashruntime/viewer/
  state.py                     MOD — enrich snapshot with telemetry + ranks
  flowmap.py                   NEW — shared flow-map + KPI component (CSS/JS
                               emitted as Python strings, house TOKENS)
  page.py                      MOD — layout A page composed from flowmap.py
service/dashboard.py           MOD (phase 3) — adopts flowmap.py
```

### 1. Telemetry: `flashruntime/monitor/sampler.py`

`ResourceSampler(output_dir, root_pid, period_s=2.0)` — a daemon thread the
SDK starts per launched attempt (started in `sdk._drive` right after
`launcher.launch(...)`, stopped when the attempt settles). Each tick it:

- **Machine sample:** hostname, cpu_count, load_avg (1/5/15), cpu_percent,
  mem_total/mem_used, plus GPU list (name, util %, mem used/total) when
  available. GPU source: `torch.cuda` if importable else `nvidia-smi
  --query-gpu` subprocess, else empty list. CPU/mem percents come from
  psutil; without psutil those fields are `null` and the sample carries
  `"limited": true`.
- **Process tree:** walk children of `root_pid` (psutil), one record per
  process: pid, ppid, cmdline (first 3 args), cpu_percent, rss_bytes,
  create_time, status. Without psutil: only the root pid with `null` stats.
- Appends one JSON line `{ts, machine, processes}` to
  `<attempt_output_dir>/telemetry.jsonl` (the viewer already tail-reads
  bounded windows — same pattern as metrics.jsonl).

Rules: import psutil lazily inside the thread (core stays pydantic-only —
verified by the existing clean-venv smoke); the sampler must be total (any
exception in a tick is swallowed, the thread never kills a run); writes are
plain appends (torn last line is already handled by the reader).

New extra in `pyproject.toml`: `monitor = ["psutil>=5.9"]`.

### 2. Rank heartbeats: `flashruntime/torch`

Each rank writes `<FLASHML_OUTPUT_DIR>/ranks/rank-<n>.json` via atomic
`os.replace` (same idiom as run.json):

```json
{"rank": 1, "local_rank": 1, "pid": 4316, "device": "cpu",
 "backend": "gloo", "world_size": 2, "step": 380, "ts": 1784850000.0}
```

Write points (every rank, not just rank 0): once in `prepare()` (step =
restored step), then refreshed inside `checkpoint()` and `log_metrics()` —
throttled to at most one write per second per rank. No new user-facing verb:
instrumented scripts get rank rows for free; the ADR-0003 capability
guardrail is untouched (this is observability, not orchestration).

### 3. Snapshot enrichment: `viewer/state.py`

`collect()` gains, per attempt (all total/degrading, same as today):

- `telemetry`: last ~150 samples of `telemetry.jsonl` (bounded tail window) —
  enough for a 5-minute usage chart at 2 s period.
- `ranks`: the parsed `ranks/rank-*.json` files (skip torn/invalid ones).

And one top-level `monitor` object: the newest machine sample across live
attempts (the dashboard tiles read this), including the `limited` flag.

### 4. Shared UI component: `viewer/flowmap.py`

Emits three Python string constants consumed by both pages:
`FLOWMAP_CSS`, `FLOWMAP_JS`, and `KPI_CSS/KPI_JS` (or one combined pair).
House rules: colors only from `viewer.page.TOKENS`, plain commented JS, no
framework, no external assets.

**KPI strip** (top): elapsed, run state badge, current step + steps/s
(derived from rank heartbeats when present, else metrics.jsonl deltas), step
latency (ms/step, inverse of steps/s), machine CPU %, memory used/total, GPU
util (only when a GPU exists), restarts used/budget, verified-checkpoint
count. Every tile renders "—" when its source is absent; tiles never invent
numbers. When `monitor.limited`, the CPU/mem tiles show an "install
flashruntime[monitor]" hint.

**Flow map** (full width): DOM nodes + one absolutely-positioned SVG layer
for edges (DOM nodes give free click targets and text layout; SVG draws the
connectors — no canvas hit-testing to maintain). Three columns:

- **machine** node: hostname, core count, live CPU/mem bar.
- **worker** nodes: one per attempt (`torchrun pid 4312 · RUNNING`), colored
  by state (running cyan / succeeded green / failed red / recovering amber),
  respawned attempts (`-rN`) badged ⟳ and linked to their predecessor,
  finished ones dimmed with ✓/✗ and their runtime.
- **rank** nodes: per live worker, from rank heartbeats (rank, pid, device,
  step) merged with process-tree stats matched by pid (cpu %, rss). Without
  heartbeats, the process tree's child processes render as unnamed worker
  processes (pid + cpu/rss only).

Nodes appear/disappear as the 2 s poll delivers new state — a respawn shows
up as a new node the next tick. The map re-renders idempotently from each
snapshot (no incremental DOM bookkeeping).

**Detail panel** (slide-in from the right, over the map, dismissable):
renders the selected node's full record — machine: all samples' latest +
sparkline history; worker/attempt: pid, job_id, state, started/finished,
exit-relevant events, log tail; rank: heartbeat fields + matched process
stats. Selection persists across polls (keyed by node id) and updates live.

### 5. Page assembly: `viewer/page.py` (layout A)

Order top-to-bottom: header (kept) → KPI strip → flow map (+detail panel) →
charts row: loss curve (kept) + NEW resource chart (CPU % and memory over
time from telemetry) → checkpoint timeline (kept) → events feed (kept) →
collapsible logs (kept). The old small topology canvas is REPLACED by the
flow map.

### 6. Coordinator dashboard adoption: `service/dashboard.py` (phase 3)

The dashboard swaps its node/lease listing for the same flow-map component:
machines = registered nodes (from `/nodes`), workers = leased attempts, rank
column absent (the coordinator doesn't see inside processes today). KPI
strip shows coordinator-level facts (nodes online, leases in flight,
expiries swept). Reuses `FLOWMAP_CSS/JS` verbatim — divergence between the
two surfaces is a bug.

## Error handling

- Everything the viewer reads stays TOTAL: torn JSON, missing dirs, absent
  psutil, no GPU — all degrade to partial snapshots, never exceptions
  (existing `state.py` doctrine extends to the new readers).
- The sampler thread swallows per-tick exceptions and must never affect the
  run (same "watched run is never interrupted" rule).
- Heartbeat writes are best-effort: an unwritable output dir must not crash
  training (wrap in try/except OSError).

## Testing

- `tests/test_monitor_sampler.py` — sampler with a fake psutil module
  injected (and with psutil absent): sample shape, `limited` flag, totality
  (a raising fake never propagates), jsonl appends.
- `tests/test_torch_helper.py` (extend) — heartbeat file appears after
  prepare(), refreshes on checkpoint/log_metrics, throttling, atomicity
  (no partial JSON), unwritable dir doesn't raise.
- `tests/test_viewer_state.py` (extend) — enrichment: telemetry tail, ranks
  parsing, torn-file skips, `monitor` top-level object.
- `tests/test_viewer_page.py` (extend) — page contains KPI/flow-map markup
  hooks, no external URLs (existing zero-CDN assertion), old sections kept.
- Manual acceptance: `examples/demo_pytorch_ddp.py` with `watch=True` on the
  Mac shows machine → torchrun → 2 ranks with PIDs, live CPU/mem, a kill
  mid-run shows the ⟳ respawned attempt and the detail panel explains the
  recovery decision.

## Phasing

1. **Telemetry** (monitor/ + torch heartbeats + state.py enrichment + tests).
2. **Viewer page** (flowmap.py + page.py layout A + tests).
3. **Dashboard adoption** (service/dashboard.py; can ship separately).

Phases 1–2 are this implementation plan; phase 3 is a follow-up once 1–2
land and look right in the demo.
