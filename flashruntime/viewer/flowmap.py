"""The flow-map + KPI strip component, shared between the run viewer page
(`viewer/page.py`) and — in a later phase — the coordinator dashboard
(`service/dashboard.py`). Divergence between the two surfaces is a bug, so
the CSS and JS live here once, as strings the host page embeds inline.

Contract with the host page:
  * CSS colors are `%%token%%` placeholders resolved by the host's render()
    from `viewer.page.TOKENS` (the single source of truth).
  * The JS expects a global `T` (the injected token object) and three
    elements: `#kpis`, `#flowmap`, `#detail`.
  * The host polls its own data and calls `renderKpiTiles(tiles)`,
    `renderFlowmap(snapshot)`, `renderDetail(snapshot)` on each tick. The
    component never fetches — data acquisition is the host's business.

The map draws machine → workers (attempts) → ranks as DOM nodes (free click
targets + text layout) with one SVG layer behind them for the edges. It is
rebuilt idempotently from each snapshot; `fmSelected` (the clicked node)
survives rebuilds and feeds the slide-in detail panel with live values.

Single-machine today, on purpose: the machine column renders exactly one
node — the local viewer's reality, since `flash.submit(watch=True)` only
ever observes the machine it is running on. Growing that to N machines for
the coordinator dashboard (phase 3) is a change to THIS component — extend
the machine column here so both surfaces gain it together, never fork a
per-host copy of the flow map. Before that adoption lands, the coordinator
dashboard also needs two host-page prerequisites it does not have yet: a
`T` token global (the injected color-token object this JS reads, see
`viewer/page.py`'s `%%tokens_json%%`) and a `%%token%%` substitution pass
over its CSS — `service/dashboard.py`'s current `_recolor` hex pipeline is a
different mechanism and must be reworked onto the same placeholder scheme
before it can host this component.
"""

from __future__ import annotations

FLOWMAP_CSS = r"""
  /* KPI strip -------------------------------------------------------------- */
  #kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; }
  .kpi { background: %%panel%%; border: 1px solid %%border%%; border-radius: 8px; padding: 8px 12px; }
  .kpi-label { color: %%muted%%; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  .kpi-value { color: %%text_bright%%; font-size: 18px; font-weight: 600; margin-top: 2px; }
  .kpi-hint { color: %%muted%%; font-size: 10px; margin-top: 2px; }

  /* flow map ---------------------------------------------------------------- */
  #flowmap { position: relative; display: flex; gap: 48px; padding: 14px;
             min-height: 180px; background: %%bg_inset%%; border-radius: 6px;
             overflow-x: auto; }
  #fm-edges { position: absolute; inset: 0; pointer-events: none; }
  .fm-col { display: flex; flex-direction: column; gap: 10px; justify-content: center;
            position: relative; z-index: 1; min-width: 180px; }
  .fm-node { background: %%panel%%; border: 1px solid %%border%%; border-radius: 8px;
             padding: 8px 10px; cursor: pointer; }
  .fm-node:hover { filter: brightness(1.25); }
  .fm-selected { outline: 2px solid %%running%%; outline-offset: 1px; }
  .fm-machine { border-color: %%running%%; }
  .fm-done { opacity: .55; }
  .fm-live .fm-title { animation: fm-pulse 2s ease-in-out infinite; }
  @keyframes fm-pulse { 50% { opacity: .55; } }
  .fm-title { color: %%text_bright%%; }
  .fm-sub { color: %%muted%%; font-size: 11px; margin-top: 2px; }
  .fm-badge { color: %%warn%%; border: 1px solid %%warn%%; border-radius: 4px;
              font-size: 10px; padding: 0 4px; }
  .fm-more { color: %%muted%%; font-size: 11px; }

  /* detail panel (slides in over the map's right edge) ---------------------- */
  #detail { position: absolute; top: 8px; right: 8px; bottom: 8px; width: min(360px, 60%);
            background: %%panel%%; border: 1px solid %%running%%; border-radius: 8px;
            padding: 14px; overflow-y: auto; z-index: 2; }
  #detail h3 { color: %%text_bright%%; font-size: 13px; margin-bottom: 10px;
               text-transform: none; letter-spacing: 0; }
  .fm-kv { display: flex; justify-content: space-between; gap: 12px; padding: 3px 0;
           border-bottom: 1px solid %%border%%; }
  .fm-kv span { color: %%muted%%; }
  .fm-kv b { color: %%text%%; font-weight: 600; text-align: right; word-break: break-all; }
  .fm-close { position: absolute; top: 8px; right: 8px; background: none;
              border: 1px solid %%border%%; color: %%muted%%; border-radius: 6px;
              cursor: pointer; padding: 2px 8px; font: inherit; }
  .fm-close:hover { color: %%text_bright%%; }
  .fm-log { margin-top: 10px; padding: 8px; background: %%bg_inset%%; border-radius: 6px;
            white-space: pre-wrap; word-break: break-word; color: %%text%%;
            font-size: 11px; max-height: 200px; overflow-y: auto; }
  .fm-hint { margin-top: 10px; color: %%warn%%; font-size: 11px; }
"""

FLOWMAP_JS = r"""
// ==== flow map + KPI components (viewer/flowmap.py — shared surface) =======
// Requires: global `T` (color tokens) and elements #kpis, #flowmap, #detail.
// The host page owns polling; these functions only render a given snapshot.

const fmEsc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Lifecycle state → accent. The shared vocabulary both surfaces use.
function fmStateColor(s) {
  switch (s) {
    case "RUNNING": return T.running;                  // cyan
    case "LEASED": case "RECOVERING": return T.warn;   // amber
    case "SUCCEEDED": case "COMPLETED": return T.ok;   // green
    case "FAILED": return T.fail;                      // red
    default: return T.muted;                           // PENDING / CANCELLED
  }
}

function fmFmtBytes(n) {
  if (typeof n !== "number" || !isFinite(n)) return "—";
  const gb = n / (1024 ** 3);
  return gb >= 1 ? gb.toFixed(1) + " GB" : Math.round(n / (1024 ** 2)) + " MB";
}

// ---- KPI strip -------------------------------------------------------------
// tiles: [{label, value, hint?, color?}]. A tile with no data source shows
// "—" (the host builds tiles that way) — the strip never invents a number.
function renderKpiTiles(tiles) {
  const el = document.getElementById("kpis");
  el.innerHTML = tiles.map((t) =>
    '<div class="kpi"><div class="kpi-label">' + fmEsc(t.label) + "</div>" +
    '<div class="kpi-value"' + (t.color ? ' style="color:' + fmEsc(t.color) + '"' : "") + ">" +
    fmEsc(t.value) + "</div>" +
    (t.hint ? '<div class="kpi-hint">' + fmEsc(t.hint) + "</div>" : "") +
    "</div>"
  ).join("");
}

// ---- flow map ---------------------------------------------------------------
// Selection survives re-renders: nodes are keyed {kind, id} and the map is
// rebuilt from scratch on every snapshot (idempotent — no DOM bookkeeping).
let fmSelected = null; // {kind: "machine"|"worker"|"rank", id: string} | null

// newest telemetry sample carried by one attempt row (or null)
function fmLastSample(a) {
  const tel = a.telemetry || [];
  return tel.length ? tel[tel.length - 1] : null;
}

function renderFlowmap(s) {
  const el = document.getElementById("flowmap");
  const attempts = s.attempts || [];
  const machine = (s.monitor && s.monitor.machine) || null;

  // machine column — one node (multi-machine arrives with the coordinator)
  const mSel = fmSelected && fmSelected.kind === "machine";
  const host = machine ? machine.hostname : "localhost";
  const mSub = machine
    ? (machine.cpu_count || "?") + " cores" +
      (machine.cpu_percent != null ? " · cpu " + Math.round(machine.cpu_percent) + "%" : "") +
      (machine.gpus && machine.gpus.length ? " · " + machine.gpus.length + " gpu" : "")
    : "no telemetry yet";
  let html = '<div class="fm-col">' +
    '<div class="fm-node fm-machine' + (mSel ? " fm-selected" : "") +
    '" data-kind="machine" data-id="machine">' +
    '<div class="fm-title">▣ ' + fmEsc(host) + "</div>" +
    '<div class="fm-sub">' + fmEsc(mSub) + "</div></div></div>";

  // worker column — one node per attempt, newest kept (cap so a 100-trial
  // sweep stays readable; the count of hidden ones is stated, not silent)
  const shown = attempts.slice(-16);
  html += '<div class="fm-col">' + (attempts.length > shown.length
    ? '<div class="fm-more">… ' + (attempts.length - shown.length) + " earlier attempts</div>" : "");
  for (const a of shown) {
    const sel = fmSelected && fmSelected.kind === "worker" && fmSelected.id === a.attempt_id;
    const color = fmStateColor(a.state);
    const respawn = /-r\d+$/.test(a.attempt_id || "");
    const mark = a.state === "RUNNING" ? "●" : a.state === "SUCCEEDED" ? "✓"
      : a.state === "FAILED" ? "✗" : "○";
    html += '<div class="fm-node fm-worker' + (sel ? " fm-selected" : "") +
      (a.state === "RUNNING" ? " fm-live" : " fm-done") + '"' +
      ' data-kind="worker" data-id="' + fmEsc(a.attempt_id) + '" style="border-color:' + color + '">' +
      '<div class="fm-title" style="color:' + color + '">' + mark + " " + fmEsc(a.attempt_id) +
      (respawn ? ' <span class="fm-badge">⟳ respawn</span>' : "") + "</div>" +
      '<div class="fm-sub">pid ' + fmEsc(a.pid) + " · " + fmEsc(a.state) + "</div></div>";
  }
  html += "</div>";

  // rank column — heartbeat ranks per attempt; for a live uninstrumented
  // attempt, fall back to its process tree's children (pid-only nodes)
  html += '<div class="fm-col">';
  for (const a of shown) {
    const ranks = a.ranks || [];
    const sample = fmLastSample(a);
    const procs = (sample && sample.processes) || [];
    const byPid = {};
    for (const p of procs) byPid[p.pid] = p;
    if (ranks.length) {
      for (const r of ranks) {
        const rid = a.attempt_id + "/" + r.rank;
        const sel = fmSelected && fmSelected.kind === "rank" && fmSelected.id === rid;
        const proc = byPid[r.pid];
        html += '<div class="fm-node fm-rank' + (sel ? " fm-selected" : "") +
          (a.state === "RUNNING" ? " fm-live" : " fm-done") + '"' +
          ' data-kind="rank" data-id="' + fmEsc(rid) + '" data-worker="' + fmEsc(a.attempt_id) + '">' +
          '<div class="fm-title">rank ' + fmEsc(r.rank) + " · pid " + fmEsc(r.pid) + "</div>" +
          '<div class="fm-sub">' + fmEsc(r.device || "?") +
          (r.step != null ? " · step " + fmEsc(r.step) : "") +
          (proc && proc.cpu_percent != null ? " · cpu " + Math.round(proc.cpu_percent) + "%" : "") +
          "</div></div>";
      }
    } else if (a.state === "RUNNING" && procs.length > 1) {
      for (const p of procs.slice(1, 9)) { // [0] is the launched root itself
        const rid = a.attempt_id + "/pid-" + p.pid;
        const sel = fmSelected && fmSelected.kind === "rank" && fmSelected.id === rid;
        html += '<div class="fm-node fm-rank fm-live' + (sel ? " fm-selected" : "") + '"' +
          ' data-kind="rank" data-id="' + fmEsc(rid) + '" data-worker="' + fmEsc(a.attempt_id) + '">' +
          '<div class="fm-title">pid ' + fmEsc(p.pid) + "</div>" +
          '<div class="fm-sub">' + fmEsc(p.cmd || "") +
          (p.cpu_percent != null ? " · cpu " + Math.round(p.cpu_percent) + "%" : "") +
          "</div></div>";
      }
    }
  }
  html += "</div>";

  el.innerHTML = html + '<svg id="fm-edges"></svg>';

  el.querySelectorAll(".fm-node").forEach((node) => {
    node.addEventListener("click", () => {
      fmSelected = { kind: node.dataset.kind, id: node.dataset.id };
      renderFlowmap(s); // re-render for the selection outline
      renderDetail(s);
    });
  });

  fmDrawEdges(el);
}

// Bézier connectors machine→worker and worker→its ranks, drawn into the SVG
// layer AFTER the DOM has laid the nodes out (positions read back from
// getBoundingClientRect, so the lines are correct at any width).
function fmDrawEdges(el) {
  const svg = el.querySelector("#fm-edges");
  const box = el.getBoundingClientRect();
  svg.setAttribute("width", box.width);
  svg.setAttribute("height", box.height);
  const anchor = (n, side) => {
    const r = n.getBoundingClientRect();
    return { x: (side === "r" ? r.right : r.left) - box.left,
             y: r.top + r.height / 2 - box.top };
  };
  const curve = (a, b) =>
    '<path d="M' + a.x + " " + a.y +
    " C" + ((a.x + b.x) / 2) + " " + a.y + "," +
    ((a.x + b.x) / 2) + " " + b.y + "," + b.x + " " + b.y +
    '" stroke="' + T.border + '" fill="none" stroke-width="1"/>';
  const machine = el.querySelector(".fm-machine");
  let lines = "";
  el.querySelectorAll(".fm-worker").forEach((w) => {
    if (machine) lines += curve(anchor(machine, "r"), anchor(w, "l"));
    el.querySelectorAll('.fm-rank[data-worker="' + CSS.escape(w.dataset.id) + '"]')
      .forEach((r) => { lines += curve(anchor(w, "r"), anchor(r, "l")); });
  });
  svg.innerHTML = lines;
}

// ---- detail panel -----------------------------------------------------------
// Re-rendered on every snapshot while open, so its numbers stay live.
function renderDetail(s) {
  const el = document.getElementById("detail");
  if (!fmSelected) { el.hidden = true; return; }
  const kv = (k, v) => '<div class="fm-kv"><span>' + fmEsc(k) + "</span><b>" + fmEsc(v) + "</b></div>";
  let body = "";
  if (fmSelected.kind === "machine") {
    const m = (s.monitor && s.monitor.machine) || {};
    body = "<h3>machine · " + fmEsc(m.hostname || "localhost") + "</h3>" +
      kv("cores", m.cpu_count != null ? m.cpu_count : "—") +
      kv("cpu", m.cpu_percent != null ? Math.round(m.cpu_percent) + "%" : "—") +
      kv("memory", m.mem_used != null ? fmFmtBytes(m.mem_used) + " / " + fmFmtBytes(m.mem_total) : "—") +
      kv("load avg", m.load_avg ? m.load_avg.map((x) => x.toFixed(2)).join("  ") : "—") +
      ((m.gpus || []).map((g, i) => kv("gpu " + i, g.name + " · " + Math.round(g.util_percent) +
        "% · " + Math.round(g.mem_used_mb) + "/" + Math.round(g.mem_total_mb) + " MB")).join("")) +
      (m.limited ? '<div class="fm-hint">full stats: pip install "flashruntime[monitor]"</div>' : "");
  } else {
    const attemptId = fmSelected.kind === "worker" ? fmSelected.id : fmSelected.id.split("/")[0];
    const a = (s.attempts || []).find((x) => x.attempt_id === attemptId);
    if (!a) { el.hidden = true; return; }
    if (fmSelected.kind === "worker") {
      const dur = a.finished_at
        ? (a.finished_at - a.started_at).toFixed(1) + "s"
        : ((Date.now() / 1000) - a.started_at).toFixed(0) + "s so far";
      body = "<h3>" + fmEsc(a.attempt_id) + "</h3>" +
        kv("state", a.state) + kv("pid", a.pid) + kv("job", a.job_id) + kv("runtime", dur) +
        '<div class="fm-log">' +
        fmEsc((a.log_tail || "").split("\n").slice(-12).join("\n") || "no log yet") + "</div>";
    } else {
      const key = fmSelected.id.split("/")[1];
      const sample = fmLastSample(a);
      const procs = (sample && sample.processes) || [];
      const isPid = key.startsWith("pid-");
      const r = isPid ? null : (a.ranks || []).find((x) => String(x.rank) === key);
      const pid = isPid ? Number(key.slice(4)) : r && r.pid;
      const proc = procs.find((p) => p.pid === pid);
      body = "<h3>" + (r ? "rank " + fmEsc(r.rank) : "pid " + fmEsc(pid)) + "</h3>" +
        (r ? kv("device", r.device || "—") + kv("backend", r.backend || "—") +
             kv("world size", r.world_size) + kv("step", r.step != null ? r.step : "—") : "") +
        kv("pid", pid != null ? pid : "—") +
        kv("cpu", proc && proc.cpu_percent != null ? Math.round(proc.cpu_percent) + "%" : "—") +
        kv("memory", proc ? fmFmtBytes(proc.rss_bytes) : "—") +
        kv("status", proc ? (proc.status || "—") : a.state);
    }
  }
  el.innerHTML = '<button id="fm-close" class="fm-close">✕</button>' + body;
  el.hidden = false;
  el.querySelector("#fm-close").addEventListener("click", () => {
    fmSelected = null;
    el.hidden = true;
    renderFlowmap(s);
  });
}
"""
