"""The coordinator's built-in status page.

One self-contained HTML page at GET / — no build step, no external assets,
no framework: it polls the same JSON API everything else uses
(/v1alpha1/nodes, /v1alpha1/jobs, /v1alpha1/jobs/{id}/{tasks,events,artifacts})
every 2 seconds. This is the self-hosted profile's window into the system;
FlashML Cloud's Next.js dashboard is the managed superset of the same data.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FlashRuntime — local coordinator</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body { font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
         background: #0d1117; color: #c9d1d9; padding: 24px; }
  h1 { font-size: 15px; margin-bottom: 4px; color: #e6edf3; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
       color: #8b949e; margin: 24px 0 8px; }
  .sub { color: #8b949e; margin-bottom: 12px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 5px 12px 5px 0; border-bottom: 1px solid #21262d;
           vertical-align: top; white-space: nowrap; }
  th { color: #8b949e; font-weight: normal; }
  td.wrap { white-space: normal; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 6px; }
  .on  { background: #3fb950; } .off { background: #f85149; }
  .st-RUNNING { color: #58a6ff; } .st-SUCCEEDED, .st-COMPLETED { color: #3fb950; }
  .st-FAILED { color: #f85149; } .st-PENDING { color: #8b949e; }
  .st-LEASED { color: #d29922; } .st-RECOVERING { color: #d29922; }
  .st-CANCELLED { color: #8b949e; }
  tr.job { cursor: pointer; } tr.job:hover td { background: #161b22; }
  tr.sel td { background: #161b22; }
  #events { max-height: 320px; overflow-y: auto; }
  .ev-time { color: #8b949e; margin-right: 8px; }
  .ev-type { color: #d2a8ff; margin-right: 8px; }
  a { color: #58a6ff; text-decoration: none; }
  .cols { display: flex; gap: 40px; flex-wrap: wrap; }
  .cols > div { flex: 1 1 420px; min-width: 0; overflow-x: auto; }
  .muted { color: #8b949e; }
</style>
</head>
<body>
<h1>FlashRuntime — local coordinator</h1>
<div class="sub" id="health">connecting…</div>

<h2>Nodes</h2>
<table><thead><tr><th></th><th>node</th><th>host</th><th>env</th><th>cpu</th>
<th>ram</th><th>accepted tasks</th><th>last heartbeat</th></tr></thead>
<tbody id="nodes"><tr><td colspan="8" class="muted">no nodes registered</td></tr></tbody></table>

<h2>Jobs</h2>
<table><thead><tr><th>job</th><th>name</th><th>backend</th><th>state</th>
<th>tasks</th><th>created</th></tr></thead>
<tbody id="jobs"><tr><td colspan="6" class="muted">no jobs yet</td></tr></tbody></table>

<div class="cols">
  <div>
    <h2>Tasks <span class="muted" id="selJob"></span></h2>
    <table><thead><tr><th>task</th><th>state</th><th>attempts</th><th>node</th></tr></thead>
    <tbody id="tasks"><tr><td colspan="4" class="muted">select a job</td></tr></tbody></table>
    <h2>Artifacts</h2>
    <div id="artifacts" class="muted">select a job</div>
  </div>
  <div>
    <h2>Events</h2>
    <div id="events" class="muted">select a job</div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let selected = null, taskCache = {};

const esc = s => String(s ?? "").replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const st = s => `<span class="st-${esc(s)}">${esc(s)}</span>`;
const t = iso => iso ? new Date(iso).toLocaleTimeString() : "";

async function j(path) { const r = await fetch(path); if (!r.ok) throw r; return r.json(); }

async function tick() {
  try {
    const h = await j("/healthz");
    $("health").textContent = `profile: ${h.profile} · ${new Date().toLocaleTimeString()}`;
  } catch { $("health").textContent = "coordinator unreachable"; return; }

  const nodes = await j("/v1alpha1/nodes");
  $("nodes").innerHTML = nodes.length ? nodes.map(n => `<tr>
    <td><span class="dot ${n.online ? "on" : "off"}"></span></td>
    <td>${esc(n.node_id)}</td><td>${esc(n.hostname)}</td><td>${esc(n.environment)}</td>
    <td>${n.capabilities.cpu_cores ?? "?"}</td>
    <td>${n.capabilities.memory_bytes ? (n.capabilities.memory_bytes/1e9).toFixed(1)+" GB" : "?"}</td>
    <td>${n.accepted_tasks}</td><td>${n.last_heartbeat_age_s}s ago</td></tr>`).join("")
    : `<tr><td colspan="8" class="muted">no nodes registered</td></tr>`;

  const jobs = await j("/v1alpha1/jobs");
  for (const job of jobs) {
    if (job.backend === "leases") {
      try { taskCache[job.job_id] = await j(`/v1alpha1/jobs/${job.job_id}/tasks`); } catch {}
    }
  }
  $("jobs").innerHTML = jobs.length ? jobs.map(job => {
    const tasks = taskCache[job.job_id] || [];
    const done = tasks.filter(x => x.state === "COMPLETED").length;
    const summary = tasks.length ? `${done}/${tasks.length} completed` : "—";
    return `<tr class="job ${job.job_id === selected ? "sel" : ""}" onclick="sel('${job.job_id}')">
      <td>${esc(job.job_id)}</td><td>${esc(job.spec.metadata.name)}</td>
      <td>${esc(job.backend)}</td><td>${st(job.state)}</td>
      <td>${summary}</td><td>${t(job.created_at)}</td></tr>`;
  }).join("") : `<tr><td colspan="6" class="muted">no jobs yet</td></tr>`;

  if (selected) await detail(selected);
}

async function detail(id) {
  $("selJob").textContent = "· " + id;
  const tasks = taskCache[id] || [];
  $("tasks").innerHTML = tasks.length ? tasks.map(x => `<tr>
    <td>${esc(x.task_id)}</td><td>${st(x.state)}</td>
    <td>${x.attempts}/${x.max_attempts}</td><td>${esc(x.node_id ?? "—")}</td></tr>`).join("")
    : `<tr><td colspan="4" class="muted">no leased tasks (ray-backend job?)</td></tr>`;

  try {
    const events = await j(`/v1alpha1/jobs/${id}/events`);
    $("events").innerHTML = events.slice().reverse().map(e =>
      `<div><span class="ev-time">${t(e.timestamp)}</span>` +
      `<span class="ev-type">${esc(e.type)}</span>${esc(e.message)}</div>`).join("");
  } catch {}
  try {
    const arts = await j(`/v1alpha1/jobs/${id}/artifacts`);
    $("artifacts").innerHTML = arts.length ? arts.map(a =>
      `<div><a href="/v1alpha1/artifacts/${esc(a.key)}" target="_blank">${esc(a.key)}</a>` +
      ` <span class="muted">(${a.size_bytes} B)</span></div>`).join("") : "none yet";
  } catch {}
}

function sel(id) { selected = id; tick(); }
tick(); setInterval(tick, 2000);
</script>
</body>
</html>"""


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(_PAGE)

    return router
