"""render() — the live run page, one self-contained HTML document.

This is the single sanctioned large file of the viewer: a `flash.submit(
watch=True)` opens it, and it polls `/api/state` (the `state.collect()`
snapshot) every couple of seconds to draw one run's story — its topology,
its loss curve, its checkpoints, and its recovery decisions — with ZERO
external assets. No CDN, no web font fetch, no remote image: the page must
render with the network cut, because the only server it may ever talk to is
the loopback `RunViewerServer` that served it. Everything (CSS, JS) is
inline; the only same-origin links are `/api/state` and `/docs/`.

DESIGN CONTRACT (the house style, shared with `service/dashboard.py`):
the palette lives in `TOKENS` at the top of this module as the SINGLE SOURCE
OF TRUTH — a dark `#0d1117` field, `#161b22` panels, `#21262d` borders,
`#c9d1d9`/`#8b949e` text, and five oklch accents (cyan running, green
succeeded+verified, amber leased+recovering, red failed, violet
checkpoints). `dashboard.py` imports these same values so the two surfaces
read as one product. Sections, top to bottom: header → topology canvas →
loss canvas → checkpoint timeline → events feed → collapsible logs.

The drawing is plain, commented JS organized by section (`drawTopology`,
`drawLoss`, `renderCheckpoints`, …) — no framework, no build step, no
minification. Same readability bar as the Python (spec §2b).
"""

from __future__ import annotations

import json

# --------------------------------------------------------------------------
# Visual tokens — the single source of truth for BOTH this page and the
# coordinator dashboard (`service/dashboard.py` imports these values). Accents
# are oklch so they stay perceptually even across hues; the neutrals are the
# established GitHub-dark family the house style already used.
# --------------------------------------------------------------------------
TOKENS: dict[str, str] = {
    "bg": "#0d1117",  # page field
    "bg_inset": "#010409",  # deepest wells (canvas backdrops, code)
    "panel": "#161b22",  # raised panels / hover
    "border": "#21262d",  # hairlines
    "text": "#c9d1d9",  # body text
    "text_bright": "#e6edf3",  # headings
    "muted": "#8b949e",  # secondary text / section labels
    "running": "oklch(0.80 0.16 200)",  # cyan  — RUNNING
    "ok": "oklch(0.76 0.18 145)",  # green — SUCCEEDED / hash-verified
    "warn": "oklch(0.80 0.18 60)",  # amber — LEASED / RECOVERING / classified
    "fail": "oklch(0.70 0.20 20)",  # red   — FAILED / invalid
    "ckpt": "oklch(0.65 0.20 290)",  # violet — checkpoints
    "font": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
}

# The subset of tokens the JS drawing code needs (canvas fillStyles). Injected
# as a JS object literal so the colors are never duplicated between CSS and JS.
_JS_TOKEN_KEYS = ("bg", "bg_inset", "panel", "border", "text", "muted", "running", "ok", "warn", "fail", "ckpt")


# The document itself. CSS uses `%%token%%` placeholders (replaced in render());
# the JS reads colors from the injected `T` object. Nothing here reaches
# off-host — every href is same-origin (`/docs/`), and there are no src= assets.
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flashruntime — live run</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body { font: 13px/1.5 %%font%%; background: %%bg%%; color: %%text%%; padding: 20px; }
  a { color: %%running%%; text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* header ------------------------------------------------------------- */
  header { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
           border-bottom: 1px solid %%border%%; padding-bottom: 14px; }
  header .title { font-size: 15px; color: %%text_bright%%; }
  header .cmd { color: %%muted%%; word-break: break-all; }
  header .spacer { flex: 1 1 auto; }
  .meta { display: flex; gap: 18px; align-items: center; margin-top: 4px;
          flex-wrap: wrap; color: %%muted%%; }
  .meta b { color: %%text%%; font-weight: 600; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
           border: 1px solid %%border%%; font-size: 12px; }

  /* section scaffolding ------------------------------------------------ */
  section { margin-top: 22px; }
  h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .1em;
       color: %%muted%%; margin-bottom: 8px; font-weight: 600; }
  .panel { background: %%panel%%; border: 1px solid %%border%%; border-radius: 8px;
           padding: 12px; }
  canvas { display: block; width: 100%; background: %%bg_inset%%; border-radius: 6px; }
  #topology { height: 220px; }
  #loss { height: 200px; }

  /* checkpoint timeline ------------------------------------------------ */
  #checkpoints { display: flex; gap: 10px; overflow-x: auto; padding: 6px 2px; }
  #checkpoints .empty { color: %%muted%%; }
  .ckpt { flex: 0 0 auto; border: 1px solid %%border%%; border-left: 3px solid %%ckpt%%;
          border-radius: 6px; padding: 8px 10px; background: %%bg%%; min-width: 96px; }
  .ckpt .step { color: %%ckpt%%; font-weight: 600; }
  .ckpt .b { display: inline-block; margin-top: 4px; font-size: 11px;
             padding: 1px 6px; border-radius: 4px; }
  .ckpt .verified { color: %%ok%%; border: 1px solid %%ok%%; }
  .ckpt .invalid  { color: %%fail%%; border: 1px solid %%fail%%; }
  .ckpt .latest { color: %%text_bright%%; }
  .ckpt .sub { color: %%muted%%; font-size: 11px; margin-top: 2px; }

  /* events feed -------------------------------------------------------- */
  #events { max-height: 300px; overflow-y: auto; }
  #events .empty { color: %%muted%%; }
  .ev { display: flex; gap: 8px; padding: 3px 0; border-bottom: 1px solid %%border%%; }
  .ev:last-child { border-bottom: 0; }
  .ev .t { color: %%muted%%; white-space: nowrap; }
  .ev .ty { white-space: nowrap; }
  .ev .msg { color: %%text%%; word-break: break-word; }
  .ty-classified { color: %%warn%%; }
  .ty-recovery { color: %%running%%; }
  .ty-other { color: %%muted%%; }

  /* logs --------------------------------------------------------------- */
  details summary { cursor: pointer; color: %%muted%%; }
  pre#logbody { margin-top: 8px; padding: 10px; background: %%bg_inset%%;
                border-radius: 6px; max-height: 320px; overflow: auto;
                color: %%text%%; white-space: pre-wrap; word-break: break-word; }

  #err { display: none; margin-top: 12px; padding: 10px; border-radius: 6px;
         border: 1px solid %%fail%%; color: %%fail%%; }
</style>
</head>
<body>

<header>
  <div>
    <div class="title">flashruntime <span id="hstate" class="badge">…</span></div>
    <div class="cmd" id="hcmd">connecting…</div>
    <div class="meta">
      <span>mode <b id="hmode">—</b></span>
      <span>restarts <b id="hrestarts">—</b></span>
      <span>attempts <b id="hattempts">—</b></span>
    </div>
  </div>
  <div class="spacer"></div>
  <div><a href="/docs/">Docs ↗</a></div>
</header>

<div id="err"></div>

<section>
  <h2>Topology</h2>
  <div class="panel"><canvas id="topology"></canvas></div>
</section>

<section>
  <h2>Loss</h2>
  <div class="panel"><canvas id="loss"></canvas></div>
</section>

<section>
  <h2>Checkpoints</h2>
  <div class="panel"><div id="checkpoints"><span class="empty">no checkpoints yet</span></div></div>
</section>

<section>
  <h2>Events</h2>
  <div class="panel"><div id="events"><span class="empty">no events yet</span></div></div>
</section>

<section>
  <details>
    <summary>Logs (tail)</summary>
    <pre id="logbody">—</pre>
  </details>
</section>

<script>
// The color tokens, shared with the CSS above and with the dashboard. Kept in
// one object so a canvas fillStyle and a CSS rule never drift apart.
const T = %%tokens_json%%;
const POLL_MS = 2000;          // /api/state cadence — see poll()
let STATE = null;              // latest snapshot; every draw reads from this

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const clockTime = (ts) => (typeof ts === "number" ? new Date(ts * 1000).toLocaleTimeString() : "");

// Map a lifecycle state to its accent. Shared vocabulary with dashboard.py so
// a node here and a row there mean the same color.
function stateColor(s) {
  switch (s) {
    case "RUNNING": return T.running;                 // cyan
    case "LEASED": case "RECOVERING": return T.warn;  // amber
    case "SUCCEEDED": case "COMPLETED": return T.ok;   // green
    case "FAILED": return T.fail;                      // red
    default: return T.muted;                           // PENDING / CANCELLED
  }
}

// Append an alpha to an oklch(...) color: `oklch(L C H)` -> `oklch(L C H / a)`.
// (Canvas accepts CSS Color 4 oklch; this keeps the pulse the same hue.)
function withAlpha(color, a) {
  return color.replace(/\)\s*$/, " / " + a.toFixed(3) + ")");
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Size a canvas for the device pixel ratio so lines are crisp on retina and
// the drawing coordinate system stays in CSS pixels. Returns {ctx, w, h}.
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // 1 unit == 1 CSS pixel
  return { ctx, w, h };
}

// ---- header --------------------------------------------------------------
function renderHeader(s) {
  const cmd = (s.workload && s.workload.command) || [];
  $("hcmd").textContent = Array.isArray(cmd) ? cmd.join(" ") : String(cmd);
  $("hmode").textContent = (s.workload && s.workload.mode) || "—";
  const badge = $("hstate");
  badge.textContent = s.state || "—";
  badge.style.color = stateColor(s.state);
  badge.style.borderColor = stateColor(s.state);
  const attempts = s.attempts || [];
  // "restarts used" = attempts whose id carries an -rN recovery suffix.
  const used = attempts.filter((a) => /-r\d+$/.test(a.attempt_id || "")).length;
  $("hrestarts").textContent = used + " / " + (s.max_restarts == null ? "?" : s.max_restarts);
  $("hattempts").textContent = String(attempts.length);
}

// ---- topology canvas (animated) -----------------------------------------
// A single machine box holds one node per attempt/rank, colored by state. A
// Mode-A fan-out is just many attempts, so the same grid reads as a task grid.
// RUNNING nodes get a soft radial pulse (the reason drawTopology runs every
// animation frame, not only on poll).
function drawTopology(now) {
  const canvas = $("topology");
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const s = STATE;
  if (!s || s.error) return; // the error banner (renderStatic) tells the story

  const pad = 14;
  const boxX = pad, boxY = pad + 16, boxW = w - pad * 2, boxH = h - boxY - pad;
  ctx.strokeStyle = T.border;
  ctx.lineWidth = 1;
  roundRect(ctx, boxX, boxY, boxW, boxH, 10);
  ctx.stroke();
  ctx.fillStyle = T.muted;
  ctx.font = "11px " + fontFamily();
  ctx.textAlign = "left";
  ctx.fillText("127.0.0.1 · localhost", boxX + 2, pad + 10);

  const attempts = s.attempts || [];
  const n = attempts.length;
  if (!n) {
    ctx.fillStyle = T.muted;
    ctx.textAlign = "center";
    ctx.fillText("waiting for first launch…", w / 2, boxY + boxH / 2);
    return;
  }

  // A near-square grid of nodes inside the box.
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);
  const cellW = boxW / cols, cellH = boxH / rows;
  const radius = Math.max(6, Math.min(cellW, cellH) * 0.26);
  attempts.forEach((a, i) => {
    const cx = boxX + ((i % cols) + 0.5) * cellW;
    const cy = boxY + (Math.floor(i / cols) + 0.5) * cellH;
    const color = stateColor(a.state);
    // soft radial pulse while this node is RUNNING
    if (a.state === "RUNNING") {
      const puls = 0.5 + 0.5 * Math.sin(now / 500);
      const grad = ctx.createRadialGradient(cx, cy, radius * 0.5, cx, cy, radius * 2.6);
      grad.addColorStop(0, withAlpha(color, 0.18 + 0.22 * puls));
      grad.addColorStop(1, withAlpha(color, 0));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 2.6, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = T.text;
    ctx.font = "10px " + fontFamily();
    ctx.textAlign = "center";
    const label = String(a.attempt_id || ("#" + i)).replace(/^task-/, "");
    ctx.fillText(label, cx, cy + radius + 12);
  });
}

function fontFamily() {
  return "ui-monospace, SFMono-Regular, Menlo, monospace";
}

// ---- loss canvas ---------------------------------------------------------
// Pick the loss series out of the metrics tail (prefer a "loss" key, else the
// first numeric field that isn't a step counter), merge across attempts, and
// draw an autoscaled curve with the last value labeled.
function lossKey(rec) {
  if (typeof rec.loss === "number") return "loss";
  for (const k of Object.keys(rec)) {
    if (k === "step" || k === "epoch" || k === "index") continue;
    if (typeof rec[k] === "number" && isFinite(rec[k])) return k;
  }
  return null;
}

function collectLoss(s) {
  const pts = [];
  let key = "loss";
  (s.attempts || []).forEach((a) => {
    (a.metrics || []).forEach((m) => {
      const k = lossKey(m);
      if (k == null) return;
      key = k;
      const y = m[k];
      if (typeof y !== "number" || !isFinite(y)) return;
      const x = typeof m.step === "number" ? m.step : pts.length;
      pts.push({ x, y });
    });
  });
  pts.sort((p, q) => p.x - q.x);
  return { pts, key };
}

function drawLoss() {
  const canvas = $("loss");
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!STATE || STATE.error) return;

  const { pts, key } = collectLoss(STATE);
  if (pts.length === 0) {
    ctx.fillStyle = T.muted;
    ctx.font = "11px " + fontFamily();
    ctx.textAlign = "center";
    ctx.fillText("no metrics yet", w / 2, h / 2);
    return;
  }

  const padL = 8, padR = 64, padT = 14, padB = 18;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  let minX = pts[0].x, maxX = pts[0].x, minY = pts[0].y, maxY = pts[0].y;
  for (const p of pts) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  // pad the y-bounds so the curve never touches the frame; guard a flat series
  const span = maxY - minY || Math.abs(maxY) || 1;
  minY -= span * 0.08; maxY += span * 0.08;
  const xRange = maxX - minX || 1;
  const yRange = maxY - minY || 1;
  const sx = (x) => padL + ((x - minX) / xRange) * plotW;
  const sy = (y) => padT + (1 - (y - minY) / yRange) * plotH;

  // baseline
  ctx.strokeStyle = T.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  // the curve
  ctx.strokeStyle = T.running;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y))));
  ctx.stroke();

  // last value marker + label
  const last = pts[pts.length - 1];
  ctx.fillStyle = T.running;
  ctx.beginPath();
  ctx.arc(sx(last.x), sy(last.y), 3, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = T.text;
  ctx.font = "11px " + fontFamily();
  ctx.textAlign = "left";
  const label = key + " " + (Math.abs(last.y) < 1e-3 ? last.y.toExponential(2) : last.y.toFixed(4));
  ctx.fillText(label, Math.min(sx(last.x) + 6, w - padR), sy(last.y) - 4);
}

// ---- checkpoint timeline -------------------------------------------------
// Each manifest is a violet marker: its step, a hash-verified / invalid badge
// (the RE-VERIFIED state from state.collect, not the manifest's own claim),
// and a note when it is the one recovery would restore (latest_valid).
function renderCheckpoints(s) {
  const el = $("checkpoints");
  const cks = s.checkpoints || [];
  if (cks.length === 0) {
    el.innerHTML = '<span class="empty">no checkpoints yet</span>';
    return;
  }
  el.innerHTML = cks
    .map((c) => {
      const verified = c.validation === "hash_verified";
      const badge = verified
        ? '<span class="b verified">hash-verified</span>'
        : '<span class="b invalid">invalid</span>';
      const latest = c.latest_valid ? ' <span class="latest">★ latest</span>' : "";
      const age = typeof c.age_s === "number" ? Math.round(c.age_s) + "s ago" : "";
      return (
        '<div class="ckpt"><div class="step">step ' + esc(c.step) + "</div>" +
        badge + latest +
        '<div class="sub">' + esc(c.parts) + " parts · " + esc(age) + "</div></div>"
      );
    })
    .join("");
}

// ---- events feed ---------------------------------------------------------
// Newest first. Recovery decisions are highlighted per the contract:
// FAILURE_CLASSIFIED amber, RECOVERY_ACTION_SELECTED cyan — the message
// already carries the failure class and the policy's human reason.
function eventClass(type) {
  if (type === "FAILURE_CLASSIFIED") return "ty-classified";
  if (type === "RECOVERY_ACTION_SELECTED") return "ty-recovery";
  return "ty-other";
}

function renderEvents(s) {
  const el = $("events");
  const evs = s.events || [];
  if (evs.length === 0) {
    el.innerHTML = '<span class="empty">no events yet</span>';
    return;
  }
  el.innerHTML = evs
    .slice()
    .reverse()
    .map((e) =>
      '<div class="ev"><span class="t">' + esc(clockTime(e.ts)) + "</span>" +
      '<span class="ty ' + eventClass(e.type) + '">' + esc(e.type) + "</span>" +
      '<span class="msg">' + esc(e.message) + "</span></div>"
    )
    .join("");
}

// ---- logs tail -----------------------------------------------------------
function renderLogs(s) {
  const attempts = s.attempts || [];
  const chunks = attempts
    .filter((a) => a.log_tail)
    .map((a) => "--- " + (a.attempt_id || "") + " (" + (a.state || "") + ") ---\n" + a.log_tail);
  $("logbody").textContent = chunks.length ? chunks.join("\n\n") : "—";
}

// ---- everything that redraws on a poll (topology animates separately) ----
function renderStatic() {
  const s = STATE;
  const err = $("err");
  if (!s) return;
  if (s.error) {
    err.style.display = "block";
    err.textContent = "snapshot error: " + s.error;
    return;
  }
  err.style.display = "none";
  renderHeader(s);
  drawLoss();
  renderCheckpoints(s);
  renderEvents(s);
  renderLogs(s);
}

// ---- polling -------------------------------------------------------------
async function poll() {
  // Pause polling when the tab is hidden: this page can sit open for a whole
  // training run, and there is no reason to spend a laptop's battery fetching
  // a snapshot nobody is looking at. A visibilitychange handler resumes it.
  if (document.hidden) return;
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    STATE = await r.json();
  } catch (e) {
    STATE = { error: String(e) };
  }
  renderStatic();
}

// The topology pulse wants smooth animation, so it draws every frame from the
// latest STATE. We keep rendering after the run finishes (a terminal snapshot
// just draws a still frame) — the page stays a usable record of the run.
function animate(now) {
  if (!document.hidden) drawTopology(now);
  requestAnimationFrame(animate);
}

window.addEventListener("resize", () => renderStatic());
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });

poll();
setInterval(poll, POLL_MS);
requestAnimationFrame(animate);
</script>
</body>
</html>
"""


def render() -> str:
    """Return the live run page as one self-contained HTML string.

    All color tokens are substituted from `TOKENS` (the single source of
    truth): `%%name%%` placeholders in the CSS, and a `T` object literal in
    the JS. No other templating — the page is otherwise a static document.
    """
    subs = {f"%%{name}%%": value for name, value in TOKENS.items()}
    subs["%%tokens_json%%"] = json.dumps({k: TOKENS[k] for k in _JS_TOKEN_KEYS})
    html = _TEMPLATE
    for placeholder, value in subs.items():
        html = html.replace(placeholder, value)
    return html
