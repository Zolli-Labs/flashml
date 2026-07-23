#!/usr/bin/env python3
"""build_docs.py — render docs/site/*.md into a self-contained docs website.

Dev-time only (NOT a runtime dependency): `markdown` + `pyyaml` live in the
`dev` extra. The output is a set of static, zero-asset HTML pages that go to
TWO places:

  * `flashruntime/viewer/_docs/`  — the packaged docs the RunViewerServer
    serves at `/docs` (so a `flash.submit(watch=True)` page can link to docs
    that work with the network cut), and
  * `site/`                       — the same tree for GitHub Pages.

Visual continuity is the whole point: the page template is built from the
SAME `TOKENS` the viewer's live-run page uses (imported from
`flashruntime.viewer.page`), so the docs read as one product with the viewer
— dark GitHub-family surface, mono font, uppercase section labels — and, like
the viewer page, every byte is inline (CSS + JS): no CDN, no web font, no
remote image. The doc-link/asset test enforces the "no off-host asset" rule.

The navigation is DATA, not code (extensibility rule §2b): `_nav.yml` is an
ordered mapping of `section -> [file, ...]`. Adding a page is a ONE-LINE nav
edit plus one `.md` file — nothing in this script changes. Internal
cross-links are authored as `.md` (so the repo's own relative-link checker
resolves them to real files) and rewritten to `.html` for the built site.

`--check` builds to a temp dir and verifies every internal link and nav entry
resolves, exiting 1 with a listing on failure (nothing is written on --check).
"""

from __future__ import annotations

import argparse
import html as _html
import json
import posixpath
import re
import shutil
import sys
import tempfile
from pathlib import Path

import markdown
import yaml

from flashruntime.viewer.page import TOKENS

# --------------------------------------------------------------------------
# Paths + constants
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "docs" / "site"
VIEWER_OUT = REPO_ROOT / "flashruntime" / "viewer" / "_docs"  # served at /docs
PAGES_OUT = REPO_ROOT / "site"  # GitHub Pages

# The benchmarks page renders its tables from the MEASURED baseline JSON at
# build time (Task 10): a page can never show a figure the suite didn't produce.
BENCH_RESULTS = REPO_ROOT / "benchmarks" / "results"
BENCH_MARKER = "<!-- BENCH_TABLES -->"

MD_EXTENSIONS = ["fenced_code", "tables", "toc"]  # toc → heading ids for anchors
# `href="foo.md"` / `href="foo.md#sec"` for a same-repo page → rewrite to .html.
_INTERNAL_MD = re.compile(r'(href=")([^"]+?)\.md(#[^"]*)?(")')
_PRE = re.compile(r"<pre>(.*?)</pre>", re.DOTALL)  # wrap for the copy button
_TAG = re.compile(r"<[^>]+>")  # strip tags → search-index plain text
_HREF = re.compile(r'href="([^"]+)"')  # link-check scan


# --------------------------------------------------------------------------
# Nav + page loading. `_nav.yml` is the single source of navigation order.
# --------------------------------------------------------------------------
def load_nav(src: Path) -> dict[str, list[str]]:
    """Read `_nav.yml` as an ordered `{section: [file, ...]}` mapping. A
    Python dict preserves insertion order (3.7+), so sections and files keep
    the author's order without any explicit sort."""
    data = yaml.safe_load((Path(src) / "_nav.yml").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("_nav.yml must be a mapping of section -> [files]")
    return {str(section): [str(f) for f in (files or [])] for section, files in data.items()}


def _bench_tables_md() -> str:
    """Render the committed baseline JSON into Markdown tables (host block,
    summary, per-scenario detail + repro command). Falls back to a build-time
    hint if no baseline is committed, so the docs still build on a fresh clone."""
    baselines = sorted(BENCH_RESULTS.glob("baseline-*.json"))
    if not baselines:
        return "_No baseline committed yet — run `python -m benchmarks run --all --repeats 5`._"
    # `benchmarks` is a repo-root dev/eval package (not shipped in the wheel), so
    # ensure the repo root is importable when this runs as a bare script.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from benchmarks import report  # dev-time only; pulls no heavy deps at import

    return report.render_document(json.loads(baselines[0].read_text(encoding="utf-8")))


def _inject_bench(text: str) -> str:
    """Substitute the measured benchmark tables for the page's marker. A no-op
    for every page that does not carry the marker."""
    if BENCH_MARKER in text:
        return text.replace(BENCH_MARKER, _bench_tables_md())
    return text


def _page_title(text: str, fallback: str) -> str:
    """First `# ` heading is the page title (fallback: the filename)."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _load_pages(src: Path, nav: dict[str, list[str]]):
    """Return (`pages`, `missing`): pages is `[(file, title, md_text), ...]`
    for every nav entry that has a source file; missing lists the nav entries
    that do not (so --check can report them instead of crashing)."""
    pages, missing = [], []
    for section, files in nav.items():
        for file in files:
            path = Path(src) / file
            if not path.is_file():
                missing.append(f"{section} -> {file}")
                continue
            text = _inject_bench(path.read_text(encoding="utf-8"))
            pages.append((file, _page_title(text, file), text))
    return pages, missing


# --------------------------------------------------------------------------
# Markdown → HTML body: render, rewrite internal .md links, add copy buttons.
# --------------------------------------------------------------------------
def _html_name(file: str) -> str:
    # The page's ROOT-RELATIVE site path (e.g. "tutorials/deep.html"). Always
    # POSIX-style ("/"): these are URLs and on-disk keys, never OS paths.
    return file[:-3] + ".html" if file.endswith(".md") else file + ".html"


def _relative_href(target: str, from_page: str) -> str:
    """Href to built page `target` (a root-relative site path) as seen from
    `from_page` (also a root-relative site path). We resolve relative to the
    CURRENT page's directory so a page in a subdir links correctly — a page at
    `tutorials/deep.html` must reach root's `index.html` as `../index.html`,
    not `index.html`. `posixpath` (not `os.path`) because these are URLs: the
    separator must be "/" on every OS, never a backslash on Windows."""
    return posixpath.relpath(target, posixpath.dirname(from_page) or ".")


def _root_prefix(from_page: str) -> str:
    """The relative path from `from_page`'s directory back to the site root,
    with a trailing "/" ("" for a root page, "../" one level deep). The page
    template substitutes it into the few links that point at root assets — the
    brand link, the search-index fetch, and (in JS) the search-result hrefs —
    so those resolve from a subdir page too."""
    depth = _html_name(from_page).count("/")
    return "../" * depth


def _render_body(text: str) -> str:
    body = markdown.Markdown(extensions=MD_EXTENSIONS).convert(text)
    body = _INTERNAL_MD.sub(_rewrite_md_link, body)
    return _PRE.sub(_wrap_code, body)


def _rewrite_md_link(m: re.Match) -> str:
    # Only rewrite repo-local links; leave any scheme'd URL (e.g. a raw .md on
    # GitHub) exactly as written.
    if "://" in m.group(2):
        return m.group(0)
    return f'{m.group(1)}{m.group(2)}.html{m.group(3) or ""}{m.group(4)}'


def _wrap_code(m: re.Match) -> str:
    # A copy button that reads the <pre>'s own text — no round-trip, no CDN.
    return (
        '<div class="codeblock"><button class="copybtn" type="button" '
        'aria-label="Copy code to clipboard">Copy</button><pre>'
        + m.group(1)
        + "</pre></div>"
    )


def _plain_text(body_html: str) -> str:
    """Body HTML → collapsed plain text for the client-side search index."""
    return re.sub(r"\s+", " ", _html.unescape(_TAG.sub(" ", body_html))).strip()


# --------------------------------------------------------------------------
# Page template — built from the viewer's TOKENS so the two surfaces match.
# --------------------------------------------------------------------------
def _sidebar(nav: dict[str, list[str]], titles: dict[str, str], current: str) -> str:
    # Every nav href is computed RELATIVE to `current`'s directory, so the same
    # sidebar links resolve whether the page sits at root or in a subdir.
    here = _html_name(current)
    parts = []
    for section, files in nav.items():
        parts.append(f'<div class="nav-section">{_html.escape(section)}</div>')
        for file in files:
            cls = "nav-link active" if file == current else "nav-link"
            label = _html.escape(titles.get(file, file))
            href = _relative_href(_html_name(file), here)
            parts.append(f'<a class="{cls}" href="{href}">{label}</a>')
    return "\n".join(parts)


def render_page(title: str, nav_html: str, body_html: str, root: str) -> str:
    doc = _TEMPLATE
    for name, value in TOKENS.items():  # %%bg%%, %%font%%, … (no key is a prefix of another)
        doc = doc.replace(f"%%{name}%%", value)
    doc = doc.replace("%%title%%", _html.escape(title))
    doc = doc.replace("%%root%%", root)  # path back to site root (root assets)
    doc = doc.replace("%%nav%%", nav_html)
    return doc.replace("%%content%%", body_html)


# --------------------------------------------------------------------------
# Emit / build / check
# --------------------------------------------------------------------------
def _emit(nav: dict[str, list[str]], pages: list, out: Path) -> None:
    """Write every page + `search-index.json` into `out`. Total over the pages
    it is given — never raises on a missing nav file (check() reports those)."""
    out.mkdir(parents=True, exist_ok=True)
    titles = {file: title for file, title, _ in pages}
    index = []
    for file, title, text in pages:
        body = _render_body(text)
        dest = out / _html_name(file)
        dest.parent.mkdir(parents=True, exist_ok=True)  # nested nav entry → subdir
        page = render_page(title, _sidebar(nav, titles, file), body, _root_prefix(file))
        dest.write_text(page, encoding="utf-8")
        # URLs in the index are ROOT-RELATIVE site paths; the page JS resolves
        # them against its own ROOT (see the template). One shared index for all.
        index.append({"url": _html_name(file), "title": title, "text": _plain_text(body)})
    (out / "search-index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


def build_site(src: Path, out: Path) -> None:
    """Render `src`'s nav into a fresh `out` dir. Raises if a nav entry has no
    source file — a normal build must fail loudly on a dangling nav line."""
    src, out = Path(src), Path(out)
    nav = load_nav(src)
    pages, missing = _load_pages(src, nav)
    if missing:
        raise FileNotFoundError("nav references missing source files: " + ", ".join(missing))
    if out.exists():
        shutil.rmtree(out)  # clean rebuild — never serve a stale page
    _emit(nav, pages, out)


def check(src: Path) -> list[str]:
    """Build to a temp dir and return a list of problems (empty = OK): missing
    nav sources, plus any internal href that does not resolve to a built page.
    Off-host URLs, same-origin absolute routes (/api/…), and pure #anchors are
    intentionally out of scope.

    Links are read from the rendered content bodies + sidebars — the authored
    and nav links — not from the full page HTML: the shared template carries
    inline JS that legitimately contains `href="…"` fragments, which are not
    site links to resolve."""
    src = Path(src)
    nav = load_nav(src)
    pages, missing = _load_pages(src, nav)
    problems = [f"nav entry has no source file: {m}" for m in missing]

    with tempfile.TemporaryDirectory() as td:
        _emit(nav, pages, Path(td))  # prove the whole emit path runs end-to-end

    built = {_html_name(file) for file, _, _ in pages}
    titles = {file: title for file, title, _ in pages}
    for file, _, text in pages:
        page = _html_name(file)  # root-relative site path of the current page
        page_dir = posixpath.dirname(page)
        links = _HREF.findall(_render_body(text)) + _HREF.findall(_sidebar(nav, titles, file))
        for href in links:
            target = href.split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "/", "data:")):
                continue
            # hrefs are page-RELATIVE (sidebar + authored cross-links) — resolve
            # each back to a root-relative site path before checking it exists,
            # with the SAME rules the browser uses (../ climbs out of the subdir).
            resolved = posixpath.normpath(posixpath.join(page_dir, target))
            if resolved not in built:
                problems.append(f"{page} -> {href} (no such built page)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the flashruntime docs site.")
    parser.add_argument("--check", action="store_true", help="verify links + nav resolve; exit 1 on failure (writes nothing)")
    parser.add_argument("--out", action="append", type=Path, help="override output dir(s); repeatable (default: viewer _docs/ and site/)")
    args = parser.parse_args(argv)

    if args.check:
        problems = check(SRC)
        if problems:
            print("docs --check FAILED:", file=sys.stderr)
            for problem in problems:
                print("  " + problem, file=sys.stderr)
            return 1
        print("docs --check OK")
        return 0

    for out in args.out or [VIEWER_OUT, PAGES_OUT]:
        build_site(SRC, out)
        print(f"built docs -> {out}")
    return 0


# --------------------------------------------------------------------------
# The self-contained document. `%%token%%` placeholders (color tokens from the
# viewer) are substituted in render_page(); nothing here reaches off-host.
# --------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%title%% — flashruntime docs</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body { font: 14px/1.65 %%font%%; background: %%bg%%; color: %%text%%; }
  a { color: %%running%%; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .layout { display: flex; align-items: flex-start; min-height: 100vh; }

  /* sidebar — PyTorch-docs-like: sections from _nav.yml, current page active */
  .sidebar { flex: 0 0 264px; position: sticky; top: 0; height: 100vh; overflow-y: auto;
             border-right: 1px solid %%border%%; background: %%bg%%; padding: 20px 16px; }
  .brand { color: %%text_bright%%; font-size: 15px; letter-spacing: .04em; }
  .brand small { display: block; margin-top: 2px; color: %%muted%%; font-size: 10px;
                 text-transform: uppercase; letter-spacing: .14em; }
  .search { position: relative; margin: 16px 0; }
  #q { width: 100%; background: %%bg_inset%%; border: 1px solid %%border%%; border-radius: 6px;
       color: %%text%%; font: 12px %%font%%; padding: 7px 9px; }
  #q:focus { outline: 1px solid %%running%%; }
  #results { position: absolute; left: 0; right: 0; top: 112%; z-index: 5; display: none;
             background: %%panel%%; border: 1px solid %%border%%; border-radius: 6px; overflow: hidden; }
  #results.open { display: block; }
  #results a { display: block; padding: 7px 9px; border-bottom: 1px solid %%border%%; color: %%text%%; }
  #results a:last-child { border-bottom: 0; }
  #results a:hover { background: %%bg_inset%%; text-decoration: none; }
  #results a b { color: %%text_bright%%; font-weight: 600; }
  #results a span { display: block; margin-top: 2px; color: %%muted%%; font-size: 11px; }
  #results .nohit { padding: 7px 9px; color: %%muted%%; }
  .nav-section { margin: 16px 0 6px; color: %%muted%%; font-size: 10px; font-weight: 600;
                 text-transform: uppercase; letter-spacing: .14em; }
  .nav-link { display: block; padding: 4px 8px; border-radius: 5px; color: %%text%%; font-size: 13px; }
  .nav-link:hover { background: %%panel%%; text-decoration: none; }
  .nav-link.active { background: %%panel%%; color: %%text_bright%%; box-shadow: inset 2px 0 0 %%running%%; }

  /* content */
  .content { flex: 1 1 auto; max-width: 840px; min-width: 0; padding: 34px 40px 80px; }
  .content h1 { color: %%text_bright%%; font-size: 26px; margin: 0 0 16px; }
  .content h2 { color: %%text_bright%%; font-size: 18px; margin: 30px 0 10px;
                padding-top: 10px; border-top: 1px solid %%border%%; }
  .content h3 { color: %%text_bright%%; font-size: 15px; margin: 22px 0 8px; }
  .content p, .content li { color: %%text%%; }
  .content ul, .content ol { padding-left: 22px; margin: 10px 0; }
  .content li { margin: 4px 0; }
  .content blockquote { margin: 14px 0; padding: 2px 14px; border-left: 3px solid %%warn%%;
                        background: %%panel%%; border-radius: 0 6px 6px 0; color: %%muted%%; }
  .content :not(pre) > code { background: %%bg_inset%%; border: 1px solid %%border%%;
                              border-radius: 4px; padding: 1px 5px; font-size: 12.5px; }
  .content table { border-collapse: collapse; margin: 14px 0; display: block; overflow-x: auto; }
  .content th, .content td { border: 1px solid %%border%%; padding: 6px 10px; text-align: left; }
  .content th { color: %%text_bright%%; background: %%panel%%; }

  /* code blocks + copy button */
  .codeblock { position: relative; margin: 14px 0; }
  .codeblock pre { background: %%bg_inset%%; border: 1px solid %%border%%; border-radius: 8px;
                   padding: 14px 16px; overflow-x: auto; }
  .codeblock pre code { padding: 0; border: 0; background: none; font-size: 12.5px; color: %%text%%; }
  .copybtn { position: absolute; top: 8px; right: 8px; padding: 3px 8px; cursor: pointer;
             background: %%panel%%; border: 1px solid %%border%%; border-radius: 5px;
             color: %%muted%%; font: 11px %%font%%; }
  .copybtn:hover { color: %%text_bright%%; border-color: %%running%%; }

  @media (max-width: 800px) {
    .layout { flex-direction: column; }
    .sidebar { position: static; height: auto; width: 100%; flex-basis: auto;
               border-right: 0; border-bottom: 1px solid %%border%%; }
    .content { padding: 24px 18px 60px; }
  }
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <a class="brand" href="%%root%%index.html">flashruntime<small>documentation</small></a>
    <div class="search">
      <input id="q" type="search" placeholder="Search docs  (press /)" autocomplete="off" spellcheck="false">
      <div id="results"></div>
    </div>
    <nav>%%nav%%</nav>
  </aside>
  <main class="content">%%content%%</main>
</div>

<script>
// ---- client-side search: fetch the builder's index, filter as you type -----
// (<=60 lines, vanilla JS, no external anything — the index is a sibling file.)
let INDEX = [];
// Path from THIS page back to the site root. The index stores root-relative
// URLs (e.g. "tutorials/deep.html"); we fetch it and build result links
// against ROOT so a page in a subdir resolves them too (see build_docs.py).
const ROOT = "%%root%%";
const q = document.getElementById("q");
const results = document.getElementById("results");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
fetch(ROOT + "search-index.json").then((r) => r.json()).then((d) => { INDEX = d; }).catch(() => {});

// A short context window around the first match, so a hit shows WHY it matched.
function snippet(text, needle) {
  const i = text.toLowerCase().indexOf(needle);
  if (i < 0) return "";
  const start = Math.max(0, i - 32);
  return (start > 0 ? "…" : "") + text.slice(start, i + needle.length + 44).trim() + "…";
}
function runSearch() {
  const needle = q.value.trim().toLowerCase();
  if (!needle) { results.className = ""; results.innerHTML = ""; return; }
  const hits = INDEX.map((p) => {
    const inTitle = p.title.toLowerCase().includes(needle);
    const inText = p.text.toLowerCase().includes(needle);
    if (!inTitle && !inText) return null;
    return { url: p.url, title: p.title, snip: inText ? snippet(p.text, needle) : "" };
  }).filter(Boolean).slice(0, 20);
  results.className = "open";
  results.innerHTML = hits.length
    ? hits.map((h) => '<a href="' + ROOT + h.url + '"><b>' + esc(h.title) + "</b>" +
        (h.snip ? "<span>" + esc(h.snip) + "</span>" : "") + "</a>").join("")
    : '<div class="nohit">no matches</div>';
}
q.addEventListener("input", runSearch);
q.addEventListener("focus", runSearch);
document.addEventListener("click", (e) => {
  if (!e.target.closest(".search")) { results.className = ""; }  // dismiss on outside click
});
document.addEventListener("keydown", (e) => {  // "/" focuses search, like PyTorch docs
  if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); }
});

// ---- copy buttons: read the <pre>'s own text into the clipboard ------------
document.querySelectorAll(".copybtn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const code = btn.parentElement.querySelector("pre").innerText;
    navigator.clipboard.writeText(code).then(() => {
      const was = btn.textContent; btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = was; }, 1200);
    }).catch(() => {});
  });
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
