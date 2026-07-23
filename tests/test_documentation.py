"""Structural checks that keep the learning-oriented documentation navigable."""

import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
# Every ```python fence in the docs site must at least be syntactically true.
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _load_builder():
    """Import scripts/build_docs.py by path (it is a dev script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "build_docs", ROOT / "scripts" / "build_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def documentation_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files.extend((ROOT / "docs").rglob("*.md"))
    files.extend(
        path
        for path in (
            ROOT / "apps" / "README.md",
            ROOT / "apps" / "dashboard" / "README.md",
            ROOT / "legacy" / "README.md",
            ROOT / "archive" / "README.md",
        )
        if path.exists()
    )
    return sorted(set(files))


def test_relative_documentation_links_resolve():
    broken = []
    for document in documentation_files():
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")

    assert not broken, "Broken documentation links:\n" + "\n".join(broken)


# --------------------------------------------------------------------------
# The docs site (scripts/build_docs.py + docs/site/): builds in-process, its
# internal links resolve, and its checker actually catches a broken one.
# --------------------------------------------------------------------------
def test_docs_site_builds_in_process(tmp_path):
    builder = _load_builder()
    builder.build_site(builder.SRC, tmp_path)

    # Every nav page produced an HTML file, plus the search index.
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "get-started.html").is_file()
    index = json.loads((tmp_path / "search-index.json").read_text(encoding="utf-8"))
    assert any(entry["url"] == "index.html" for entry in index)
    assert all({"url", "title", "text"} <= set(entry) for entry in index)

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    # Visual continuity with the viewer: the page is built from the SAME tokens.
    from flashruntime.viewer.page import TOKENS

    assert TOKENS["bg"] in html
    assert TOKENS["font"] in html
    # Self-contained: no off-host assets (same rule as the viewer page).
    assert "http://" not in html.replace("http://127.0.0.1", "").replace(
        "http://www.w3.org", ""
    )
    assert "https://cdn" not in html and "src=\"http" not in html


def test_docs_site_internal_links_resolve():
    builder = _load_builder()
    problems = builder.check(builder.SRC)
    assert problems == [], "Broken docs-site links/nav:\n" + "\n".join(problems)


def test_docs_site_linkcheck_catches_bad_link(tmp_path):
    # A deliberate-bad-link fixture proves --check would exit non-zero.
    (tmp_path / "index.md").write_text(
        "# Home\n\nA [dangling link](does-not-exist.md) here.\n", encoding="utf-8"
    )
    (tmp_path / "_nav.yml").write_text("Home:\n  - index.md\n", encoding="utf-8")
    builder = _load_builder()
    problems = builder.check(tmp_path)
    assert problems, "the checker must catch a deliberate broken link"
    assert any("does-not-exist" in problem for problem in problems)


def test_docs_site_missing_nav_entry_is_caught(tmp_path):
    # A nav that names a file with no .md source must fail the check too.
    (tmp_path / "index.md").write_text("# Home\n", encoding="utf-8")
    (tmp_path / "_nav.yml").write_text(
        "Home:\n  - index.md\n  - ghost.md\n", encoding="utf-8"
    )
    builder = _load_builder()
    problems = builder.check(tmp_path)
    assert any("ghost" in problem for problem in problems)


# --------------------------------------------------------------------------
# Nested nav entries (tutorials/…, guides/…): the next docs slice ships them.
# Files must land in subdirs, and every href must resolve FROM ITS OWN PAGE'S
# directory — a page one level deep links `../index.html`, not `index.html`.
# --------------------------------------------------------------------------
def _internal_html_hrefs(html_text: str) -> list[str]:
    """The real page links in a built page: internal `href="….html"` only.
    Skips anchors, off-host/absolute URLs, and the search JS's `href="` + expr
    fragments (those carry spaces / `+`, never end in `.html`)."""
    hrefs = []
    for href in re.findall(r'href="([^"]+)"', html_text):
        target = href.split("#", 1)[0]
        if not target.endswith(".html"):
            continue
        if " " in target or "://" in target or target.startswith("/") or "%%" in target:
            continue
        hrefs.append(target)
    return hrefs


def test_docs_site_nested_pages_build_and_resolve(tmp_path):
    src = tmp_path / "src"
    (src / "tutorials").mkdir(parents=True)
    (src / "index.md").write_text(
        "# Home\n\nGo [deep](tutorials/deep.md).\n", encoding="utf-8"
    )
    (src / "tutorials" / "deep.md").write_text(
        "# Deep\n\nBack [home](../index.md).\n", encoding="utf-8"
    )
    (src / "_nav.yml").write_text(
        "Section:\n  - index.md\n  - tutorials/deep.md\n", encoding="utf-8"
    )

    builder = _load_builder()
    out = tmp_path / "out"
    builder.build_site(src, out)

    # Files land nested (parent dir was created).
    assert (out / "index.html").is_file()
    assert (out / "tutorials" / "deep.html").is_file()

    # Every internal .html href resolves ON DISK from its own page's directory.
    for page in (out / "index.html", out / "tutorials" / "deep.html"):
        hrefs = _internal_html_hrefs(page.read_text(encoding="utf-8"))
        assert hrefs, f"expected internal links on {page}"
        for href in hrefs:
            resolved = (page.parent / href).resolve()
            assert resolved.is_file(), f"{page} -> {href} does not resolve on disk"

    # The cross-links point the right way: down into the subdir, back up to root.
    home = (out / "index.html").read_text(encoding="utf-8")
    deep = (out / "tutorials" / "deep.html").read_text(encoding="utf-8")
    assert 'href="tutorials/deep.html"' in home
    assert 'href="../index.html"' in deep

    # The search index stores root-relative site paths (JS resolves against ROOT).
    index = json.loads((out / "search-index.json").read_text(encoding="utf-8"))
    assert {entry["url"] for entry in index} == {"index.html", "tutorials/deep.html"}


def test_docs_site_linkcheck_catches_bad_nested_link(tmp_path):
    # A broken link FROM a nested page: check() must list it, never traceback —
    # and must NOT false-flag the valid `../index.md` up-link on the same page.
    (tmp_path / "tutorials").mkdir()
    (tmp_path / "index.md").write_text("# Home\n", encoding="utf-8")
    (tmp_path / "tutorials" / "deep.md").write_text(
        "# Deep\n\nBroken [nope](../missing.md); valid [home](../index.md).\n",
        encoding="utf-8",
    )
    (tmp_path / "_nav.yml").write_text(
        "Section:\n  - index.md\n  - tutorials/deep.md\n", encoding="utf-8"
    )
    builder = _load_builder()
    problems = builder.check(tmp_path)  # must return a listing, not raise
    assert any("missing" in problem for problem in problems), problems
    assert not any(
        "index.html" in problem and "no such" in problem for problem in problems
    ), problems


def test_docs_site_python_blocks_compile():
    builder = _load_builder()
    for md in sorted(builder.SRC.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for i, block in enumerate(PYTHON_FENCE.findall(text)):
            # compile (not exec): docs must be syntactically true, not run here.
            compile(block, f"{md.name}#python-{i}", "exec")
