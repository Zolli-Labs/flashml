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


def test_docs_site_python_blocks_compile():
    builder = _load_builder()
    for md in sorted(builder.SRC.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for i, block in enumerate(PYTHON_FENCE.findall(text)):
            # compile (not exec): docs must be syntactically true, not run here.
            compile(block, f"{md.name}#python-{i}", "exec")
