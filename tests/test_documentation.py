"""Structural checks that keep the learning-oriented documentation navigable."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


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
