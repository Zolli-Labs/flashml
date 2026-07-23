# Snippet fixtures — counted, never executed

Each file is a comparator's **minimal working setup**, written honestly from
that project's own documentation (source URL cited in a comment at the top of
every file). The benchmark suite *counts* their non-blank, non-comment lines
(`benchmarks/_util.count_loc`) and diffs the adoption pairs
(`adopt_vanilla.py` → `adopt_flashruntime.py` / `adopt_accelerate.py`) — it
does **not** run them, so an uninstalled comparator (ray, accelerate) still
contributes an honest, auditable line count instead of a fabricated one.

If you change a comparator's snippet, keep it faithful to the cited docs and
keep the citation.
