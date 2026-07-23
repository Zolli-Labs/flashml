"""One file per scenario. Each module exposes ``name``, ``hypothesis`` (one
sentence shown in the docs), and ``run(repeats) -> ResultRow`` — that structure
is all ``registry.Scenario`` requires. Heavy deps (torch, sklearn) are imported
lazily inside ``run`` so importing the registry stays cheap and dependency-free.
"""
