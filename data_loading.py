"""
Loading utilities for the candidate pool.

Supports both the full candidates.jsonl(.gz) (100K rows) and the
sample_candidates.json (50 rows, pretty-printed list) used for development.
Streams rather than loading everything into a list of dicts at once where
practical, to stay inside the 16GB RAM budget on the full pool.
"""

from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Iterator, Any


def iter_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield candidate dicts one at a time from a jsonl, jsonl.gz, or a
    pretty-printed JSON list file (the sample format)."""
    path = Path(path)

    if path.suffix == ".json":
        # sample_candidates.json: a single JSON array, pretty-printed.
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        for c in candidates:
            yield c
        return

    opener = gzip.open if path.suffixes[-1:] == [".gz"] or path.name.endswith(".gz") else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Materialize the full candidate list. Fine for the 50-row sample;
    for the full 100K pool prefer iter_candidates() in a streaming pass
    where the per-candidate work doesn't need cross-candidate context."""
    return list(iter_candidates(path))


def candidate_count(path: str | Path) -> int:
    """Count candidates without loading everything into memory."""
    n = 0
    for _ in iter_candidates(path):
        n += 1
    return n