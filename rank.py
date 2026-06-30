#!/usr/bin/env python3
"""
Main entrypoint. Produces the submission CSV from a candidates file.

Usage (matches submission_metadata_template.yaml's reproduce_command):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Also accepts the 100-row sample_candidates.json format for local testing:
    python rank.py --candidates ./data/sample_candidates.json --out ./outputs/submission.csv --top-n 100

Designed to respect submission_spec.md Section 3 compute constraints:
    - CPU only, no GPU calls anywhere in this path
    - No network calls anywhere in this path (semantic_similarity.py uses
      scikit-learn TF-IDF+SVD, fit locally on the candidate batch - zero
      external calls)
    - Should comfortably fit 16GB RAM and 5 minutes on the full 100K pool;
      see README.md "Performance notes" for the measured numbers once run
      against the real candidates.jsonl.
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loading import load_candidates  # noqa: E402
from ranker import score_all_candidates, rank_top_n  # noqa: E402
from reasoning import generate_reasoning  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates against the Redrob JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl(.gz) or sample_candidates.json")
    parser.add_argument("--out", required=True, help="Path to write the output CSV")
    parser.add_argument("--top-n", type=int, default=100, help="Number of ranked rows to output (spec requires 100 for the real submission)")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Loading candidates from {args.candidates} ...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"  loaded {len(candidates)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    t1 = time.time()
    print("Scoring candidates ...", file=sys.stderr)
    scores = score_all_candidates(candidates)
    print(f"  scored {len(scores)} candidates in {time.time()-t1:.1f}s", file=sys.stderr)

    honeypot_count = sum(1 for s in scores if s.is_honeypot)
    print(f"  flagged {honeypot_count} honeypots (excluded from ranking)", file=sys.stderr)

    top = rank_top_n(scores, n=args.top_n, exclude_honeypots=True)

    by_id = {c["candidate_id"]: c for c in candidates}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, s in enumerate(top, start=1):
            candidate = by_id[s.candidate_id]
            reasoning = generate_reasoning(candidate, s)
            writer.writerow([s.candidate_id, rank, f"{s.final_score:.8f}", reasoning])

    elapsed = time.time() - t0
    print(f"Wrote {len(top)} ranked rows to {out_path} in {elapsed:.1f}s total", file=sys.stderr)

    if elapsed > 300:
        print("WARNING: exceeded the 5-minute (300s) compute budget from submission_spec.md Section 3.", file=sys.stderr)


if __name__ == "__main__":
    main()