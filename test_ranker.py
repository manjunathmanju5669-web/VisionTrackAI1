"""
Regression tests for the ranker, built directly from real bugs caught
during development against sample_candidates.json. Each test documents
the specific failure it guards against, not just "does the function run."

Run with:
    python -m pytest tests/ -v

Or, in an environment without pytest installed, run this file directly:
    python tests/test_ranker.py
"""

from __future__ import annotations
import sys
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_loading import load_candidates  # noqa: E402
from ranker import score_all_candidates, rank_top_n  # noqa: E402
from skill_matching import score_skills  # noqa: E402
from disqualifiers import detect_disqualifiers  # noqa: E402
from honeypot import detect_honeypot  # noqa: E402

SAMPLE_PATH = ROOT / "data" / "sample_candidates.json"

TRAP_CANDIDATE_ID = "CAND_0000021"
GENUINE_MATCH_ID = "CAND_0000031"
IRRELEVANT_TITLES = [
    "CAND_0000007",
    "CAND_0000022",
    "CAND_0000020",
    "CAND_0000026",
]


def _load():
    return load_candidates(SAMPLE_PATH)


def _by_id(candidates):
    return {c["candidate_id"]: c for c in candidates}


def test_genuine_match_outranks_trap_candidate():
    candidates = _load()
    scores = score_all_candidates(candidates)
    by_id = {s.candidate_id: s for s in scores}

    trap = by_id[TRAP_CANDIDATE_ID]
    genuine = by_id[GENUINE_MATCH_ID]

    assert genuine.final_score > trap.final_score, (
        f"Genuine match ({genuine.final_score:.3f}) must outscore the "
        f"keyword-stuffer trap ({trap.final_score:.3f})"
    )
    assert genuine.final_score > trap.final_score * 2, (
        "Gap between genuine match and trap candidate should be substantial, "
        "not a marginal edge"
    )


def test_genuine_match_ranks_in_top_3_of_sample():
    candidates = _load()
    scores = score_all_candidates(candidates)
    ranked = sorted(scores, key=lambda s: -s.final_score)
    position = next(i for i, s in enumerate(ranked, 1) if s.candidate_id == GENUINE_MATCH_ID)
    assert position <= 3, f"Genuine match landed at rank {position}, expected top 3"


def test_irrelevant_titles_excluded_from_top_10():
    candidates = _load()
    scores = score_all_candidates(candidates)
    top10_ids = {s.candidate_id for s in rank_top_n(scores, n=10)}

    leaked = top10_ids & set(IRRELEVANT_TITLES)
    assert not leaked, f"Irrelevant-title candidates leaked into top 10: {leaked}"


def test_title_chaser_does_not_flag_lateral_specialization():
    candidates = _load()
    by_id = _by_id(candidates)
    genuine = by_id[GENUINE_MATCH_ID]

    result = detect_disqualifiers(genuine)
    assert "title_chaser" not in result.triggered, (
        f"Genuine match incorrectly flagged as title_chaser; evidence: {result.evidence}"
    )


def test_skill_rescue_requires_history_corroboration():
    candidates = _load()
    by_id = _by_id(candidates)

    for cid in IRRELEVANT_TITLES:
        result = score_skills(by_id[cid])
        assert result.score < 0.15, (
            f"{cid} ({by_id[cid]['profile']['current_title']}) scored "
            f"{result.score:.3f} on skills - expected near-zero, no ML-relevant "
            f"signal in this profile"
        )


def test_salary_inversion_alone_does_not_flag_honeypot():
    candidates = _load()
    salary_inverted_only = []
    for c in candidates:
        salary = c["redrob_signals"].get("expected_salary_range_inr_lpa", {})
        lo, hi = salary.get("min"), salary.get("max")
        if lo is not None and hi is not None and lo > hi:
            h = detect_honeypot(c)
            if len(h.flags) <= 1:
                salary_inverted_only.append(c["candidate_id"])

    for cid in salary_inverted_only:
        h = detect_honeypot(next(c for c in candidates if c["candidate_id"] == cid))
        assert not h.is_honeypot or len(h.flags) > 1, (
            f"{cid} flagged as honeypot on salary inversion alone"
        )


def test_csv_output_passes_official_validator_format_rules():
    out_path = ROOT / "output" / "test_validator_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(ROOT / "rank.py"),
         "--candidates", str(SAMPLE_PATH),
         "--out", str(out_path),
         "--top-n", "48"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"rank.py failed: {result.stderr}"

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if any(c.strip() for c in r)]

    assert header == ["candidate_id", "rank", "score", "reasoning"], f"Bad header: {header}"
    assert len(rows) == 48, f"Expected 48 rows from the 50-sample minus 2 honeypots, got {len(rows)}"

    ranks = [int(r[1]) for r in rows]
    assert ranks == sorted(ranks), "Ranks must be in ascending order"
    assert ranks == list(range(1, len(rows) + 1)), "Ranks must be contiguous starting at 1"

    scores = [float(r[2]) for r in rows]
    assert all(a >= b for a, b in zip(scores, scores[1:])), "Score must be non-increasing by rank"

    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate candidate_id in output"
    for cid in ids:
        assert cid.startswith("CAND_") and len(cid) == 12, f"Malformed candidate_id: {cid}"

    out_path.unlink(missing_ok=True)


def test_rank_top_n_always_returns_requested_count():
    candidates = _load()
    scores = score_all_candidates(candidates)
    n = min(20, len(candidates) - 5)
    top = rank_top_n(scores, n=n)
    assert len(top) == n, f"Expected exactly {n} rows, got {len(top)}"


def test_rank_top_n_scores_non_increasing():
    candidates = _load()
    scores = score_all_candidates(candidates)
    top = rank_top_n(scores, n=40)
    for a, b in zip(top, top[1:]):
        assert round(a.final_score, 8) >= round(b.final_score, 8), (
            f"Score increased: {a.candidate_id}={a.final_score:.8f} -> "
            f"{b.candidate_id}={b.final_score:.8f}"
        )


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    return failed == 0


if __name__ == "__main__":
    success = _run_all()
    sys.exit(0 if success else 1)