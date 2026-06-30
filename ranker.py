"""
Main ranker: combines semantic similarity, skill matching, structured
scoring, disqualifiers, behavioral multiplier, and honeypot filtering into
a single ranked list with grounded, per-candidate reasoning.

Composite formula (weights live in jd_profile.ScoringWeights):

  profile_fit = w.semantic_similarity * semantic
              + w.skill_match         * skills
              + w.experience_fit      * experience
              + w.career_trajectory   * trajectory
              + w.location            * location
              + w.education           * education

  disqualifier_multiplier = product of triggered disqualifier penalties

  final_score = profile_fit * disqualifier_multiplier * behavioral_multiplier

Honeypots are excluded entirely before ranking (not just down-weighted) -
the spec's >10%-in-top-100 cutoff is a hard disqualification regardless of
score, so the safest design is to never let a flagged candidate occupy a
top-100 slot in the first place.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

from jd_profile import JD_TEXT, ScoringWeights
from semantic_similarity import SemanticSimilarityScorer, candidate_text
from skill_matching import score_skills
from structured_scoring import (
    score_experience, score_location, score_education, score_career_trajectory,
)
from disqualifiers import detect_disqualifiers
from behavioral_scoring import score_behavioral
from honeypot import detect_honeypot


@dataclass
class CandidateScore:
    candidate_id: str
    final_score: float
    profile_fit: float
    semantic: float
    skill_score: float
    experience: float
    trajectory: float
    location: float
    education: float
    disqualifier_multiplier: float
    disqualifiers_triggered: list[str]
    disqualifier_evidence: list[str]
    behavioral_multiplier: float
    behavioral_evidence: str
    location_evidence: str
    matched_skills: list[str]
    is_honeypot: bool
    honeypot_evidence: list[str]


def score_all_candidates(
    candidates: list[dict],
    weights: ScoringWeights | None = None,
) -> list[CandidateScore]:
    """Score every candidate against the JD. Honeypots are still scored
    (so we can report/audit them) but get is_honeypot=True so the ranking
    step can exclude them before producing the top-100."""
    weights = weights or ScoringWeights()

    texts = [candidate_text(c) for c in candidates]
    semantic_scorer = SemanticSimilarityScorer()
    semantic_scorer.fit(JD_TEXT, texts)
    semantic_scores = semantic_scorer.scores()

    results = []
    for candidate, semantic in zip(candidates, semantic_scores):
        cid = candidate.get("candidate_id", "?")

        honeypot = detect_honeypot(candidate)

        skill_result = score_skills(candidate)
        experience = score_experience(candidate)
        trajectory, trajectory_evidence = score_career_trajectory(candidate)
        location, location_evidence = score_location(candidate)
        education = score_education(candidate)

        profile_fit = (
            weights.semantic_similarity * float(semantic)
            + weights.skill_match * skill_result.score
            + weights.experience_fit * experience
            + weights.career_trajectory * trajectory
            + weights.location * location
            + weights.education * education
        )
        weight_sum = (
            weights.semantic_similarity + weights.skill_match + weights.experience_fit
            + weights.career_trajectory + weights.location + weights.education
        )
        profile_fit = profile_fit / weight_sum if weight_sum else profile_fit

        # HARD RELEVANCE GATE - candidates with zero matched required-skill-
        # family signal were landing in the top 10 on generic profile
        # strength alone (good experience band, good tenure, good city,
        # decent degree) despite having nothing to do with AI/ML/retrieval
        # engineering. Below this floor, profile_fit is quadratically
        # suppressed rather than zeroed.
        SKILL_RELEVANCE_FLOOR = 0.15
        if skill_result.score < SKILL_RELEVANCE_FLOOR:
            relevance_ratio = skill_result.score / SKILL_RELEVANCE_FLOOR if SKILL_RELEVANCE_FLOOR else 0.0
            profile_fit *= max(relevance_ratio ** 2, 0.05)

        dq = detect_disqualifiers(candidate)
        dq_multiplier = 1.0
        for key in dq.triggered:
            dq_multiplier *= weights.disqualifier_penalty.get(key, 1.0)

        behavioral_mult, behavioral_evidence = score_behavioral(candidate)

        final_score = profile_fit * dq_multiplier * behavioral_mult

        results.append(CandidateScore(
            candidate_id=cid,
            final_score=final_score,
            profile_fit=profile_fit,
            semantic=float(semantic),
            skill_score=skill_result.score,
            experience=experience,
            trajectory=trajectory,
            location=location,
            education=education,
            disqualifier_multiplier=dq_multiplier,
            disqualifiers_triggered=dq.triggered,
            disqualifier_evidence=dq.evidence,
            behavioral_multiplier=behavioral_mult,
            behavioral_evidence=behavioral_evidence,
            location_evidence=location_evidence,
            matched_skills=skill_result.matched_skill_names,
            is_honeypot=honeypot.is_honeypot,
            honeypot_evidence=honeypot.flags,
        ))

    return results


SCORE_OUTPUT_DECIMALS = 8  # must match the format string in rank.py's CSV writer


def rank_top_n(
    scores: list[CandidateScore],
    n: int = 100,
    exclude_honeypots: bool = True,
) -> list[CandidateScore]:
    """Sort by final_score descending, excluding honeypots, return top n.
    Ties broken by candidate_id ascending. Sorts on the ROUNDED score so
    what's sorted matches what's written to the CSV."""
    clean_pool = [s for s in scores if not (exclude_honeypots and s.is_honeypot)]
    clean_pool.sort(key=lambda s: (-round(s.final_score, SCORE_OUTPUT_DECIMALS), s.candidate_id))

    if exclude_honeypots and len(clean_pool) < n:
        # SAFETY NET: with the real 100K-candidate pool this should never
        # trigger, but if it ever does, backfill with honeypot-flagged
        # candidates so we always return exactly n rows, then re-sort the
        # combined pool so the monotonic-score-by-rank invariant holds.
        flagged_pool = [s for s in scores if s.is_honeypot]
        flagged_pool.sort(key=lambda s: (-round(s.final_score, SCORE_OUTPUT_DECIMALS), s.candidate_id))
        needed = n - len(clean_pool)
        clean_pool = clean_pool + flagged_pool[:needed]
        clean_pool.sort(key=lambda s: (-round(s.final_score, SCORE_OUTPUT_DECIMALS), s.candidate_id))

    return clean_pool[:n]