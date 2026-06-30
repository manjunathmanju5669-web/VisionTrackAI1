"""
Skill-match scoring with anti-keyword-stuffing trust weighting.

The JD is explicit: "the right answer is not 'find candidates whose skills
section contains the most AI keywords'." From the CAND_0000021 case found
during dev (a Project Manager with 'advanced Fine-tuning LLMs' at 4 months
duration and 'advanced Recommendation Systems' at 13 months, sitting next to
beginner-proficiency unrelated skills) - a raw skill-name match would rank
this person highly. The fix: every skill match is weighted by a "trust"
factor derived from proficiency, duration_months, and endorsements, so a
skill someone has had for 4 months with no endorsements counts for much
less than the same skill with 60 months and 40 endorsements.
"""

from __future__ import annotations
from dataclasses import dataclass

from jd_profile import REQUIRED_SKILL_FAMILIES, NICE_TO_HAVE_SKILLS

PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}


@dataclass
class SkillMatchResult:
    required_family_hits: dict[str, float]   # family -> best trust-weighted match score (0-1)
    nice_to_have_hits: list[tuple[str, float]]
    matched_skill_names: list[str]            # for reasoning generation
    score: float                              # 0-1 overall


def _skill_trust(skill: dict) -> float:
    """How much weight to give a claimed skill, independent of whether the
    JD wants it. Combines proficiency, duration, and endorsements - a skill
    needs corroboration across more than one of these to score highly. This
    is the mechanism that down-weights keyword-stuffed recent skills."""
    prof_w = PROFICIENCY_WEIGHT.get(skill.get("proficiency"), 0.25)

    duration = skill.get("duration_months") or 0
    # duration trust ramps up to 1.0 at 36 months, doesn't punish short
    # duration on legitimately-junior skills, just refuses to give a skill
    # full trust on duration alone until ~3 years of claimed use.
    duration_w = min(duration / 36.0, 1.0)

    endorsements = skill.get("endorsements") or 0
    endorsement_w = min(endorsements / 20.0, 1.0)

    # Require BOTH a duration signal and at least some endorsement signal to
    # approach full trust; a skill with high proficiency claimed but zero
    # duration and zero endorsements should score low regardless of the
    # proficiency label, since proficiency is self-reported.
    corroboration = (duration_w * 0.6) + (endorsement_w * 0.4)
    return prof_w * (0.4 + 0.6 * corroboration)  # proficiency alone floors at 0.4x


def score_skills(candidate: dict) -> SkillMatchResult:
    skills = candidate.get("skills", [])
    skill_index = {s["name"].strip().lower(): s for s in skills if s.get("name")}

    required_hits: dict[str, float] = {}
    matched_names: list[str] = []

    for family, names in REQUIRED_SKILL_FAMILIES.items():
        best = 0.0
        best_name = None
        for name in names:
            if name in skill_index:
                trust = _skill_trust(skill_index[name])
                if trust > best:
                    best = trust
                    best_name = name
        required_hits[family] = best
        if best_name:
            matched_names.append(skill_index[best_name]["name"])

    nice_hits: list[tuple[str, float]] = []
    for name in NICE_TO_HAVE_SKILLS:
        if name in skill_index:
            nice_hits.append((skill_index[name]["name"], _skill_trust(skill_index[name])))

    # also pull credit from career_history descriptions even when the exact
    # skill word isn't in the skills list - this is the JD's "Tier 5
    # candidate may not use the words 'RAG' or 'Pinecone' ... but if their
    # career history shows they built a recommendation system" case.
    history_text = " ".join(
        role.get("description", "") for role in candidate.get("career_history", [])
    ).lower()
    history_signal_terms = {
        "embeddings": ["embedding", "vector representation"],
        "vector_db": ["vector search", "vector database", "similarity search", "ann index", "faiss", "elasticsearch", "opensearch"],
        "eval": ["ndcg", "offline evaluation", "a/b test", "learning to rank", "learning-to-rank", "relevance labeling"],
        # Python rescue: only 6% of sample candidates list "Python" as a
        # literal skill entry even though it's a stated must-have. Python
        # usage is more often implied by adjacent tools (pyspark, pandas,
        # scikit-learn, django, flask) or named directly in a role
        # description than added as its own skill-list row, so check both.
        "python": ["python", "pyspark", "pandas", "django", "flask", "scikit-learn"],
    }
    # IMPORTANT - regression note: an earlier version also rescued from a
    # bare adjacent-skill-list match (e.g. "Django" appearing anywhere in
    # skills) without requiring career-history corroboration. That let
    # irrelevant titles (Mechanical Engineer, Graphic Designer with Django
    # somewhere in a long, mostly-irrelevant skill list) pick up a flat
    # 0.55 "python" credit and climb into the top 10 alongside genuinely
    # relevant engineers. Adjacent-tool rescue now REQUIRES the same term
    # (or a same-family term) to also appear in the career_history
    # description text, not just the skills list - a tool name sitting
    # alone in a long unrelated skill list is not corroboration.
    for family, terms in history_signal_terms.items():
        if required_hits.get(family, 0) < 0.5 and any(t in history_text for t in terms):
            required_hits[family] = max(required_hits.get(family, 0), 0.55)

    n_families = len(REQUIRED_SKILL_FAMILIES)
    required_score = sum(required_hits.values()) / n_families if n_families else 0.0
    nice_bonus = min(sum(v for _, v in nice_hits) / 10.0, 0.15)  # small bonus, capped

    overall = min(required_score * 0.85 + nice_bonus, 1.0)

    return SkillMatchResult(
        required_family_hits=required_hits,
        nice_to_have_hits=nice_hits,
        matched_skill_names=matched_names,
        score=overall,
    )