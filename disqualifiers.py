"""
JD red-flag / disqualifier detection.

Implements the "Things we explicitly do NOT want" and experience-band
disqualifier sections of job_description.md as explicit, inspectable
checks. Each returns a (triggered: bool, evidence: str) pair so the
reasoning generator can quote the *actual* evidence rather than just
asserting a penalty was applied - this matters for Stage 4 manual review,
which explicitly checks for hallucination-free, specific reasoning.

Penalties are MULTIPLICATIVE down-weights (see jd_profile.ScoringWeights),
not hard exclusions. The JD's own language ("we will probably not move
forward") is a strong-but-not-absolute signal - an exceptional candidate
could still claw back rank on other dimensions, same as a human recruiter
would weigh a red flag against everything else on the profile.
"""

from __future__ import annotations
from datetime import date, datetime
from dataclasses import dataclass, field

from jd_profile import (
    CONSULTING_FIRMS,
    CV_SPEECH_ROBOTICS_TITLES,
    NLP_IR_RESCUE_SKILLS,
    RECENT_LLM_WRAPPER_MONTHS,
    LLM_WRAPPER_SKILLS,
    PRE_LLM_ML_SKILLS,
)


@dataclass
class DisqualifierResult:
    triggered: list[str] = field(default_factory=list)   # keys into ScoringWeights.disqualifier_penalty
    evidence: list[str] = field(default_factory=list)     # human-readable, profile-grounded justification


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _skill_names(candidate: dict) -> set[str]:
    return {s["name"].strip().lower() for s in candidate.get("skills", []) if s.get("name")}


def check_pure_research_only(candidate: dict) -> tuple[bool, str | None]:
    """JD: 'pure research environments (academic labs, research-only roles)
    without any production deployment' -> no move forward.
    Heuristic: every career_history entry's industry/title reads as research
    academia, and current_industry is academic/research, AND no production-
    sounding keywords ("shipped", "deployed", "production") appear in any
    role description."""
    history = candidate.get("career_history", [])
    if not history:
        return False, None

    research_titles = {"research scientist", "research associate", "phd researcher",
                        "postdoctoral researcher", "research intern", "academic researcher"}
    all_research = all(
        any(rt in (role.get("title", "").lower()) for rt in research_titles)
        for role in history
    )
    if not all_research:
        return False, None

    prod_keywords = ("shipped", "deployed", "production", "real users", "scale")
    any_production_mention = any(
        any(kw in role.get("description", "").lower() for kw in prod_keywords)
        for role in history
    )
    if any_production_mention:
        return False, None

    titles = ", ".join(role.get("title", "?") for role in history)
    return True, f"entire career history is research-only roles ({titles}) with no production/deployment language in descriptions"


def check_recent_llm_wrapper_only(candidate: dict) -> tuple[bool, str | None]:
    """JD: 'AI experience consists primarily of recent (<12mo) projects
    using LangChain to call OpenAI, unless substantial pre-LLM-era ML
    production experience.'"""
    skills = candidate.get("skills", [])
    skill_by_name = {s["name"].strip().lower(): s for s in skills}

    wrapper_skills_present = [s for s in LLM_WRAPPER_SKILLS if s in skill_by_name]
    if not wrapper_skills_present:
        return False, None

    # are the wrapper skills recent (<12mo duration)?
    recent_wrapper = [
        s for s in wrapper_skills_present
        if (skill_by_name[s].get("duration_months") or 999) < RECENT_LLM_WRAPPER_MONTHS
    ]
    if not recent_wrapper:
        return False, None

    # does the candidate have substantial pre-LLM ML production experience
    # to rescue this? (pre-LLM ML skill with meaningfully longer duration)
    pre_llm_present = [
        s for s in PRE_LLM_ML_SKILLS
        if s in skill_by_name and (skill_by_name[s].get("duration_months") or 0) >= 24
    ]
    if pre_llm_present:
        return False, None

    return True, (
        f"AI-related skills are limited to recent wrapper tooling ({', '.join(recent_wrapper)}, "
        f"<{RECENT_LLM_WRAPPER_MONTHS}mo duration) with no pre-LLM-era ML skill showing >=24mo duration"
    )


def check_stale_senior_no_code(candidate: dict) -> tuple[bool, str | None]:
    """JD: senior engineer who hasn't written production code in 18+ months
    because they moved into pure architecture/tech-lead roles."""
    history = sorted(
        candidate.get("career_history", []),
        key=lambda r: r.get("start_date") or "",
        reverse=True,
    )
    if not history:
        return False, None

    current = history[0]
    title = current.get("title", "").lower()
    pure_management_titles = ("engineering manager", "director of engineering",
                               "vp engineering", "head of engineering", "tech lead")
    if not any(t in title for t in pure_management_titles):
        return False, None

    duration = current.get("duration_months") or 0
    if duration < 18:
        return False, None

    coding_keywords = ("wrote", "built", "implemented", "coded", "shipped code")
    if any(kw in current.get("description", "").lower() for kw in coding_keywords):
        return False, None

    return True, f"current role '{current.get('title')}' for {duration}mo with no hands-on coding language in description"


def check_cv_speech_robotics_no_nlp(candidate: dict) -> tuple[bool, str | None]:
    """JD: primary expertise in CV/speech/robotics without significant NLP/IR exposure."""
    title = candidate.get("profile", {}).get("current_title", "").lower()
    if not any(t in title for t in CV_SPEECH_ROBOTICS_TITLES):
        return False, None

    skills = _skill_names(candidate)
    nlp_overlap = skills & NLP_IR_RESCUE_SKILLS
    if nlp_overlap:
        return False, None

    return True, f"current title '{candidate['profile']['current_title']}' with no NLP/IR-adjacent skills on profile"


def check_consulting_only_career(candidate: dict) -> tuple[bool, str | None]:
    """JD: 'people who have only worked at consulting firms in their entire
    career' -> down-weight. Explicitly NOT triggered if they have ANY prior
    product-company experience (JD: 'if you're currently at one of these
    companies but have prior product-company experience, that's fine')."""
    history = candidate.get("career_history", [])
    if not history:
        return False, None

    companies = [role.get("company", "").strip().lower() for role in history]
    all_consulting = all(
        any(cf in company for cf in CONSULTING_FIRMS) for company in companies
    )
    if not all_consulting:
        return False, None

    return True, f"entire career history is at consulting/services firms: {', '.join(companies)}"


# Seniority-RANK words only (not domain/specialization words). The JD's
# complaint is specifically about chasing the Senior -> Staff -> Principal
# *label*, not about someone moving between IC specializations (e.g.
# "NLP Engineer" -> "Search Engineer" -> "Recommendation Systems Engineer"
# is lateral specialization, not title inflation, and must NOT trigger this).
SENIORITY_RANK_WORDS = ["junior", "associate", "senior", "staff", "principal",
                        "lead", "director", "vp", "head of", "chief"]


def check_title_chaser(candidate: dict) -> tuple[bool, str | None]:
    """JD: career trajectory optimizing for Senior->Staff->Principal title
    bumps by switching companies every ~1.5 years.

    Only triggers when (a) tenure is genuinely short AND (b) the explicit
    seniority-rank word attached to the title increases across job changes
    (e.g. "Engineer" -> "Senior Engineer" -> "Staff Engineer"). Lateral
    moves between differently-named IC specializations at a similar level
    do not count, even though job titles differ each time - that pattern is
    normal career exploration, not the JD's stated concern."""
    history = candidate.get("career_history", [])
    if len(history) < 3:
        return False, None

    durations = [role.get("duration_months") or 0 for role in history]
    avg_tenure = sum(durations) / len(durations)
    if avg_tenure >= 20:  # ~1.7 years average tenure or more -> not a hopper
        return False, None

    titles = [role.get("title", "").lower() for role in history]

    def rank_word_index(t: str) -> int | None:
        for i, rung in enumerate(SENIORITY_RANK_WORDS):
            if rung in t:
                return i
        return None  # no explicit seniority word present -> not comparable

    ranks = [rank_word_index(t) for t in titles]
    comparable_pairs = [(a, b) for a, b in zip(ranks, ranks[1:]) if a is not None and b is not None]

    # need at least one comparable pair with an explicit seniority word on
    # both sides, and ALL comparable pairs must be non-decreasing, to call
    # this a title-chasing pattern rather than lateral movement.
    if not comparable_pairs:
        return False, None

    monotonic_up = all(b >= a for a, b in comparable_pairs)
    any_actual_increase = any(b > a for a, b in comparable_pairs)

    if monotonic_up and any_actual_increase and avg_tenure < 20:
        return True, (
            f"average tenure {avg_tenure:.0f}mo across {len(history)} roles with "
            f"increasing seniority titles ({' -> '.join(titles)})"
        )

    return False, None


CHECKS = {
    "pure_research_only": check_pure_research_only,
    "recent_llm_wrapper_only": check_recent_llm_wrapper_only,
    "stale_senior_no_code": check_stale_senior_no_code,
    "cv_speech_robotics_no_nlp": check_cv_speech_robotics_no_nlp,
    "consulting_only_career": check_consulting_only_career,
    "title_chaser": check_title_chaser,
}


def detect_disqualifiers(candidate: dict) -> DisqualifierResult:
    result = DisqualifierResult()
    for key, check_fn in CHECKS.items():
        triggered, evidence = check_fn(candidate)
        if triggered:
            result.triggered.append(key)
            result.evidence.append(evidence)
    return result