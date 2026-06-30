"""
Reasoning generator for the CSV's `reasoning` column.

Stage 4 manual review (submission_spec.md Section 3) samples 10 rows and
checks each reasoning entry for: specific facts from the profile, JD
connection, honest acknowledgment of gaps, no hallucination, variation
across samples, and rank-consistent tone. This module is built around
those six checks directly - every sentence it emits is generated FROM a
field we actually scored, never from a fixed template string, and gaps
flagged by disqualifiers/honeypot/behavioral scoring are always surfaced
rather than hidden behind generic praise.
"""

from __future__ import annotations
from ranker import CandidateScore


def _cap_first(s: str) -> str:
    """Capitalize only the first character, unlike str.capitalize() which
    also lowercases every other character in the string - that mangled
    proper nouns (e.g. "Hyderabad, Telangana" -> "Hyderabad, telangana")
    when applied to reasoning fragments that quote location/company names
    verbatim from the profile."""
    return s[:1].upper() + s[1:] if s else s


def _format_skill_fragment(matched_skills: list[str]) -> str | None:
    if not matched_skills:
        return None
    if len(matched_skills) == 1:
        return f"has {matched_skills[0]} on their profile"
    return f"has {', '.join(matched_skills[:3])} on their profile"


def generate_reasoning(candidate: dict, score: CandidateScore) -> str:
    """Build a 1-2 sentence reasoning string grounded entirely in fields
    we actually examined for this specific candidate."""
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "their current role")
    yoe = profile.get("years_of_experience")
    company = profile.get("current_company", "")

    sentence_parts = []

    lead = f"{title}"
    if yoe is not None:
        lead += f" with {yoe:.1f} years experience"
    if company:
        lead += f" at {company}"

    skill_fragment = _format_skill_fragment(score.matched_skills)
    if skill_fragment and score.skill_score >= 0.4:
        lead += f"; {skill_fragment} with meaningful tenure/endorsements behind it"
    elif skill_fragment and score.skill_score < 0.4:
        lead += f"; lists {', '.join(score.matched_skills[:2])} but with limited duration/endorsement backing"

    sentence_parts.append(lead + ".")

    concerns = []
    if score.disqualifier_evidence:
        concerns.extend(score.disqualifier_evidence)
    if score.honeypot_evidence:
        concerns.append(f"profile-consistency concern: {score.honeypot_evidence[0]}")
    if score.behavioral_multiplier < 0.5:
        concerns.append(f"availability concern ({score.behavioral_evidence})")

    if concerns:
        sentence_parts.append(_cap_first(concerns[0]) + ".")
    else:
        positives = []
        if score.location >= 0.85:
            positives.append(score.location_evidence)
        if score.trajectory >= 0.8:
            positives.append("strong product-company career trajectory")
        if score.semantic >= 0.65:
            positives.append("profile narrative closely matches the JD's ranking/retrieval mandate")
        if score.behavioral_multiplier >= 0.8:
            positives.append(f"actively engaged on platform ({score.behavioral_evidence})")

        if positives:
            sentence_parts.append(_cap_first(positives[0]) + ".")
        elif score.behavioral_evidence:
            sentence_parts.append(_cap_first(score.behavioral_evidence) + ".")

    return " ".join(sentence_parts)