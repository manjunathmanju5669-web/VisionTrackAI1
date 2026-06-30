"""
Structured scoring components: experience fit, location fit, education, and
career trajectory quality. These are the "easy" structured fields - no NLP
needed - but still encode JD-specific judgment calls (e.g. the JD's
experience band is a soft preference, not a hard cutoff; "willing to
relocate" should count almost as much as already being in a target city).
"""

from __future__ import annotations
from dataclasses import dataclass

from jd_profile import EXPERIENCE_SWEET_SPOT, EXPERIENCE_IDEAL, TARGET_CITIES


def score_experience(candidate: dict) -> float:
    """0-1. JD: 5-9yr band stated as flexible, 6-8yr called out as 'ideal'.
    Use a smooth trapezoid rather than a hard cutoff - the JD itself says
    "we'll seriously consider candidates outside the band if other signals
    are strong", so experience should shape score, not gate it outright."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    ideal_lo, ideal_hi = EXPERIENCE_IDEAL
    band_lo, band_hi = EXPERIENCE_SWEET_SPOT

    if ideal_lo <= yoe <= ideal_hi:
        return 1.0
    if band_lo <= yoe <= band_hi:
        # inside the stated band but outside the "ideal" sub-band
        return 0.85
    # outside the band entirely - decay smoothly, never to zero, since the
    # JD explicitly keeps the door open for strong outliers.
    if yoe < band_lo:
        gap = band_lo - yoe
        return max(0.85 - gap * 0.12, 0.15)
    gap = yoe - band_hi
    return max(0.85 - gap * 0.08, 0.20)


def score_location(candidate: dict) -> tuple[float, str]:
    """0-1 plus a short reasoning fragment. JD: Pune/Noida preferred,
    Hyderabad/Pune/Mumbai/Delhi NCR welcome, outside India case-by-case with
    no visa sponsorship. Willingness to relocate substitutes for current
    location, since the JD explicitly frames this as relocation-friendly."""
    profile = candidate.get("profile", {})
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    in_target_city = any(city in location for city in TARGET_CITIES)
    in_india = "india" in country or any(
        c in location for c in ["india", "bengaluru", "bangalore", "chennai",
                                  "hyderabad", "pune", "noida", "mumbai", "delhi",
                                  "gurgaon", "gurugram", "kolkata", "ahmedabad"]
    )

    raw_location = profile.get('location') or "an unspecified location"
    if in_target_city:
        return 1.0, f"based in {raw_location}, a target city for this role"
    if in_india and willing:
        return 0.85, f"based in {raw_location}, India, and willing to relocate"
    if in_india:
        return 0.55, f"based in {raw_location}, India but relocation preference not confirmed"
    if not in_india and willing:
        return 0.35, f"based outside India ({raw_location}) but willing to relocate; no visa sponsorship available per JD"
    return 0.10, f"based outside India ({raw_location}), not confirmed willing to relocate, no visa sponsorship"


TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.55, "tier_4": 0.40, "unknown": 0.45}


def score_education(candidate: dict) -> float:
    """0-1. The JD never mentions pedigree as a requirement - it's a very
    minor signal here by design (low weight in ScoringWeights), present
    mainly as a tiebreaker, since the JD's whole framing is anti-credentialist
    ('skills are teachable')."""
    education = candidate.get("education", [])
    if not education:
        return 0.4  # no penalty for missing data, just neutral-low

    best_tier_score = max(TIER_SCORE.get(e.get("tier", "unknown"), 0.45) for e in education)

    relevant_fields = {"computer science", "data science", "machine learning",
                        "artificial intelligence", "statistics", "mathematics",
                        "information technology", "software engineering"}
    has_relevant_field = any(
        any(rf in (e.get("field_of_study") or "").lower() for rf in relevant_fields)
        for e in education
    )
    field_bonus = 0.15 if has_relevant_field else 0.0

    return min(best_tier_score * 0.85 + field_bonus, 1.0)


def score_career_trajectory(candidate: dict) -> tuple[float, str]:
    """0-1. JD wants: shipped end-to-end systems at meaningful scale,
    product-company experience (not pure services), reasonable tenure
    (not pure title-chasing, but also not stagnant). This is a softer,
    more holistic signal than the hard disqualifiers - rewards depth and
    product-company exposure rather than penalizing absence of red flags."""
    history = candidate.get("career_history", [])
    if not history:
        return 0.3, "no career history on file"

    durations = [r.get("duration_months") or 0 for r in history]
    avg_tenure = sum(durations) / len(durations)
    # sweet spot ~18-48mo average tenure: long enough to ship things,
    # not so long it signals stagnation at a single level for a decade.
    if 18 <= avg_tenure <= 48:
        tenure_score = 1.0
    elif avg_tenure < 18:
        tenure_score = max(0.4, avg_tenure / 18)
    else:
        tenure_score = max(0.5, 1.0 - (avg_tenure - 48) / 96)

    company_sizes = [r.get("company_size", "") for r in history]
    # treat 1-200 employee companies and well-known product companies as
    # "product company" signal; this is a coarse proxy since we don't have
    # an explicit "is_product_company" field in the schema.
    services_keywords = ("it services", "consulting", "outsourcing", "bpo")
    industries = [(r.get("industry") or "").lower() for r in history]
    services_fraction = sum(1 for i in industries if any(k in i for k in services_keywords)) / len(industries)
    product_score = 1.0 - services_fraction

    scale_keywords = ("scale", "million users", "production", "real users", "shipped")
    history_text = " ".join(r.get("description", "") for r in history).lower()
    scale_signal = 1.0 if any(k in history_text for k in scale_keywords) else 0.5

    overall = tenure_score * 0.35 + product_score * 0.40 + scale_signal * 0.25

    fragments = []
    fragments.append(f"avg tenure {avg_tenure:.0f}mo across {len(history)} roles")
    if services_fraction > 0.5:
        fragments.append("majority of career at services/consulting firms")
    elif services_fraction == 0:
        fragments.append("entirely product-company background")

    return overall, "; ".join(fragments)