"""
Honeypot / internal-consistency detector.

From README.md: "The dataset contains traps. Keyword stuffers, plain-language
Tier 5s, behavioral twins, and ~80 honeypots with subtly impossible profiles."
From submission_spec.md Section 7: honeypots have profiles like "8 years of
experience at a company founded 3 years ago" or "'expert' proficiency in 10
skills with 0 years used." If honeypot rate in our top 100 exceeds 10% we are
disqualified at Stage 3 regardless of composite score - so this check runs
BEFORE ranking and removes/heavily penalizes flagged candidates rather than
hoping the scoring model naturally avoids them.

This is deliberately a set of explicit, auditable checks rather than a
learned model: we don't have labeled honeypots to train against, and a
hand-written checklist is exactly the kind of thing we can defend candidate-
by-candidate in the Stage 5 interview ("why did you flag this person").

We don't have ground-truth company founding dates in the schema, so the
"tenure exceeds company age" check from the spec's example can't be applied
literally. Instead we apply the checks the schema *does* let us verify:
internal date/duration arithmetic, skill-proficiency-vs-duration mismatches,
and a few other impossibility classes. If candidates.jsonl turns out to
include a company-founding-date field we don't have in the sample, extend
check 1 below.
"""

from __future__ import annotations
from datetime import date, datetime
from dataclasses import dataclass


@dataclass
class HoneypotFlags:
    candidate_id: str
    flags: list[str]

    @property
    def is_honeypot(self) -> bool:
        return len(self.flags) > 0

    @property
    def severity(self) -> int:
        """Number of independent impossibility classes triggered. Used to
        rank confidence when we need to decide whether a borderline case is
        worth excluding vs just down-weighting."""
        return len(self.flags)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def check_career_arithmetic(candidate: dict) -> list[str]:
    """Career-history dates/durations must be internally consistent, and
    total career time must not wildly exceed stated years_of_experience."""
    flags = []
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    history = candidate.get("career_history", [])

    total_months = 0
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date")) or date.today()
        stated_duration = role.get("duration_months", 0) or 0
        total_months += stated_duration

        if start and end:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            # allow +/- 2 months slack for day-of-month rounding
            if abs(actual_months - stated_duration) > 2:
                flags.append(
                    f"role at {role.get('company','?')}: stated duration "
                    f"{stated_duration}mo doesn't match dates ({actual_months}mo)"
                )
            if start > end:
                flags.append(f"role at {role.get('company','?')}: start date after end date")

    # total career time shouldn't exceed stated YOE by more than ~12 months
    # (a little slack for concurrent/overlapping roles, internships, etc.)
    if total_months > (yoe * 12) + 12:
        flags.append(
            f"career history totals {total_months}mo but years_of_experience "
            f"is {yoe} ({yoe*12:.0f}mo) - {total_months - yoe*12:.0f}mo unaccounted"
        )

    return flags


def check_skill_duration_mismatch(candidate: dict) -> list[str]:
    """'Expert' or 'advanced' proficiency claimed with near-zero duration_months
    is the clearest version of the spec's 'expert in 10 skills with 0 years
    used' example."""
    flags = []
    expert_zero_count = 0
    for skill in candidate.get("skills", []):
        prof = skill.get("proficiency")
        dur = skill.get("duration_months")
        if dur is None:
            continue
        if prof == "expert" and dur <= 2:
            expert_zero_count += 1
        elif prof == "advanced" and dur <= 1:
            expert_zero_count += 1

    # one or two could be a genuinely fast learner; a pattern across many
    # skills in the same profile is the honeypot signature.
    if expert_zero_count >= 3:
        flags.append(
            f"{expert_zero_count} skills claim expert/advanced proficiency "
            f"with <=1-2 months duration"
        )
    return flags


def check_education_arithmetic(candidate: dict) -> list[str]:
    """Education years must be internally consistent and plausible given
    years_of_experience (can't have 10 years of work history that predates
    your only degree by decades in a way that contradicts the stated total)."""
    flags = []
    for edu in candidate.get("education", []):
        start, end = edu.get("start_year"), edu.get("end_year")
        if start and end and end < start:
            flags.append(f"education at {edu.get('institution','?')}: end_year before start_year")
        if start and end and (end - start) > 8:
            flags.append(f"education at {edu.get('institution','?')}: implausible {end-start}-year program")
    return flags


def check_signal_consistency(candidate: dict) -> list[str]:
    """Redrob signal fields that contradict each other within the same
    profile (not career-history based, but still an internal-consistency
    check, same spirit as the honeypot examples)."""
    flags = []
    sig = candidate.get("redrob_signals", {})

    last_active = _parse_date(sig.get("last_active_date"))
    signup = _parse_date(sig.get("signup_date"))
    if last_active and signup and last_active < signup:
        flags.append("last_active_date is before signup_date")

    return flags


def check_salary_inversion(candidate: dict) -> list[str]:
    """expected_salary min > max. NOTE: measured at ~30% base rate on the
    50-candidate sample - far too common to be the rare (~80/100,000)
    honeypot signal described in the spec. Kept as its OWN check, separate
    from the honeypot-class checks below, because it looks like generator
    noise rather than a deliberate trap. It contributes supporting evidence
    only and never trips honeypot status on its own (see SUPPORTING_CHECKS)."""
    flags = []
    salary = candidate.get("redrob_signals", {}).get("expected_salary_range_inr_lpa", {}) or {}
    lo, hi = salary.get("min"), salary.get("max")
    if lo is not None and hi is not None and lo > hi:
        flags.append(f"expected_salary min ({lo}) > max ({hi})")
    return flags


HONEYPOT_CHECKS = [
    check_career_arithmetic,
    check_skill_duration_mismatch,
    check_education_arithmetic,
    check_signal_consistency,
]

SUPPORTING_CHECKS = [
    check_salary_inversion,
]


def detect_honeypot(candidate: dict) -> HoneypotFlags:
    core_flags: list[str] = []
    for check in HONEYPOT_CHECKS:
        core_flags.extend(check(candidate))

    supporting_flags: list[str] = []
    for check in SUPPORTING_CHECKS:
        supporting_flags.extend(check(candidate))

    is_flagged = len(core_flags) >= 1

    return HoneypotFlags(
        candidate_id=candidate.get("candidate_id", "?"),
        flags=(core_flags + supporting_flags) if is_flagged else [],
    )