"""
Behavioral signal multiplier.

JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and has
a 5% recruiter response rate is, for hiring purposes, not actually
available. Down-weight them appropriately."

This is applied MULTIPLICATIVELY on top of the profile-fit composite (see
ranker.py), not as an additive component - a candidate with a great profile
but who is completely unreachable should have their rank suppressed
regardless of how good the rest of their profile is, which an additive
weight wouldn't guarantee (a 0 here could still get outranked by sheer
profile-fit volume in a purely additive model).
"""

from __future__ import annotations
from datetime import date, datetime


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _recency_score(last_active: date | None, today: date) -> float:
    """1.0 if active in the last 30 days, decaying to a floor of 0.15 by
    ~180 days inactive. Never goes fully to 0 - even a stale profile might
    still be worth a recruiter's outreach attempt, just much less likely
    to convert, which the JD's own example treats as a strong but not
    absolute down-weight."""
    if last_active is None:
        return 0.5  # missing data, neutral
    days = (today - last_active).days
    if days <= 30:
        return 1.0
    if days >= 180:
        return 0.15
    # linear decay from 1.0 at 30 days to 0.15 at 180 days
    return 1.0 - (days - 30) / (180 - 30) * 0.85


def score_behavioral(candidate: dict, today: date | None = None) -> tuple[float, str]:
    """Returns (multiplier in [0.15, 1.0]-ish range, reasoning fragment).
    Combines: recency, recruiter response rate, interview completion rate,
    open_to_work flag, and verification status."""
    today = today or date.today()
    sig = candidate.get("redrob_signals", {})

    last_active = _parse_date(sig.get("last_active_date"))
    recency = _recency_score(last_active, today)

    response_rate = sig.get("recruiter_response_rate")
    response_score = response_rate if response_rate is not None else 0.5

    interview_rate = sig.get("interview_completion_rate")
    interview_score = interview_rate if interview_rate is not None else 0.5

    open_to_work = sig.get("open_to_work_flag", False)
    open_score = 1.0 if open_to_work else 0.65  # not fatal, just a smaller boost

    verified = (
        0.5 * (1.0 if sig.get("verified_email") else 0.0)
        + 0.5 * (1.0 if sig.get("verified_phone") else 0.0)
    )
    # verification floors at 0.7, not 0 - unverified contact info is a
    # logistics problem, not evidence the person isn't a fit.
    verified_score = 0.7 + 0.3 * verified

    # weighted combination; recency and response rate matter most since
    # those are the JD's explicit example ("hasn't logged in", "5%
    # response rate").
    multiplier = (
        recency * 0.40
        + response_score * 0.30
        + interview_score * 0.15
        + open_score * 0.10
        + verified_score * 0.05
    )

    fragments = []
    if last_active:
        days = (today - last_active).days
        fragments.append(f"last active {days}d ago")
    if response_rate is not None:
        fragments.append(f"recruiter response rate {response_rate:.0%}")
    if not open_to_work:
        fragments.append("not flagged open-to-work")

    return multiplier, "; ".join(fragments)