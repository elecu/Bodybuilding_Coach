from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from ..utils.time import parse_date, days_until


@dataclass(frozen=True)
class PrepSummary:
    days_to_show: Optional[int]
    phase_hint: str
    weight_loss_rate_hint: str
    peak_week_hint: str


def build_prep_summary(
    competition_date: Optional[str],
    mode: str,
    start_weight: Optional[float],
    current_weight: Optional[float],
    start_date: Optional[str],
) -> PrepSummary:
    d = parse_date(competition_date or "")
    dts = days_until(d)

    # Evidence-based defaults (high level):
    # - Typical fat loss rate for contest prep to preserve lean mass: ~0.5–1% BW per week.
    #   (Helms et al., 2014)
    # - Peak week: common strategies exist, but evidence is limited and aggressive dehydration/sodium
    #   manipulation can be risky; avoid extreme tactics.
    #   (Escalante et al., 2021)

    if mode == "cutting":
        phase_hint = "Cutting phase: aim for a steady, manageable rate of loss and keep posing practice consistent."
        weight_loss_rate_hint = "Target rate: roughly 0.5–1% of bodyweight per week (trend-based)."
    elif mode == "bulking":
        phase_hint = "Bulking phase: keep surplus modest, monitor waist trend, and prioritise weak points."
        weight_loss_rate_hint = "If you have a show date, plan a cut window early enough to avoid a crash diet."
    else:
        phase_hint = "Maintenance: keep bodyweight stable while improving posing and weak points."
        weight_loss_rate_hint = "If you set a show date, the app will estimate a sensible runway."

    if dts is not None and dts <= 14:
        peak_week_hint = (
            "Peak fortnight: keep changes conservative. Evidence for extreme water/sodium tricks is limited; "
            "avoid risky dehydration. Focus on sleep, digestion, and rehearsing stage routine."
        )
    elif dts is not None and dts <= 60:
        peak_week_hint = (
            "Final 8 weeks: refine conditioning while protecting training performance; practise compulsory poses often."
        )
    else:
        peak_week_hint = "Set a show date to unlock a countdown and time-based check-ins."

    return PrepSummary(
        days_to_show=dts,
        phase_hint=phase_hint,
        weight_loss_rate_hint=weight_loss_rate_hint,
        peak_week_hint=peak_week_hint,
    )
