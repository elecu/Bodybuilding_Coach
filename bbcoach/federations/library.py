from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

Federation = Literal["WNBF_UK", "PCA", "UKBFF", "BNBF", "NABBA"]


@dataclass(frozen=True)
class FederationRules:
    key: Federation
    display_name: str
    url: str
    notes: str


RULES: Dict[Federation, FederationRules] = {
    "WNBF_UK": FederationRules(
        key="WNBF_UK",
        display_name="WNBF UK",
        url="https://wnbfuk.com/",
        notes=(
            "Natural bodybuilding federation. Judging criteria varies by division; men's physique typically "
            "emphasises aesthetics and V-taper rather than extreme mass."
        ),
    ),
    "UKBFF": FederationRules(
        key="UKBFF",
        display_name="UKBFF (IFBB)",
        url="https://www.ukbff.co.uk/",
        notes=(
            "UKBFF/IFBB affiliate. Men's Physique is typically judged via quarter turns with a final front pose."
        ),
    ),
    "PCA": FederationRules(
        key="PCA",
        display_name="Physical Culture Association (PCA)",
        url="https://www.pcaofficial.com/",
        notes=(
            "UK federation with detailed class criteria and compulsory poses; many shows include First Timers classes."
        ),
    ),
    "BNBF": FederationRules(
        key="BNBF",
        display_name="BNBF UK",
        url="https://bnbf.co.uk/competition/judging-criteria",
        notes=(
            "Natural bodybuilding federation with symmetry/muscularity rounds and routine rounds by division."
        ),
    ),
    "NABBA": FederationRules(
        key="NABBA",
        display_name="NABBA UK",
        url="https://nabbaofficial.com/",
        notes=(
            "Federation with quarter-turn rounds, compulsory comparisons, and routine rounds that vary by show level."
        ),
    ),
}


def cycle_federation(current: Federation) -> Federation:
    order = ["WNBF_UK", "UKBFF", "PCA", "BNBF", "NABBA"]
    try:
        idx = order.index(current)
    except ValueError:
        return "WNBF_UK"
    return order[(idx + 1) % len(order)]
