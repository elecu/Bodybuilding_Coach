from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

Federation = Literal["WNBF_UK", "PCA"]


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
    "PCA": FederationRules(
        key="PCA",
        display_name="Physical Culture Association (PCA)",
        url="https://www.pcaofficial.com/",
        notes=(
            "UK federation with detailed class criteria and compulsory poses; many shows include First Timers classes."
        ),
    ),
}


def cycle_federation(current: Federation) -> Federation:
    return "PCA" if current == "WNBF_UK" else "WNBF_UK"
