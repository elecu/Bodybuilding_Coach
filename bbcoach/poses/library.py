from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PoseDef:
    key: str
    display: str
    # Target angles (degrees) with tolerance for scoring. Keys correspond to computed features.
    target: dict
    tolerance: dict
    guidance: List[str]


# NOTE: These are *generic* templates to get you started.
# The user can capture personalised templates (Space key) to improve scoring.

POSES: Dict[str, PoseDef] = {
    "mp_front": PoseDef(
        key="mp_front",
        display="Men's Physique — Front",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "elbow_sym": 0.0,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
            "elbow_sym": 10.0,
        },
        guidance=[
            "Stand tall. Keep ribcage up and pelvis neutral.",
            "Open lats slightly to show V-taper.",
            "Keep shoulders level and relaxed (no shrugging).",
        ],
    ),
    "mp_back": PoseDef(
        key="mp_back",
        display="Men's Physique — Back",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
        },
        guidance=[
            "Stand tall; show upper back width.",
            "Keep shoulders level.",
        ],
    ),
    "bb_front_double_biceps": PoseDef(
        key="bb_front_double_biceps",
        display="Bodybuilding — Front Double Biceps",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "elbow_height": 0.0,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
            "elbow_height": 12.0,
        },
        guidance=[
            "Raise elbows until biceps peak without shrugging.",
            "Keep waist tight and chest up.",
        ],
    ),
}


# Simple routines for live coaching.
ROUTINES: Dict[str, List[str]] = {
    "Mens Physique": ["mp_front", "mp_back"],
    "Classic": ["bb_front_double_biceps"],
    "Bodybuilding": ["bb_front_double_biceps"],
}
