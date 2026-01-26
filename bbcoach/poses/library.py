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

POSE_GUIDES: Dict[str, Dict[str, tuple[float, float]]] = {
    "mp_front": {
        "nose": (0.50, 0.18),
        "left_shoulder": (0.42, 0.30),
        "right_shoulder": (0.58, 0.30),
        "left_elbow": (0.34, 0.42),
        "right_elbow": (0.66, 0.42),
        "left_wrist": (0.32, 0.55),
        "right_wrist": (0.68, 0.55),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.46, 0.72),
        "right_knee": (0.54, 0.72),
        "left_ankle": (0.47, 0.90),
        "right_ankle": (0.53, 0.90),
    },
    "mp_back": {
        "nose": (0.50, 0.18),
        "left_shoulder": (0.42, 0.30),
        "right_shoulder": (0.58, 0.30),
        "left_elbow": (0.34, 0.42),
        "right_elbow": (0.66, 0.42),
        "left_wrist": (0.32, 0.55),
        "right_wrist": (0.68, 0.55),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.46, 0.72),
        "right_knee": (0.54, 0.72),
        "left_ankle": (0.47, 0.90),
        "right_ankle": (0.53, 0.90),
    },
    "bb_front_double_biceps": {
        "nose": (0.50, 0.20),
        "left_shoulder": (0.42, 0.32),
        "right_shoulder": (0.58, 0.32),
        "left_elbow": (0.30, 0.24),
        "right_elbow": (0.70, 0.24),
        "left_wrist": (0.28, 0.32),
        "right_wrist": (0.72, 0.32),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.46, 0.74),
        "right_knee": (0.54, 0.74),
        "left_ankle": (0.47, 0.92),
        "right_ankle": (0.53, 0.92),
    },
}


# Simple routines for live coaching.
ROUTINES: Dict[str, List[str]] = {
    "Mens Physique": ["mp_front", "mp_back"],
    "Classic": ["bb_front_double_biceps"],
    "Bodybuilding": ["bb_front_double_biceps"],
}
