from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PoseDef:
    key: str
    display: str
    # Target angles (degrees) with tolerance for scoring. Keys correspond to computed features.
    target: dict
    tolerance: dict
    guidance: List[str]
    weights: Optional[Dict[str, float]] = None


# NOTE: These are *generic* templates to get you started.
# The user can capture personalised templates (Space key) to improve scoring.

POSES: Dict[str, PoseDef] = {
    "mp_front": PoseDef(
        key="mp_front",
        display="Men's Physique - Front",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "elbow_sym": 0.0,
            "hip_level": 0.0,
            "stance_width": 18.0,
            "shoulder_to_waist": 1.30,
            "chest_to_waist": 1.20,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
            "elbow_sym": 10.0,
            "hip_level": 6.0,
            "stance_width": 10.0,
            "shoulder_to_waist": 0.30,
            "chest_to_waist": 0.25,
        },
        guidance=[
            "Stand tall. Keep ribcage up and pelvis neutral.",
            "Open lats slightly to show V-taper.",
            "Keep shoulders level and relaxed (no shrugging).",
        ],
        weights={
            "shoulder_to_waist": 0.6,
            "chest_to_waist": 0.6,
        },
    ),
    "mp_back": PoseDef(
        key="mp_back",
        display="Men's Physique - Back",
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
    "mp_left": PoseDef(
        key="mp_left",
        display="Men's Physique - Quarter Turn (Left)",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "hip_level": 0.0,
            "hip_to_shoulder_parallel": 0.0,
            "shoulder_width_x": 16.0,
            "stance_width": 14.0,
            "shoulder_to_waist": 1.30,
            "chest_to_waist": 1.20,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 8.0,
            "hip_level": 8.0,
            "hip_to_shoulder_parallel": 10.0,
            "shoulder_width_x": 8.0,
            "stance_width": 8.0,
            "shoulder_to_waist": 0.30,
            "chest_to_waist": 0.25,
        },
        guidance=[
            "Quarter turn to the left; keep chest open.",
            "Keep shoulders level and hips aligned.",
        ],
        weights={
            "shoulder_width_x": 1.2,
            "hip_to_shoulder_parallel": 1.1,
            "shoulder_to_waist": 0.6,
            "chest_to_waist": 0.6,
        },
    ),
    "mp_right": PoseDef(
        key="mp_right",
        display="Men's Physique - Quarter Turn (Right)",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "hip_level": 0.0,
            "hip_to_shoulder_parallel": 0.0,
            "shoulder_width_x": 16.0,
            "stance_width": 14.0,
            "shoulder_to_waist": 1.30,
            "chest_to_waist": 1.20,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 8.0,
            "hip_level": 8.0,
            "hip_to_shoulder_parallel": 10.0,
            "shoulder_width_x": 8.0,
            "stance_width": 8.0,
            "shoulder_to_waist": 0.30,
            "chest_to_waist": 0.25,
        },
        guidance=[
            "Quarter turn to the right; keep chest open.",
            "Keep shoulders level and hips aligned.",
        ],
        weights={
            "shoulder_width_x": 1.2,
            "hip_to_shoulder_parallel": 1.1,
            "shoulder_to_waist": 0.6,
            "chest_to_waist": 0.6,
        },
    ),
    "bb_front_double_biceps": PoseDef(
        key="bb_front_double_biceps",
        display="Bodybuilding - Front Double Biceps",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "elbow_height": 0.0,
            "hip_level": 0.0,
            "stance_width": 18.0,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
            "elbow_height": 12.0,
            "hip_level": 6.0,
            "stance_width": 10.0,
        },
        guidance=[
            "Raise elbows until biceps peak without shrugging.",
            "Keep waist tight and chest up.",
        ],
    ),
    "bb_front_lat_spread": PoseDef(
        key="bb_front_lat_spread",
        display="Bodybuilding - Front Lat Spread",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "hip_level": 0.0,
            "shoulder_width_x": 28.0,
            "hip_to_shoulder_parallel": 0.0,
            "stance_width": 18.0,
            "chest_to_waist": 1.15,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
            "hip_level": 6.0,
            "shoulder_width_x": 12.0,
            "hip_to_shoulder_parallel": 10.0,
            "stance_width": 10.0,
            "chest_to_waist": 0.25,
        },
        guidance=[
            "Lift chest and drive elbows out to flare the lats.",
            "Keep hips square and waist tight.",
        ],
        weights={
            "chest_to_waist": 0.6,
        },
    ),
    "bb_side_chest": PoseDef(
        key="bb_side_chest",
        display="Bodybuilding - Side Chest",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "hip_level": 0.0,
            "hip_to_shoulder_parallel": 0.0,
            "shoulder_width_x": 14.0,
            "hip_width_x": 12.0,
            "stance_width": 14.0,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 8.0,
            "hip_level": 8.0,
            "hip_to_shoulder_parallel": 10.0,
            "shoulder_width_x": 8.0,
            "hip_width_x": 8.0,
            "stance_width": 8.0,
        },
        guidance=[
            "Rotate fully to the side and lift chest high.",
            "Keep shoulders and hips stacked.",
        ],
    ),
    "bb_back_double_biceps": PoseDef(
        key="bb_back_double_biceps",
        display="Bodybuilding - Back Double Biceps",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "elbow_height": 0.0,
            "elbow_sym": 0.0,
            "hip_level": 0.0,
            "shoulder_width_x": 26.0,
            "stance_width": 18.0,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
            "elbow_height": 12.0,
            "elbow_sym": 12.0,
            "hip_level": 6.0,
            "shoulder_width_x": 12.0,
            "stance_width": 10.0,
        },
        guidance=[
            "Raise elbows and spread the back without shrugging.",
            "Keep waist tight and hips level.",
        ],
    ),
    "bb_back_lat_spread": PoseDef(
        key="bb_back_lat_spread",
        display="Bodybuilding - Back Lat Spread",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "hip_level": 0.0,
            "hip_to_shoulder_parallel": 0.0,
            "shoulder_width_x": 28.0,
            "hip_width_x": 16.0,
            "stance_width": 18.0,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
            "hip_level": 6.0,
            "hip_to_shoulder_parallel": 10.0,
            "shoulder_width_x": 12.0,
            "hip_width_x": 10.0,
            "stance_width": 10.0,
        },
        guidance=[
            "Drive elbows out and widen the back.",
            "Keep hips square and level.",
        ],
    ),
    "bb_side_triceps": PoseDef(
        key="bb_side_triceps",
        display="Bodybuilding - Side Triceps",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "hip_level": 0.0,
            "hip_to_shoulder_parallel": 0.0,
            "shoulder_width_x": 14.0,
            "hip_width_x": 12.0,
            "stance_width": 14.0,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 8.0,
            "hip_level": 8.0,
            "hip_to_shoulder_parallel": 10.0,
            "shoulder_width_x": 8.0,
            "hip_width_x": 8.0,
            "stance_width": 8.0,
        },
        guidance=[
            "Rotate fully and lock the arm for triceps.",
            "Keep hips stacked under shoulders.",
        ],
    ),
    "bb_abs_and_thighs": PoseDef(
        key="bb_abs_and_thighs",
        display="Bodybuilding - Abs and Thighs",
        target={
            "torso_upright": 0.0,
            "shoulder_level": 0.0,
            "hip_level": 0.0,
            "hip_to_shoulder_parallel": 0.0,
            "stance_width": 16.0,
            "chest_to_waist": 1.15,
        },
        tolerance={
            "torso_upright": 10.0,
            "shoulder_level": 6.0,
            "hip_level": 6.0,
            "hip_to_shoulder_parallel": 10.0,
            "stance_width": 10.0,
            "chest_to_waist": 0.25,
        },
        guidance=[
            "Lift ribcage and crunch abs without hunching.",
            "Show the thigh and keep hips level.",
        ],
        weights={
            "chest_to_waist": 0.6,
        },
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
    "mp_left": {
        "nose": (0.48, 0.20),
        "left_shoulder": (0.44, 0.32),
        "right_shoulder": (0.56, 0.32),
        "left_elbow": (0.40, 0.45),
        "right_elbow": (0.62, 0.42),
        "left_wrist": (0.38, 0.58),
        "right_wrist": (0.64, 0.54),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.47, 0.74),
        "right_knee": (0.53, 0.74),
        "left_ankle": (0.48, 0.92),
        "right_ankle": (0.52, 0.92),
    },
    "mp_right": {
        "nose": (0.52, 0.20),
        "left_shoulder": (0.44, 0.32),
        "right_shoulder": (0.56, 0.32),
        "left_elbow": (0.38, 0.42),
        "right_elbow": (0.60, 0.45),
        "left_wrist": (0.36, 0.54),
        "right_wrist": (0.62, 0.58),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.47, 0.74),
        "right_knee": (0.53, 0.74),
        "left_ankle": (0.48, 0.92),
        "right_ankle": (0.52, 0.92),
    },
    "bb_front_double_biceps": {
        "nose": (0.50, 0.20),
        "left_shoulder": (0.42, 0.32),
        "right_shoulder": (0.58, 0.32),
        "left_elbow": (0.30, 0.26),
        "right_elbow": (0.70, 0.26),
        "left_wrist": (0.38, 0.18),
        "right_wrist": (0.62, 0.18),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.46, 0.74),
        "right_knee": (0.54, 0.74),
        "left_ankle": (0.47, 0.92),
        "right_ankle": (0.53, 0.92),
    },
    "bb_front_lat_spread": {
        "nose": (0.50, 0.20),
        "left_shoulder": (0.38, 0.30),
        "right_shoulder": (0.62, 0.30),
        "left_elbow": (0.30, 0.40),
        "right_elbow": (0.70, 0.40),
        "left_wrist": (0.32, 0.52),
        "right_wrist": (0.68, 0.52),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.46, 0.74),
        "right_knee": (0.54, 0.74),
        "left_ankle": (0.47, 0.92),
        "right_ankle": (0.53, 0.92),
    },
    "bb_side_chest": {
        "nose": (0.46, 0.20),
        "left_shoulder": (0.47, 0.32),
        "right_shoulder": (0.53, 0.32),
        "left_elbow": (0.44, 0.45),
        "right_elbow": (0.60, 0.40),
        "left_wrist": (0.42, 0.58),
        "right_wrist": (0.62, 0.56),
        "left_hip": (0.48, 0.56),
        "right_hip": (0.52, 0.56),
        "left_knee": (0.47, 0.74),
        "right_knee": (0.53, 0.74),
        "left_ankle": (0.47, 0.92),
        "right_ankle": (0.53, 0.92),
    },
    "bb_back_double_biceps": {
        "nose": (0.50, 0.18),
        "left_shoulder": (0.40, 0.30),
        "right_shoulder": (0.60, 0.30),
        "left_elbow": (0.28, 0.24),
        "right_elbow": (0.72, 0.24),
        "left_wrist": (0.26, 0.32),
        "right_wrist": (0.74, 0.32),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.46, 0.74),
        "right_knee": (0.54, 0.74),
        "left_ankle": (0.47, 0.92),
        "right_ankle": (0.53, 0.92),
    },
    "bb_back_lat_spread": {
        "nose": (0.50, 0.18),
        "left_shoulder": (0.38, 0.32),
        "right_shoulder": (0.62, 0.32),
        "left_elbow": (0.30, 0.42),
        "right_elbow": (0.70, 0.42),
        "left_wrist": (0.32, 0.52),
        "right_wrist": (0.68, 0.52),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.46, 0.74),
        "right_knee": (0.54, 0.74),
        "left_ankle": (0.47, 0.92),
        "right_ankle": (0.53, 0.92),
    },
    "bb_side_triceps": {
        "nose": (0.46, 0.20),
        "left_shoulder": (0.47, 0.32),
        "right_shoulder": (0.53, 0.32),
        "left_elbow": (0.46, 0.46),
        "right_elbow": (0.56, 0.48),
        "left_wrist": (0.45, 0.62),
        "right_wrist": (0.58, 0.62),
        "left_hip": (0.48, 0.56),
        "right_hip": (0.52, 0.56),
        "left_knee": (0.47, 0.74),
        "right_knee": (0.53, 0.74),
        "left_ankle": (0.47, 0.92),
        "right_ankle": (0.53, 0.92),
    },
    "bb_abs_and_thighs": {
        "nose": (0.50, 0.20),
        "left_shoulder": (0.42, 0.30),
        "right_shoulder": (0.58, 0.30),
        "left_elbow": (0.30, 0.22),
        "right_elbow": (0.70, 0.22),
        "left_wrist": (0.32, 0.28),
        "right_wrist": (0.68, 0.28),
        "left_hip": (0.46, 0.56),
        "right_hip": (0.54, 0.56),
        "left_knee": (0.45, 0.74),
        "right_knee": (0.55, 0.74),
        "left_ankle": (0.46, 0.92),
        "right_ankle": (0.54, 0.92),
    },
}


# Federation-specific routines for Men's Physique quarter turns.
MP_WNBF_UK = ["mp_front", "mp_left", "mp_back", "mp_right"]
MP_UKBFF = ["mp_front", "mp_left", "mp_back", "mp_right", "mp_front"]


# Simple routines for live coaching.
ROUTINES: Dict[str, List[str]] = {
    "Mens Physique": MP_WNBF_UK,
    "Classic": [
        "bb_front_double_biceps",
        "bb_front_lat_spread",
        "bb_side_chest",
        "bb_back_double_biceps",
        "bb_back_lat_spread",
        "bb_side_triceps",
        "bb_abs_and_thighs",
    ],
    "Bodybuilding": [
        "bb_front_double_biceps",
        "bb_front_lat_spread",
        "bb_side_chest",
        "bb_back_double_biceps",
        "bb_back_lat_spread",
        "bb_side_triceps",
        "bb_abs_and_thighs",
    ],
}


def routine_for(category: str, federation: str) -> List[str]:
    if category == "Mens Physique":
        if federation == "WNBF_UK":
            return list(MP_WNBF_UK)
        if federation in ("UKBFF", "PCA"):
            return list(MP_UKBFF)
    return list(ROUTINES.get(category, ROUTINES["Mens Physique"]))
