from __future__ import annotations

from typing import Dict


# Tunable constants for scan-based proxy scoring. These are internal values and
# do not represent official judging thresholds.
THRESHOLDS: Dict[str, float] = {
    # Proportion ratios
    "v_taper_min": 1.50,
    "v_taper_target_low": 1.65,
    "v_taper_target_high": 1.95,
    "v_taper_max": 2.30,
    "x_frame_min": 1.20,
    "x_frame_target_low": 1.35,
    "x_frame_target_high": 1.65,
    "x_frame_max": 2.10,
    "waist_to_hips_target_max": 0.92,
    "waist_to_hips_soft_max": 0.98,
    "leg_to_torso_min": 0.85,
    "leg_to_torso_target_low": 0.95,
    "leg_to_torso_target_high": 1.10,
    "leg_to_torso_max": 1.25,

    # Symmetry (lower is better)
    "symmetry_good": 0.03,
    "symmetry_ok": 0.06,
    "symmetry_poor": 0.10,

    # Conditioning (0-100 proxy score)
    "condition_high": 75.0,
    "condition_mid": 55.0,

    # Presentation / posture tolerances (degrees where applicable)
    "posture_shoulder_level_deg": 6.0,
    "posture_hip_level_deg": 6.0,
    "posture_torso_rotation_deg": 8.0,
    "stance_width_ratio_min": 0.50,
    "stance_width_ratio_max": 1.10,

    # Muscularity index (derived from circumference/circumference proxies)
    "muscularity_index_min": 0.95,
    "muscularity_index_target_low": 1.10,
    "muscularity_index_target_high": 1.35,
    "muscularity_index_max": 1.60,
}


HARMONY_TARGETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "mens_physique": {
        "taper": {"min": 1.35, "target_low": 1.55, "target_high": 1.85, "max": 2.05},
        "upper_blockiness": {"min": 1.10, "target_low": 1.28, "target_high": 1.55, "max": 1.80},
        "lower_balance": {"min": 0.60, "target_low": 0.75, "target_high": 0.95, "max": 1.10},
        "leg_to_waist": {"min": 0.62, "target_low": 0.70, "target_high": 0.95, "max": 1.15},
    },
    "classic_physique": {
        "taper": {"min": 1.25, "target_low": 1.45, "target_high": 1.75, "max": 1.95},
        "upper_blockiness": {"min": 1.05, "target_low": 1.18, "target_high": 1.45, "max": 1.70},
        "lower_balance": {"min": 0.72, "target_low": 0.85, "target_high": 1.05, "max": 1.20},
        "leg_to_waist": {"min": 0.75, "target_low": 0.82, "target_high": 1.08, "max": 1.22},
    },
    "bodybuilding": {
        "taper": {"min": 1.20, "target_low": 1.35, "target_high": 1.65, "max": 1.85},
        "upper_blockiness": {"min": 1.00, "target_low": 1.12, "target_high": 1.40, "max": 1.65},
        "lower_balance": {"min": 0.82, "target_low": 0.95, "target_high": 1.15, "max": 1.30},
        "leg_to_waist": {"min": 0.88, "target_low": 0.95, "target_high": 1.20, "max": 1.35},
    },
}


def get_threshold(name: str, default: float | None = None) -> float | None:
    if name in THRESHOLDS:
        return THRESHOLDS[name]
    return default
