from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _pick_view(metrics: Dict[str, object], angle_deg: int = 0) -> Optional[Dict[str, object]]:
    views = metrics.get("views")
    if not isinstance(views, list) or not views:
        return None
    for view in views:
        if not isinstance(view, dict):
            continue
        if int(view.get("angle_deg", -1)) == angle_deg:
            return view
    # fallback to the first view
    return views[0] if isinstance(views[0], dict) else None


def _mean_band(values: List[float], heights: List[float], low: float, high: float) -> Optional[float]:
    if not values or not heights or len(values) != len(heights):
        return None
    band = [v for v, h in zip(values, heights) if low <= h <= high and v > 0]
    if not band:
        return None
    return float(sum(band) / len(band))


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def compute_shape_metrics(
    metrics: Dict[str, object],
    pose_features: Optional[Dict[str, float]] = None,
    extras: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    extras = extras or {}
    view_front = _pick_view(metrics, angle_deg=0) or {}
    key_widths = view_front.get("key_widths_m") if isinstance(view_front, dict) else None
    key_widths = key_widths if isinstance(key_widths, dict) else {}

    shoulder_width = key_widths.get("shoulders")
    chest_width = key_widths.get("chest")
    waist_width = key_widths.get("waist")
    hip_width = key_widths.get("hips")
    thigh_width = key_widths.get("thigh")
    calf_width = key_widths.get("calf")

    thigh_width_l = extras.get("thigh_width_L")
    thigh_width_r = extras.get("thigh_width_R")
    calf_width_l = extras.get("calf_width_L")
    calf_width_r = extras.get("calf_width_R")
    arm_girth_l = extras.get("arm_girth_proxy_L")
    arm_girth_r = extras.get("arm_girth_proxy_R")

    v_taper_ratio = _ratio(shoulder_width, waist_width)
    x_frame_ratio = _ratio(shoulder_width, hip_width)
    waist_to_hips_ratio = _ratio(waist_width, hip_width)

    def symmetry(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        mean = (a + b) / 2.0
        if mean == 0:
            return None
        return abs(a - b) / mean

    symmetry_left_right = {
        "thigh_symmetry": symmetry(thigh_width_l, thigh_width_r),
        "arm_symmetry": symmetry(arm_girth_l, arm_girth_r),
        "calf_symmetry": symmetry(calf_width_l, calf_width_r),
    }

    circum = metrics.get("circumference_profile_m")
    heights = metrics.get("height_bins")
    circum = circum if isinstance(circum, list) else []
    heights = heights if isinstance(heights, list) else []

    upper_circ = _mean_band(circum, heights, 0.60, 0.75)
    waist_circ = _mean_band(circum, heights, 0.50, 0.62)
    hip_circ = _mean_band(circum, heights, 0.45, 0.55)
    thigh_circ = _mean_band(circum, heights, 0.30, 0.45)
    leg_to_torso_balance = _ratio(thigh_circ, upper_circ)

    posture_flags: Dict[str, Optional[float]] = {
        "shoulder_level_delta": None,
        "hip_level_delta": None,
        "torso_rotation_deg": None,
        "stance_width_ratio": None,
    }
    if pose_features:
        shoulder_level = pose_features.get("shoulder_level")
        hip_level = pose_features.get("hip_level")
        torso_rotation = pose_features.get("hip_to_shoulder_parallel")
        stance_width = pose_features.get("stance_width")
        shoulder_width_x = pose_features.get("shoulder_width_x")

        posture_flags = {
            "shoulder_level_delta": abs(float(shoulder_level)) if shoulder_level is not None else None,
            "hip_level_delta": abs(float(hip_level)) if hip_level is not None else None,
            "torso_rotation_deg": abs(float(torso_rotation)) if torso_rotation is not None else None,
            "stance_width_ratio": _ratio(stance_width, shoulder_width_x),
        }

    return {
        "shoulder_width": shoulder_width,
        "chest_width": chest_width,
        "waist_width": waist_width,
        "hip_width": hip_width,
        "thigh_width_L": thigh_width_l,
        "thigh_width_R": thigh_width_r,
        "calf_width_L": calf_width_l,
        "calf_width_R": calf_width_r,
        "arm_girth_proxy_L": arm_girth_l,
        "arm_girth_proxy_R": arm_girth_r,
        "v_taper_ratio": v_taper_ratio,
        "x_frame_ratio": x_frame_ratio,
        "waist_to_hips_ratio": waist_to_hips_ratio,
        "symmetry_left_right": symmetry_left_right,
        "leg_to_torso_balance": leg_to_torso_balance,
        "volume_proxies": {
            "upper_torso_volume_proxy": upper_circ,
            "thigh_volume_proxy": thigh_circ,
        },
        "circumference_proxies": {
            "upper_torso_circ": upper_circ,
            "waist_circ": waist_circ,
            "hip_circ": hip_circ,
            "thigh_circ": thigh_circ,
        },
        "posture_flags": posture_flags,
    }
