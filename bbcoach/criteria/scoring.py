from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .kb import load_kb, get_category
from .thresholds import HARMONY_TARGETS, THRESHOLDS
from ..metrics.shape_metrics import compute_shape_metrics

_CONFIDENCE_ORDER = {"low": 0, "med": 1, "high": 2}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _score_ratio(
    value: Optional[float],
    target_low: float,
    target_high: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return None
    if min_val is None:
        min_val = target_low * 0.85
    if max_val is None:
        max_val = target_high * 1.15
    if value <= min_val or value >= max_val:
        return 0.0
    if target_low <= value <= target_high:
        return 100.0
    if value < target_low:
        return 100.0 * (value - min_val) / max(1e-6, target_low - min_val)
    return 100.0 * (max_val - value) / max(1e-6, max_val - target_high)


def _score_symmetry(asym: Optional[float]) -> Optional[float]:
    if asym is None:
        return None
    good = THRESHOLDS["symmetry_good"]
    ok = THRESHOLDS["symmetry_ok"]
    poor = THRESHOLDS["symmetry_poor"]
    if asym <= good:
        return 100.0
    if asym <= ok:
        return 80.0 - 20.0 * (asym - good) / max(1e-6, ok - good)
    if asym <= poor:
        return 60.0 - 40.0 * (asym - ok) / max(1e-6, poor - ok)
    return 20.0


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _extract_metrics_from_session(session_summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = session_summary.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    metrics_path = session_summary.get("metrics_path")
    if metrics_path:
        path = Path(metrics_path)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    if "session_dir" in session_summary:
        base = Path(session_summary["session_dir"])
        for path in (base / "metrics.json", base / "derived" / "metrics.json"):
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _extract_pose_features(session_summary: Dict[str, Any]) -> Dict[str, float]:
    feats = session_summary.get("pose_features")
    if isinstance(feats, dict):
        return {k: float(v) for k, v in feats.items() if v is not None}
    return {}


def _extract_shape_metrics(session_summary: Dict[str, Any]) -> Dict[str, Any]:
    shape = session_summary.get("shape_metrics")
    if isinstance(shape, dict):
        return shape
    metrics = _extract_metrics_from_session(session_summary)
    if not metrics:
        return {}
    pose_features = _extract_pose_features(session_summary)
    extras = session_summary.get("shape_extras")
    extras = extras if isinstance(extras, dict) else None
    return compute_shape_metrics(metrics, pose_features=pose_features or None, extras=extras)


def _confidence_from_gates(session_summary: Dict[str, Any]) -> Tuple[str, List[str]]:
    gates = session_summary.get("quality_gates")
    gates = gates if isinstance(gates, dict) else {}
    reasons_raw = gates.get("reasons")
    reasons = [str(v) for v in reasons_raw] if isinstance(reasons_raw, list) else []
    flags: List[str] = []
    issues = 0
    for key in [
        "lighting_stable",
        "distance_stable",
        "pose_locked_ok",
        "pointcloud_quality",
    ]:
        val = gates.get(key, None)
        if val is False:
            flags.append(f"{key}=false")
            issues += 1
        elif val is None:
            flags.append(f"{key}=unknown")
            issues += 1
    if "pose_lock_unknown_single_frame" in reasons:
        flags.append("pose_lock_unknown_single_frame")
        issues += 1
    if "quality_gates_fallback_used" in reasons:
        flags.append("quality_gates_fallback_used")
        issues += 1
    if issues >= 2:
        return "low", flags
    if issues == 1:
        return "med", flags
    return "high", flags


def _downgrade_confidence(conf: str, max_conf: str) -> str:
    curr = _CONFIDENCE_ORDER.get(conf, 1)
    cap = _CONFIDENCE_ORDER.get(max_conf, 1)
    if curr > cap:
        return max_conf
    return conf


def _score_symmetry_axis(shape: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    sym = shape.get("symmetry_left_right", {}) or {}
    scores = [
        _score_symmetry(sym.get("thigh_symmetry")),
        _score_symmetry(sym.get("arm_symmetry")),
        _score_symmetry(sym.get("calf_symmetry")),
    ]
    score = _mean_or_none(scores)
    notes = []
    if sym.get("thigh_symmetry") is not None and sym.get("thigh_symmetry") > THRESHOLDS["symmetry_ok"]:
        notes.append("thigh_symmetry")
    if sym.get("arm_symmetry") is not None and sym.get("arm_symmetry") > THRESHOLDS["symmetry_ok"]:
        notes.append("arm_symmetry")
    if sym.get("calf_symmetry") is not None and sym.get("calf_symmetry") > THRESHOLDS["symmetry_ok"]:
        notes.append("calf_symmetry")
    return score, notes


def _score_balance_axis(shape: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    v_taper = shape.get("v_taper_ratio")
    x_frame = shape.get("x_frame_ratio")
    waist_hips = shape.get("waist_to_hips_ratio")
    leg_torso = shape.get("leg_to_torso_balance")

    s_v = _score_ratio(
        v_taper,
        THRESHOLDS["v_taper_target_low"],
        THRESHOLDS["v_taper_target_high"],
        THRESHOLDS["v_taper_min"],
        THRESHOLDS["v_taper_max"],
    )
    s_x = _score_ratio(
        x_frame,
        THRESHOLDS["x_frame_target_low"],
        THRESHOLDS["x_frame_target_high"],
        THRESHOLDS["x_frame_min"],
        THRESHOLDS["x_frame_max"],
    )

    s_wh = None
    if waist_hips is not None:
        if waist_hips <= THRESHOLDS["waist_to_hips_target_max"]:
            s_wh = 100.0
        elif waist_hips <= THRESHOLDS["waist_to_hips_soft_max"]:
            s_wh = 80.0
        else:
            s_wh = 60.0 - 40.0 * (
                (waist_hips - THRESHOLDS["waist_to_hips_soft_max"])
                / max(1e-6, 1.10 - THRESHOLDS["waist_to_hips_soft_max"])
            )
            s_wh = _clamp(s_wh)

    s_lt = _score_ratio(
        leg_torso,
        THRESHOLDS["leg_to_torso_target_low"],
        THRESHOLDS["leg_to_torso_target_high"],
        THRESHOLDS["leg_to_torso_min"],
        THRESHOLDS["leg_to_torso_max"],
    )

    score = _mean_or_none([s_v, s_x, s_wh, s_lt])
    notes = []
    if v_taper is not None and v_taper < THRESHOLDS["v_taper_min"]:
        notes.append("v_taper_ratio")
    if x_frame is not None and x_frame < THRESHOLDS["x_frame_min"]:
        notes.append("x_frame_ratio")
    if leg_torso is not None and (
        leg_torso < THRESHOLDS["leg_to_torso_min"] or leg_torso > THRESHOLDS["leg_to_torso_max"]
    ):
        notes.append("leg_to_torso_balance")
    if waist_hips is not None and waist_hips > THRESHOLDS["waist_to_hips_soft_max"]:
        notes.append("waist_to_hips_ratio")
    return score, notes


def _score_muscularity_axis(shape: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    circs = shape.get("circumference_proxies", {}) or {}
    vols = shape.get("volume_proxies", {}) or {}
    upper = circs.get("upper_torso_circ")
    waist = circs.get("waist_circ")
    thigh = circs.get("thigh_circ")
    hip = circs.get("hip_circ")

    # Backward compatibility for historical metrics that don't include circumference_proxies.
    if upper is None:
        upper = vols.get("upper_torso_volume_proxy")
    if thigh is None:
        thigh = vols.get("thigh_volume_proxy")

    comps: List[Optional[float]] = []
    comps.append(_ratio_or_none(upper, waist))
    comps.append(_ratio_or_none(thigh, hip))

    index = _mean_or_none(comps)
    score = _score_ratio(
        index,
        THRESHOLDS["muscularity_index_target_low"],
        THRESHOLDS["muscularity_index_target_high"],
        THRESHOLDS["muscularity_index_min"],
        THRESHOLDS["muscularity_index_max"],
    )
    notes = []
    if index is not None and index < THRESHOLDS["muscularity_index_min"]:
        notes.append("overall_muscularity")
    return score, notes


def _harmony_template_id(category_id: str) -> str:
    cid = str(category_id or "").lower()
    if "mens_physique" in cid:
        return "mens_physique"
    if "classic" in cid:
        return "classic_physique"
    return "bodybuilding"


def _harmony_feedback(worst_component: Optional[str]) -> Optional[str]:
    if worst_component == "taper":
        return "Work on delts/lats or reduce waist."
    if worst_component == "lower_balance":
        return "Legs lag relative to pelvis."
    if worst_component == "flow":
        return "Scan quality/pose/props affecting silhouette; retake."
    if worst_component == "symmetry":
        return "Reduce left-right asymmetry across key bands."
    return None


def _component_target(template: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
    return template.get(key, {}) if isinstance(template, dict) else {}


def _score_harmony_axis(shape: Dict[str, Any], metrics: Dict[str, Any], category_id: str) -> Dict[str, Any]:
    template_id = _harmony_template_id(category_id)
    template = HARMONY_TARGETS.get(template_id, HARMONY_TARGETS["classic_physique"])
    harmony_feats = metrics.get("harmony_features", {}) if isinstance(metrics, dict) else {}
    harmony_feats = harmony_feats if isinstance(harmony_feats, dict) else {}
    profile_widths = harmony_feats.get("profile_widths_m", {}) if isinstance(harmony_feats.get("profile_widths_m"), dict) else {}

    shoulder = profile_widths.get("shoulder_width")
    chest = profile_widths.get("chest_width")
    waist = profile_widths.get("waist_width")
    hip = profile_widths.get("hip_width")
    thigh = profile_widths.get("thigh_width")

    if shoulder is None:
        shoulder = shape.get("shoulder_width")
    if chest is None:
        chest = shape.get("chest_width")
    if waist is None:
        waist = shape.get("waist_width")
    if hip is None:
        hip = shape.get("hip_width")

    if thigh is None:
        thigh = shape.get("thigh_width")
    if thigh is None:
        thigh = (shape.get("volume_proxies", {}) or {}).get("thigh_volume_proxy")

    taper_ratio = _ratio_or_none(shoulder, waist)
    upper_blockiness = _ratio_or_none(chest, waist)
    lower_balance_ratio = _ratio_or_none(thigh, hip)
    leg_to_waist_ratio = _ratio_or_none(thigh, waist)

    taper_target = _component_target(template, "taper")
    upper_target = _component_target(template, "upper_blockiness")
    lower_target = _component_target(template, "lower_balance")
    leg_target = _component_target(template, "leg_to_waist")

    taper_score = _mean_or_none(
        [
            _score_ratio(
                taper_ratio,
                taper_target.get("target_low", 1.45),
                taper_target.get("target_high", 1.75),
                taper_target.get("min"),
                taper_target.get("max"),
            ),
            _score_ratio(
                upper_blockiness,
                upper_target.get("target_low", 1.15),
                upper_target.get("target_high", 1.45),
                upper_target.get("min"),
                upper_target.get("max"),
            ),
        ]
    )

    lower_score = _mean_or_none(
        [
            _score_ratio(
                lower_balance_ratio,
                lower_target.get("target_low", 0.85),
                lower_target.get("target_high", 1.05),
                lower_target.get("min"),
                lower_target.get("max"),
            ),
            _score_ratio(
                leg_to_waist_ratio,
                leg_target.get("target_low", 0.82),
                leg_target.get("target_high", 1.08),
                leg_target.get("min"),
                leg_target.get("max"),
            ),
        ]
    )

    flow_score = harmony_feats.get("flow_score")
    try:
        flow_score = float(flow_score) if flow_score is not None else None
    except Exception:
        flow_score = None

    sym_score = harmony_feats.get("symmetry_proxy_score")
    try:
        sym_score = float(sym_score) if sym_score is not None else None
    except Exception:
        sym_score = None

    weights = {"taper": 0.30, "lower_balance": 0.25, "flow": 0.25, "symmetry": 0.20}
    components = {
        "taper": taper_score,
        "lower_balance": lower_score,
        "flow": flow_score,
        "symmetry": sym_score,
    }

    present = {k: v for k, v in components.items() if v is not None}
    total_w = sum(weights[k] for k in present.keys())
    score = None
    if present and total_w > 0:
        score = float(sum(float(v) * weights[k] for k, v in present.items()) / total_w)

    missing_components = [k for k, v in components.items() if v is None]
    flow_stats = harmony_feats.get("flow_stats", {}) if isinstance(harmony_feats.get("flow_stats"), dict) else {}
    sym_stats = harmony_feats.get("symmetry_proxy_stats", {}) if isinstance(harmony_feats.get("symmetry_proxy_stats"), dict) else {}

    sub_for_worst = {k: float(v) for k, v in present.items()}
    worst_component = min(sub_for_worst, key=sub_for_worst.get) if sub_for_worst else None

    notes: List[str] = []
    if taper_score is not None and taper_score < 55:
        notes.append("harmony_taper_low")
    if lower_score is not None and lower_score < 55:
        notes.append("harmony_lower_balance_low")
    if flow_score is not None and flow_score < 55:
        notes.append("harmony_flow_low")
    if sym_score is not None and sym_score < 55:
        notes.append("harmony_symmetry_low")
    if "symmetry" in missing_components:
        notes.append("harmony_symmetry_missing")

    return {
        "template_id": template_id,
        "score": None if score is None else round(_clamp(score), 2),
        "components": {
            "taper": {
                "score": None if taper_score is None else round(float(taper_score), 2),
                "shoulder_to_waist": None if taper_ratio is None else round(float(taper_ratio), 4),
                "chest_to_waist": None if upper_blockiness is None else round(float(upper_blockiness), 4),
                "target": [taper_target.get("target_low"), taper_target.get("target_high")],
            },
            "lower_balance": {
                "score": None if lower_score is None else round(float(lower_score), 2),
                "thigh_to_hip": None if lower_balance_ratio is None else round(float(lower_balance_ratio), 4),
                "thigh_to_waist": None if leg_to_waist_ratio is None else round(float(leg_to_waist_ratio), 4),
                "target": [lower_target.get("target_low"), lower_target.get("target_high")],
            },
            "flow": {
                "score": None if flow_score is None else round(float(flow_score), 2),
                "spikes": flow_stats.get("spikes"),
                "curvature_outliers": flow_stats.get("curvature_outliers"),
            },
            "symmetry": {
                "score": None if sym_score is None else round(float(sym_score), 2),
                "upper_deviation_pct": sym_stats.get("upper_deviation_pct"),
                "thigh_deviation_pct": sym_stats.get("thigh_deviation_pct"),
                "calf_deviation_pct": sym_stats.get("calf_deviation_pct"),
            },
        },
        "missing_components": missing_components,
        "worst_component": worst_component,
        "feedback": _harmony_feedback(worst_component),
        "notes": notes,
    }


def _ratio_or_none(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def _score_conditioning_axis(session_summary: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    cond = session_summary.get("condition", {}) or {}
    score = cond.get("condition_score")
    if score is None:
        return None, []
    try:
        score_f = float(score)
    except Exception:
        return None, []
    notes = []
    if score_f < THRESHOLDS["condition_mid"]:
        notes.append("condition_score_low")
    return _clamp(score_f), notes


def _score_presentation_axis(session_summary: Dict[str, Any], shape: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    base = 100.0
    flags = []

    posture = shape.get("posture_flags", {}) or {}
    shoulder = posture.get("shoulder_level_delta")
    hip = posture.get("hip_level_delta")
    torso = posture.get("torso_rotation_deg")
    stance = posture.get("stance_width_ratio")

    if shoulder is not None and shoulder > THRESHOLDS["posture_shoulder_level_deg"]:
        base -= 12.0
        flags.append("shoulder_level_delta")
    if hip is not None and hip > THRESHOLDS["posture_hip_level_deg"]:
        base -= 12.0
        flags.append("hip_level_delta")
    if torso is not None and torso > THRESHOLDS["posture_torso_rotation_deg"]:
        base -= 12.0
        flags.append("torso_rotation_deg")
    if stance is not None and (
        stance < THRESHOLDS["stance_width_ratio_min"] or stance > THRESHOLDS["stance_width_ratio_max"]
    ):
        base -= 10.0
        flags.append("stance_width_ratio")

    pose_score = session_summary.get("pose_compliance")
    if pose_score is not None:
        try:
            pose_val = float(pose_score)
            base = base * _clamp(pose_val, 0.0, 1.0)
        except Exception:
            pass

    return _clamp(base), flags


def _eval_rule(expr: str, proxy_values: Dict[str, Any]) -> bool:
    ctx = {**proxy_values, "thresholds": THRESHOLDS}
    try:
        return bool(eval(expr, {"__builtins__": {}}, ctx))
    except Exception:
        return False


def score_category(session_summary: Dict[str, Any], federation_id: str, category_id: str) -> Dict[str, Any]:
    kb = load_kb()
    category = get_category(kb, federation_id, category_id)
    metrics = _extract_metrics_from_session(session_summary)
    shape = _extract_shape_metrics(session_summary)
    harmony = _score_harmony_axis(shape, metrics, category_id)

    axis_scores: Dict[str, Optional[float]] = {}
    flags: List[str] = []
    strengths: List[str] = []
    priorities: List[str] = []
    proxy_values: Dict[str, Any] = {}

    proxy_values.update(shape)
    proxy_values["condition_score"] = (session_summary.get("condition") or {}).get("condition_score")
    proxy_values["harmony_score"] = harmony.get("score")

    conf, conf_flags = _confidence_from_gates(session_summary)
    flags.extend(conf_flags)

    score_map: Dict[str, Tuple[Optional[float], List[str]]] = {}
    all_axes = list(dict.fromkeys(list(category["judging_axes"]) + ["harmony"]))
    for axis in all_axes:
        if axis == "symmetry":
            score_map[axis] = _score_symmetry_axis(shape)
        elif axis == "balance_proportions":
            score_map[axis] = _score_balance_axis(shape)
        elif axis == "muscularity":
            score_map[axis] = _score_muscularity_axis(shape)
        elif axis == "conditioning":
            score_map[axis] = _score_conditioning_axis(session_summary)
        elif axis == "presentation":
            score_map[axis] = _score_presentation_axis(session_summary, shape)
        elif axis == "harmony":
            score_map[axis] = (harmony.get("score"), harmony.get("notes", []))
        else:
            score_map[axis] = (None, [])

    for axis, (score, notes) in score_map.items():
        axis_scores[axis] = _clamp(score) if score is not None else None
        for note in notes:
            priorities.append(note)

    weights = category["scoring_model"]["axis_weights"]
    missing_axes = [axis for axis, score in axis_scores.items() if score is None]
    if "symmetry" in missing_axes:
        flags.append("symmetry_missing")
        conf = _downgrade_confidence(conf, "med")
    if "muscularity" in missing_axes:
        flags.append("muscularity_missing")
        conf = _downgrade_confidence(conf, "med")
    if "harmony_symmetry_missing" in (harmony.get("notes") or []):
        flags.append("harmony_symmetry_missing")
        conf = _downgrade_confidence(conf, "med")

    overall = 0.0
    total_w = 0.0
    for axis, score in axis_scores.items():
        if score is None:
            continue
        w = float(weights.get(axis, 0.0))
        overall += score * w
        total_w += w
    overall = overall / total_w if total_w > 0 else 0.0

    # Strengths/priorities from axis scores
    sorted_axes = sorted(
        [(axis, score) for axis, score in axis_scores.items() if score is not None and axis in category["judging_axes"]],
        key=lambda it: it[1],
        reverse=True,
    )
    strengths = [name for name, _ in sorted_axes[:2]]
    priorities.extend([name for name, _ in sorted_axes[-2:]])

    # Evaluate feedback rules
    triggered_rules = []
    rule_citations: List[str] = []
    for rule in category.get("feedback_rules", []):
        if _eval_rule(rule.get("when", ""), proxy_values):
            triggered_rules.append(rule)
            rule_citations.extend([str(c) for c in rule.get("citations", [])])

    # If confidence low, instruct rescan
    if conf == "low":
        flags.append(
            "Do a locked rescan: stable lighting, fixed distance, pose locked, full pointcloud coverage."
        )

    citations_used = []
    for cid in category.get("citations", []):
        citations_used.append(kb["citations"][cid])
    for cid in rule_citations:
        citations_used.append(kb["citations"][cid])

    # de-dup citations
    seen = set()
    citations_unique = []
    for c in citations_used:
        key = (c["title"], c["url"])
        if key in seen:
            continue
        seen.add(key)
        citations_unique.append(c)

    return {
        "kb_version": kb["kb_version"],
        "federation_id": federation_id,
        "category_id": category_id,
        "axis_scores": axis_scores,
        "missing_axes": missing_axes,
        "harmony": harmony,
        "proxy_values": proxy_values,
        "overall_score": _clamp(overall),
        "confidence": conf,
        "flags": flags,
        "strengths": strengths,
        "priorities": priorities,
        "feedback": [
            {
                "id": rule.get("id"),
                "message": rule.get("message"),
                "action": rule.get("action"),
                "training_prescription": rule.get("training_prescription"),
            }
            for rule in triggered_rules
        ],
        "citations_used": citations_unique,
    }
