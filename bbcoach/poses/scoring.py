from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Iterable

from ..metrics.pose_features import compute_features, PoseLandmarks


@dataclass(frozen=True)
class PoseScore:
    score_0_100: float
    per_feature: Dict[str, float]
    ok_flags: Dict[str, bool]
    advice: List[str]


def _extra_metrics(props: Optional[object]) -> Dict[str, float]:
    if props is None:
        return {}
    if isinstance(props, dict):
        items: Iterable[Tuple[str, Optional[float]]] = props.items()
    else:
        items = (
            ("shoulder_to_waist", getattr(props, "shoulder_to_waist", None)),
            ("chest_to_waist", getattr(props, "chest_to_waist", None)),
            ("hip_to_waist", getattr(props, "hip_to_waist", None)),
            ("upper_to_lower_area", getattr(props, "upper_to_lower_area", None)),
        )
    feats: Dict[str, float] = {}
    for k, v in items:
        if v is None:
            continue
        try:
            feats[k] = float(v)
        except Exception:
            continue
    return feats


def score_pose(
    landmarks: Dict[str, Tuple[float, float]],
    target: Dict[str, float],
    tolerance: Dict[str, float],
    template_override: Dict[str, float] | None = None,
    weights: Dict[str, float] | None = None,
    props: Optional[object] = None,
) -> PoseScore:
    feats = compute_features(PoseLandmarks(pts=landmarks))
    if not feats:
        return PoseScore(score_0_100=0.0, per_feature={}, ok_flags={}, advice=[])

    extra = _extra_metrics(props)
    if extra:
        feats.update(extra)

    # If user captured a personal template, use that as target baseline.
    if template_override:
        tgt = dict(target)
        tgt.update(template_override)
    else:
        tgt = target

    per: Dict[str, float] = {}
    ok: Dict[str, bool] = {}
    total = 0.0
    total_weight = 0.0
    n = 0
    advice_items: List[Tuple[float, str]] = []
    weights = weights or {}

    hints = {
        "shoulder_level": "Keep shoulders level (no shrugging).",
        "hip_level": "Keep hips level.",
        "hip_to_shoulder_parallel": "Align hips with shoulders.",
        "torso_upright": "Stand tall; avoid leaning.",
        "elbow_sym": "Match elbow spacing.",
        "elbow_height": "Match elbow height.",
        "shoulder_width_x": "Turn slightly more or less.",
        "hip_width_x": "Square your hips to the pose.",
        "stance_width": "Adjust foot width.",
        "knee_sym": "Level your knees/legs.",
        "chest_to_waist": "Lift chest and keep waist tight.",
        "shoulder_to_waist": "Open lats; keep waist tight.",
        "hip_to_waist": "Keep midsection tight and hips set.",
        "upper_to_lower_area": "Balance upper and lower body posture.",
    }

    for k, tv in tgt.items():
        if k not in feats:
            continue
        tol = float(tolerance.get(k, 10.0))
        if tol <= 0.0:
            tol = 1e-6
        err = abs(float(feats[k]) - float(tv))
        # Score: 1 at 0 error, down to 0 at >= tol
        s = max(0.0, 1.0 - (err / tol))
        per[k] = s
        ok_flag = (err <= tol)
        ok[k] = ok_flag
        w = float(weights.get(k, 1.0))
        total += s * w
        total_weight += w
        n += 1
        if not ok_flag:
            hint = hints.get(k, f"Adjust {k.replace('_', ' ')}.")
            severity = err / tol if tol > 0 else err
            advice_items.append((severity, hint))

    score = (total / total_weight) * 100.0 if total_weight else 0.0
    if n < 4:
        score = min(score, 70.0)

    advice = [hint for _sev, hint in sorted(advice_items, key=lambda it: it[0], reverse=True)]
    return PoseScore(score_0_100=score, per_feature=per, ok_flags=ok, advice=advice)
