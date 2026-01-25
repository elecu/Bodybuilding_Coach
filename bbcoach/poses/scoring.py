from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..metrics.pose_features import compute_features, PoseLandmarks


@dataclass(frozen=True)
class PoseScore:
    score_0_100: float
    per_feature: Dict[str, float]
    ok_flags: Dict[str, bool]


def score_pose(
    landmarks: Dict[str, Tuple[float, float]],
    target: Dict[str, float],
    tolerance: Dict[str, float],
    template_override: Dict[str, float] | None = None,
) -> PoseScore:
    feats = compute_features(PoseLandmarks(pts=landmarks))
    if not feats:
        return PoseScore(score_0_100=0.0, per_feature={}, ok_flags={})

    # If user captured a personal template, use that as target baseline.
    if template_override:
        tgt = dict(target)
        tgt.update(template_override)
    else:
        tgt = target

    per: Dict[str, float] = {}
    ok: Dict[str, bool] = {}
    total = 0.0
    n = 0

    for k, tv in tgt.items():
        if k not in feats:
            continue
        tol = float(tolerance.get(k, 10.0))
        err = abs(float(feats[k]) - float(tv))
        # Score: 1 at 0 error, down to 0 at >= tol
        s = max(0.0, 1.0 - (err / tol))
        per[k] = s
        ok[k] = (err <= tol)
        total += s
        n += 1

    score = (total / n) * 100.0 if n else 0.0
    return PoseScore(score_0_100=score, per_feature=per, ok_flags=ok)
