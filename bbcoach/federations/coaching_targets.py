from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class CoachingTargets:
    target_bodyfat_range: dict[str, dict[str, float]]
    safe_weekly_weight_change_range: dict[str, float]


_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "coaching_targets.yaml"


def load_coaching_targets(path: Optional[Path] = None) -> CoachingTargets:
    cfg_path = Path(path or _DEFAULT_PATH)
    if not cfg_path.exists():
        return CoachingTargets(target_bodyfat_range={}, safe_weekly_weight_change_range={})
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return CoachingTargets(target_bodyfat_range={}, safe_weekly_weight_change_range={})
    if not isinstance(data, dict):
        return CoachingTargets(target_bodyfat_range={}, safe_weekly_weight_change_range={})

    out_ranges: dict[str, dict[str, float]] = {}
    for key, val in (data.get("target_bodyfat_range") or {}).items():
        if not isinstance(val, dict):
            continue
        min_val = _as_float(val.get("min_percent"))
        max_val = _as_float(val.get("max_percent"))
        if min_val is None and max_val is None:
            continue
        out_ranges[str(key)] = {}
        if min_val is not None:
            out_ranges[str(key)]["min_percent"] = min_val
        if max_val is not None:
            out_ranges[str(key)]["max_percent"] = max_val

    safe_change: dict[str, float] = {}
    safe_raw = data.get("safe_weekly_weight_change_range") or {}
    if isinstance(safe_raw, dict):
        min_change = _as_float(safe_raw.get("min_percent_bodyweight"))
        max_change = _as_float(safe_raw.get("max_percent_bodyweight"))
        if min_change is not None:
            safe_change["min_percent_bodyweight"] = min_change
        if max_change is not None:
            safe_change["max_percent_bodyweight"] = max_change

    return CoachingTargets(target_bodyfat_range=out_ranges, safe_weekly_weight_change_range=safe_change)


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None
