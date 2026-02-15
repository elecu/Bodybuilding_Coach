from __future__ import annotations

from typing import Dict, List, Tuple, Optional


def score_metrics(metrics: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    regions = (metrics or {}).get("proxy_regions") or {}
    proportions = (metrics or {}).get("proportions") or {}

    region_targets = (config or {}).get("region_targets") or {}
    region_weights = (config or {}).get("region_weights") or {}
    ratio_targets = (config or {}).get("ratio_targets") or {}

    deficits: List[Tuple[float, str]] = []

    for region, target in region_targets.items():
        if region not in regions:
            continue
        ratio = regions[region].get("ratio")
        if ratio is None:
            continue
        try:
            tgt = float(target)
        except Exception:
            continue
        shortfall = max(0.0, tgt - float(ratio))
        if shortfall <= 0:
            continue
        weight = float(region_weights.get(region, 1.0))
        deficits.append((shortfall * weight, region))

    ratio_to_region = {
        "shoulder_to_waist": "waist",
        "chest_to_waist": "waist",
        "hip_to_waist": "waist",
    }
    for ratio_key, target in ratio_targets.items():
        val = proportions.get(ratio_key)
        if val is None:
            continue
        try:
            tgt = float(target)
        except Exception:
            continue
        shortfall = max(0.0, tgt - float(val))
        if shortfall <= 0:
            continue
        region = ratio_to_region.get(ratio_key, ratio_key)
        weight = float(region_weights.get(region, 1.0))
        deficits.append((shortfall * weight, region))

    deficits.sort(key=lambda it: it[0], reverse=True)
    missing = [name for _score, name in deficits if name not in ("", None)]
    # unique preserve order
    seen = set()
    missing_unique = []
    for name in missing:
        if name in seen:
            continue
        seen.add(name)
        missing_unique.append(name)
        if len(missing_unique) >= 3:
            break

    if missing_unique:
        summary = "Focus areas: " + ", ".join(missing_unique)
    else:
        summary = "Balanced across proxy regions."

    return {
        "summary": summary,
        "missing_areas": missing_unique,
    }
