from __future__ import annotations

from typing import Dict, Optional


def compare_metrics(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, object]:
    regions_a = (a or {}).get("proxy_regions") or {}
    regions_b = (b or {}).get("proxy_regions") or {}

    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for name in sorted(set(regions_a.keys()) | set(regions_b.keys())):
        ra = regions_a.get(name, {})
        rb = regions_b.get(name, {})
        aw = ra.get("width_cm")
        bw = rb.get("width_cm")
        ar = ra.get("ratio")
        br = rb.get("ratio")
        deltas[name] = {
            "delta_cm": (float(bw) - float(aw)) if aw is not None and bw is not None else None,
            "delta_ratio": (float(br) - float(ar)) if ar is not None and br is not None else None,
        }

    return {
        "region_deltas": deltas,
    }
