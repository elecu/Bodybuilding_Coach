from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from bbcoach.criteria.kb import load_kb
from bbcoach.criteria.scoring import score_category
from bbcoach.core.quality_gates import compute_quality_gates
from bbcoach.metrics.condition import compute_condition_for_session
from bbcoach.metrics.shape_metrics import compute_shape_metrics


_RATIO_MEANINGS = {
    "shoulder_to_waist": "Higher usually means stronger V-taper (wide shoulders vs tight waist).",
    "chest_to_waist": "Upper torso fullness relative to waist.",
    "hip_to_waist": "Lower-body frame relative to waist; useful for X-frame balance.",
    "thigh_to_waist": "Leg sweep relative to waist.",
}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _resolve_session_info(session_dir: Path, meta: Dict[str, Any]) -> Tuple[str, str, str, str]:
    parts = session_dir.resolve().parts
    user = "unknown"
    date = "unknown"
    if "sessions" in parts:
        idx = parts.index("sessions")
        if idx + 2 < len(parts):
            user = parts[idx + 1]
            date = parts[idx + 2]
    time_tag = session_dir.name.split("_")[0]
    pose_mode = str(meta.get("pose_mode") or "locked")
    return user, date, time_tag, pose_mode


def _view_map(metrics: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    views = metrics.get("views") or []
    out: Dict[int, Dict[str, Any]] = {}
    for v in views:
        if not isinstance(v, dict):
            continue
        try:
            ang = int(v.get("angle_deg", 0))
        except Exception:
            ang = 0
        out[ang] = v
    return out


def _key_widths_cm(metrics: Dict[str, Any]) -> Dict[str, float]:
    views = _view_map(metrics)
    front = views.get(0) or {}
    widths = front.get("key_widths_m") or {}
    out: Dict[str, float] = {}
    for k, v in widths.items():
        fv = _safe_float(v)
        if fv is None:
            continue
        out[str(k)] = round(fv * 100.0, 2)
    return out


def _key_ratios(metrics: Dict[str, Any]) -> Dict[str, float]:
    views = _view_map(metrics)
    front = views.get(0) or {}
    widths = front.get("key_widths_m") or {}
    shoulders = float(widths.get("shoulders") or 0.0)
    chest = float(widths.get("chest") or 0.0)
    waist = float(widths.get("waist") or 0.0)
    hips = float(widths.get("hips") or 0.0)
    thigh = float(widths.get("thigh") or 0.0)

    def ratio(a: float, b: float) -> float:
        return round(a / b, 3) if b > 0 else 0.0

    return {
        "shoulder_to_waist": ratio(shoulders, waist),
        "chest_to_waist": ratio(chest, waist),
        "hip_to_waist": ratio(hips, waist),
        "thigh_to_waist": ratio(thigh, waist),
    }


def _median_v_taper(metrics: Dict[str, Any]) -> float:
    views = _view_map(metrics)
    ratios = []
    for angle in (0, 180):
        v = views.get(angle) or {}
        widths = v.get("key_widths_m") or {}
        s = float(widths.get("shoulders") or 0.0)
        w = float(widths.get("waist") or 0.0)
        if w > 0:
            ratios.append(s / w)
    if not ratios:
        return 0.0
    return float(np.median(ratios))


def _category_hint(ratios: Dict[str, float]) -> str:
    v_taper = ratios.get("shoulder_to_waist", 0.0)
    if v_taper >= 1.6:
        return "Mens Physique"
    if v_taper >= 1.45:
        return "Classic Physique"
    return "Bodybuilding"


def _scan_quality_gates(session_dir: Path, meta: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    views_meta = meta.get("views") or meta.get("captures") or []
    views_ok = len(views_meta) >= 4
    rgb_ok = True
    mask_ok = True
    for v in views_meta:
        paths = v.get("paths") or {}
        rgb_rel = paths.get("rgb")
        mask_rel = paths.get("mask")
        if rgb_rel:
            rgb_path = Path(rgb_rel)
            if not rgb_path.is_absolute():
                rgb_path = session_dir / rgb_path
            if not rgb_path.exists():
                rgb_ok = False
        else:
            rgb_ok = False
        if mask_rel:
            mask_path = Path(mask_rel)
            if not mask_path.is_absolute():
                mask_path = session_dir / mask_path
            if not mask_path.exists():
                mask_ok = False
        else:
            mask_ok = False
    silhouette_ok = True
    for v in metrics.get("views") or []:
        if float(v.get("silhouette_area_m2", 0.0)) < 0.08:
            silhouette_ok = False
            break
    passed = views_ok and rgb_ok and mask_ok and silhouette_ok
    return {
        "views_complete": views_ok,
        "rgb_present": rgb_ok,
        "mask_present": mask_ok,
        "silhouette_ok": silhouette_ok,
        "passed": passed,
    }


def _load_or_compute_quality_gates(
    session_dir: Path,
    derived_dir: Path,
    meta: Dict[str, Any],
    metrics: Dict[str, Any],
    scan_quality: Dict[str, Any],
) -> Dict[str, Any]:
    stored = _load_json(derived_dir / "quality_gates.json")
    if isinstance(stored, dict) and stored:
        return stored

    masks: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    for angle in (0, 90, 180, 270):
        view_dir = session_dir / "raw" / f"view_{angle:03d}"
        mask_path = next(view_dir.glob("*_mask.png"), None)
        depth_path = next(view_dir.glob("*_depth.npy"), None)
        if not mask_path or not depth_path:
            masks = []
            depths = []
            break
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            masks = []
            depths = []
            break
        try:
            depth = np.load(str(depth_path))
        except Exception:
            masks = []
            depths = []
            break
        masks.append(mask)
        depths.append(depth)

    if len(masks) == 4 and len(depths) == 4:
        views_meta = meta.get("views") or meta.get("captures") or []
        intrinsics = views_meta[0].get("intrinsics", {}) if views_meta else {}
        try:
            computed = compute_quality_gates(
                view_masks=masks,
                view_depths_m=depths,
                view_points=[None, None, None, None],
                intrinsics=intrinsics,
                metrics=metrics,
                meta=meta,
            )
            _save_json(derived_dir / "quality_gates.json", computed)
            return computed
        except Exception:
            pass

    pose_mode = str(meta.get("pose_mode") or "").strip().lower()
    return {
        "lighting_stable": bool(scan_quality.get("rgb_present", False)),
        "distance_stable": bool(scan_quality.get("views_complete", False)),
        "pose_locked_ok": pose_mode != "free",
        "pointcloud_quality": bool(scan_quality.get("silhouette_ok", False)),
        "reasons": ["quality_gates_fallback_used"],
        "stats": {"fallback": True},
    }


def _strengths_and_priorities(ratios: Dict[str, float], condition_score: float) -> Tuple[List[str], List[str]]:
    strengths: List[str] = []
    priorities: List[str] = []

    if ratios.get("shoulder_to_waist", 0.0) >= 1.6:
        strengths.append("Strong shoulder-to-waist taper.")
    if ratios.get("chest_to_waist", 0.0) >= 1.3:
        strengths.append("Chest-to-waist ratio reads balanced.")
    if condition_score >= 75:
        strengths.append("Condition shows clear separation.")

    if ratios.get("shoulder_to_waist", 0.0) < 1.5:
        priorities.append("Build shoulder width relative to waist.")
    if ratios.get("chest_to_waist", 0.0) < 1.2:
        priorities.append("Bring up upper chest thickness.")
    if condition_score < 70:
        priorities.append("Push conditioning in the next block.")

    while len(strengths) < 3:
        strengths.append("Overall balance is improving.")
    while len(priorities) < 3:
        priorities.append("Add lower-body sweep and detail.")
    return strengths[:3], priorities[:3]


def _update_index(
    session_dir: Path,
    date: str,
    time_tag: str,
    pose_mode: str,
    condition_score: float,
    median_v_taper: float,
) -> Optional[Dict[str, Any]]:
    user_root = session_dir.parents[2]
    index_path = user_root / "index.jsonl"
    existing = []
    if index_path.exists():
        try:
            existing = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            existing = []
    if any(str(row.get("session_dir")) == str(session_dir) for row in existing):
        return None

    entry = {
        "kind": "scan3d",
        "date": date,
        "time": time_tag,
        "pose_mode": pose_mode,
        "session_dir": str(session_dir),
        "condition_score": round(condition_score, 2),
        "median_v_taper": round(median_v_taper, 3),
    }
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def _load_scan_history(session_dir: Path) -> List[Dict[str, Any]]:
    user_root = session_dir.parents[2]
    index_path = user_root / "index.jsonl"
    if not index_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("kind") != "scan3d":
            continue
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("time", ""))))
    return rows


def _find_last_baseline(session_dir: Path, pose_mode: str) -> Optional[Dict[str, Any]]:
    rows = [r for r in _load_scan_history(session_dir) if str(r.get("pose_mode", "")) == pose_mode and r.get("session_dir") != str(session_dir)]
    return rows[-1] if rows else None


def _load_profile(user: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "profiles" / f"{user}.json"
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


def _map_federation_to_kb(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "pca" in s:
        return "PCA"
    if "ifbb" in s:
        return "IFBB_PRO"
    if "ukbff" in s or "wnbf" in s or "uk" in s:
        return "UKBFF"
    return "PCA"


def _federation_categories(kb: Dict[str, Any], federation_id: str) -> List[str]:
    for fed in kb.get("federations", []):
        if str(fed.get("id", "")).upper() != federation_id.upper():
            continue
        return [str(c.get("id")) for c in fed.get("categories", []) if c.get("id")]
    return []


def _map_category_to_kb(raw: str, federation_id: str, kb: Dict[str, Any]) -> Optional[str]:
    avail = _federation_categories(kb, federation_id)
    if not avail:
        return None
    key = (raw or "").lower().replace("'", "").replace(" ", "")

    pref: List[str] = []
    if "mensphysique" in key or ("physique" in key and "classic" not in key):
        pref = ["mens_physique", "classic_physique"]
    elif "classic" in key and "body" in key:
        pref = ["classic_bodybuilding", "classic_physique"]
    elif "classic" in key and "physique" in key:
        pref = ["classic_physique", "classic_bodybuilding"]
    elif "classic" in key:
        pref = ["classic_physique", "classic_bodybuilding"]
    elif "bodybuilding" in key or key == "bodybuilding":
        pref = ["mens_bodybuilding", "bodybuilding", "classic_bodybuilding"]

    for p in pref:
        if p in avail:
            return p

    # Fallback substring matching
    for c in avail:
        if key and key in c.replace("_", ""):
            return c
    return avail[0]


def _session_summary_for_criteria(
    session_dir: Path,
    metrics: Dict[str, Any],
    condition: Dict[str, Any],
    quality_gates: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "session_dir": str(session_dir),
        "metrics": metrics,
        "condition": condition,
        "quality_gates": quality_gates if isinstance(quality_gates, dict) else {},
    }


def _build_criteria_bundle(
    session_dir: Path,
    user: str,
    category_hint: str,
    metrics: Dict[str, Any],
    condition: Dict[str, Any],
    quality_gates: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        kb = load_kb()
    except Exception as exc:
        return {
            "federation_id": None,
            "category_ids": [],
            "results": [],
            "error": str(exc),
        }

    profile = _load_profile(user)
    fed_raw = str(profile.get("federation") or "")
    federation_id = _map_federation_to_kb(fed_raw)
    avail = _federation_categories(kb, federation_id)
    if not avail:
        federation_id = "PCA"
        avail = _federation_categories(kb, federation_id)

    raw_cats: List[str] = []
    for c in (profile.get("selected_categories") or []):
        raw_cats.append(str(c))
    for c in ((profile.get("plan") or {}).get("target_categories") or []):
        raw_cats.append(str(c))
    raw_cats.append(category_hint)

    cat_ids: List[str] = []
    for c in raw_cats:
        mapped = _map_category_to_kb(c, federation_id, kb)
        if mapped and mapped not in cat_ids:
            cat_ids.append(mapped)
    if not cat_ids and avail:
        cat_ids = [avail[0]]

    summary = _session_summary_for_criteria(session_dir, metrics, condition, quality_gates)
    results: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    for cid in cat_ids[:3]:
        try:
            out = score_category(summary, federation_id, cid)
            results.append(out)
            citations.extend(out.get("citations_used") or [])
        except Exception:
            continue

    # de-dup citations
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in citations:
        key = (str(c.get("title", "")), str(c.get("url", "")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    return {
        "federation_id": federation_id,
        "category_ids": cat_ids,
        "results": results,
        "citations": uniq,
    }


def _load_scorecard_for_session(session_dir: Path) -> Dict[str, Any]:
    sc = _load_json(session_dir / "derived" / "scorecard.json")
    if isinstance(sc, dict) and sc:
        return sc
    metrics = _load_json(session_dir / "derived" / "metrics.json") or {}
    condition = _load_json(session_dir / "derived" / "condition.json") or {}
    return {
        "session_id": session_dir.name,
        "ratios": _key_ratios(metrics),
        "key_widths_cm": _key_widths_cm(metrics),
        "condition_score": float(condition.get("condition_score", 0.0)),
        "confidence": str(condition.get("confidence", "low")),
    }


def _scorecard_deltas(current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {"ratios": {}, "widths_cm": {}}
    if not previous:
        return out
    curr_r = current.get("ratios") or {}
    prev_r = previous.get("ratios") or {}
    for k in set(curr_r.keys()) | set(prev_r.keys()):
        a = _safe_float(curr_r.get(k))
        b = _safe_float(prev_r.get(k))
        if a is None or b is None:
            continue
        out["ratios"][k] = round(a - b, 3)

    curr_w = current.get("key_widths_cm") or {}
    prev_w = previous.get("key_widths_cm") or {}
    for k in set(curr_w.keys()) | set(prev_w.keys()):
        a = _safe_float(curr_w.get(k))
        b = _safe_float(prev_w.get(k))
        if a is None or b is None:
            continue
        out["widths_cm"][k] = round(a - b, 2)
    return out


def _history_ratio_series(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    labels: List[str] = []
    shoulder_waist: List[Optional[float]] = []
    chest_waist: List[Optional[float]] = []
    for row in rows:
        sdir = Path(str(row.get("session_dir", "")))
        if not sdir.exists():
            continue
        sc = _load_scorecard_for_session(sdir)
        ratios = sc.get("ratios") or {}
        labels.append(f"{row.get('date','')}\n{row.get('time','')}")
        shoulder_waist.append(_safe_float(ratios.get("shoulder_to_waist")))
        chest_waist.append(_safe_float(ratios.get("chest_to_waist")))
    return {
        "labels": labels,
        "shoulder_to_waist": shoulder_waist,
        "chest_to_waist": chest_waist,
    }


def _symmetry_summary(shape_metrics: Dict[str, Any]) -> List[str]:
    sym = (shape_metrics.get("symmetry_left_right") or {}) if isinstance(shape_metrics, dict) else {}
    lines: List[str] = []
    for key, label in (
        ("thigh_symmetry", "Thigh symmetry"),
        ("arm_symmetry", "Arm symmetry"),
        ("calf_symmetry", "Calf symmetry"),
    ):
        val = _safe_float(sym.get(key))
        if val is None:
            lines.append(f"{label}: not available in this scan.")
            continue
        if val <= 0.05:
            grade = "balanced"
        elif val <= 0.10:
            grade = "mild asymmetry"
        else:
            grade = "notable asymmetry"
        lines.append(f"{label}: {val:.3f} ({grade}).")
    return lines


def _read_ascii_pcd_points(path: Path, max_points: int = 60000) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    points_total = None
    in_data = False
    rows: List[Tuple[float, float, float]] = []

    # First pass: detect point count and header.
    header_lines = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                header_lines.append(line)
                s = line.strip()
                if s.upper().startswith("POINTS"):
                    parts = s.split()
                    if len(parts) >= 2:
                        try:
                            points_total = int(parts[1])
                        except Exception:
                            points_total = None
                if s.upper().startswith("DATA"):
                    in_data = True
                    break
    except Exception:
        return None
    if not in_data:
        return None

    stride = 1
    if points_total and points_total > max_points:
        stride = max(1, int(points_total / float(max_points)))

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            data_started = False
            idx = 0
            for line in f:
                s = line.strip()
                if not data_started:
                    if s.upper().startswith("DATA"):
                        data_started = True
                    continue
                if not s:
                    continue
                if (idx % stride) != 0:
                    idx += 1
                    continue
                parts = s.split()
                if len(parts) < 3:
                    idx += 1
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    rows.append((x, y, z))
                except Exception:
                    pass
                idx += 1
    except Exception:
        return None

    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def _pick_pcd_preview_path(session_dir: Path) -> Optional[Path]:
    merged = session_dir / "exports" / "merged.pcd"
    if merged.exists():
        return merged
    for angle in (0, 90, 180, 270):
        p = session_dir / "raw" / f"view_{angle:03d}" / f"view_{angle:03d}.pcd"
        if p.exists():
            return p
    return None


def _fig_text_wrapped(fig: plt.Figure, x: float, y: float, text: str, width: int = 100, fontsize: int = 10, step: float = 0.022) -> float:
    for line in textwrap.wrap(text, width=width):
        fig.text(x, y, line, fontsize=fontsize)
        y -= step
        if y < 0.05:
            break
    return y


def _fmt_delta(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}"


def _gate_state(v: Any) -> str:
    if v is True:
        return "pass"
    if v is False:
        return "fail"
    return "unknown"


def _make_compact_report(
    out_path: Path,
    session_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    strengths: List[str],
    priorities: List[str],
    last_delta: Optional[Dict[str, float]],
    deltas: Dict[str, Dict[str, float]],
) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    plt.axis("off")

    y = 0.95
    fig.text(0.05, y, "BB Coach — Compact Scan Report", fontsize=18, weight="bold")
    y -= 0.04
    fig.text(0.05, y, f"Session: {scorecard.get('session_id', '-')}")
    y -= 0.03
    fig.text(0.05, y, f"Condition: {scorecard.get('condition_score', '-')} ({scorecard.get('confidence', '-')})")
    if last_delta:
        y -= 0.03
        fig.text(
            0.05,
            y,
            f"Last delta: condition {last_delta.get('condition_score', 0):+.1f}, v-taper {last_delta.get('median_v_taper', 0):+.3f}",
        )
    y -= 0.05

    fig.text(0.05, y, "Ratios", fontsize=12, weight="bold")
    y -= 0.03
    ratios = scorecard.get("ratios") or {}
    for key, val in ratios.items():
        delta_val = (deltas.get("ratios") or {}).get(key)
        fig.text(0.06, y, f"{key.replace('_', ' ').title()}: {val} (Δ {_fmt_delta(delta_val, 3)})")
        y -= 0.024

    y -= 0.01
    fig.text(0.05, y, "Strengths", fontsize=12, weight="bold")
    y -= 0.03
    for item in strengths:
        fig.text(0.06, y, f"- {item}")
        y -= 0.024

    y -= 0.01
    fig.text(0.05, y, "Priorities", fontsize=12, weight="bold")
    y -= 0.03
    for item in priorities:
        fig.text(0.06, y, f"- {item}")
        y -= 0.024

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_full_report(
    out_path: Path,
    session_dir: Path,
    session_summary: Dict[str, Any],
    scorecard: Dict[str, Any],
    metrics: Dict[str, Any],
    condition: Dict[str, Any],
    history_rows: List[Dict[str, Any]],
    criteria_bundle: Dict[str, Any],
    deltas: Dict[str, Dict[str, float]],
    shape_metrics: Dict[str, Any],
    quality_gates: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        # Page 1: Executive summary
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        y = 0.95
        fig.text(0.05, y, "BB Coach — Full Scan Report", fontsize=18, weight="bold")
        y -= 0.04
        fig.text(0.05, y, f"Session: {scorecard.get('session_id', '-')}")
        y -= 0.025
        fig.text(0.05, y, f"Category hint: {session_summary.get('category_hint', '-')}")
        y -= 0.025
        fig.text(0.05, y, f"Condition: {scorecard.get('condition_score', '-')} ({scorecard.get('confidence', '-')})")
        y -= 0.035

        fig.text(0.05, y, "Ratios (with meaning)", fontsize=12, weight="bold")
        y -= 0.028
        for key, val in (scorecard.get("ratios") or {}).items():
            meaning = _RATIO_MEANINGS.get(key, "")
            delta_val = (deltas.get("ratios") or {}).get(key)
            y = _fig_text_wrapped(
                fig,
                0.06,
                y,
                f"{key.replace('_', ' ').title()}: {val} (Δ {_fmt_delta(delta_val, 3)}). {meaning}",
                width=98,
                fontsize=10,
                step=0.021,
            )
            y -= 0.004

        y -= 0.01
        fig.text(0.05, y, "Symmetry", fontsize=12, weight="bold")
        y -= 0.028
        for line in _symmetry_summary(shape_metrics):
            fig.text(0.06, y, f"- {line}")
            y -= 0.022

        y -= 0.008
        fig.text(0.05, y, "Data Quality", fontsize=12, weight="bold")
        y -= 0.028
        scan_quality = session_summary.get("scan_quality") or {}
        for key, val in scan_quality.items():
            fig.text(0.06, y, f"{key}: {val}")
            y -= 0.021

        gates = quality_gates if isinstance(quality_gates, dict) else {}
        for key in ("lighting_stable", "distance_stable", "pose_locked_ok", "pointcloud_quality"):
            fig.text(0.06, y, f"{key}: {_gate_state(gates.get(key))}")
            y -= 0.021

        reasons = gates.get("reasons") if isinstance(gates.get("reasons"), list) else []
        if reasons:
            y = _fig_text_wrapped(
                fig,
                0.06,
                y,
                "reasons: " + ", ".join(str(r) for r in reasons),
                width=95,
                fontsize=9,
                step=0.019,
            )
            y -= 0.004

        stats = gates.get("stats") if isinstance(gates.get("stats"), dict) else {}
        for stat_key in (
            "depth_valid_ratio_mean",
            "depth_valid_ratio_std",
            "distance_variation_ratio",
            "torso_depth_mean_m",
            "point_count_per_view",
            "flow_spikes",
            "flow_curvature_outliers",
            "flow_score",
        ):
            if stat_key not in stats:
                continue
            fig.text(0.06, y, f"{stat_key}: {stats.get(stat_key)}")
            y -= 0.019

        primary_result = (criteria_bundle.get("results") or [None])[0]
        primary_harmony = primary_result.get("harmony") if isinstance(primary_result, dict) else None
        if isinstance(primary_harmony, dict):
            comps = primary_harmony.get("components") or {}
            taper = comps.get("taper") or {}
            lower = comps.get("lower_balance") or {}
            flow = comps.get("flow") or {}
            sym = comps.get("symmetry") or {}

            y -= 0.008
            fig.text(0.05, y, "Harmony Breakdown", fontsize=12, weight="bold")
            y -= 0.026

            tscore = taper.get("score")
            sw = taper.get("shoulder_to_waist")
            ttarget = taper.get("target") or [None, None]
            fig.text(0.06, y, f"Taper: {tscore if tscore is not None else 'N/A'}/100 (Shoulder/Waist={sw}; target {ttarget[0]}-{ttarget[1]})")
            y -= 0.021

            lscore = lower.get("score")
            lh = lower.get("thigh_to_hip")
            ltarget = lower.get("target") or [None, None]
            fig.text(0.06, y, f"Lower Balance: {lscore if lscore is not None else 'N/A'}/100 (Thigh/Hip={lh}; target {ltarget[0]}-{ltarget[1]})")
            y -= 0.021

            fscore = flow.get("score")
            fspikes = flow.get("spikes")
            fcurv = flow.get("curvature_outliers")
            fig.text(0.06, y, f"Flow: {fscore if fscore is not None else 'N/A'}/100 (spikes={fspikes}; curvature_outliers={fcurv})")
            y -= 0.021

            sscore = sym.get("score")
            sthigh = sym.get("thigh_deviation_pct")
            fig.text(0.06, y, f"Symmetry: {sscore if sscore is not None else 'N/A'}/100 (L/R deviation thigh={sthigh}%)")
            y -= 0.021

            hfb = primary_harmony.get("feedback")
            if hfb:
                y = _fig_text_wrapped(fig, 0.06, y, f"Harmony feedback: {hfb}", width=95, fontsize=9, step=0.019)
                y -= 0.004
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Progress trends across sessions
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        rows = history_rows
        if len(rows) >= 2:
            labels = [f"{r.get('date','')}\n{r.get('time','')}" for r in rows]
            x = np.arange(len(rows))
            cond_vals = [float(r.get("condition_score") or 0.0) for r in rows]
            vt_vals = [float(r.get("median_v_taper") or 0.0) for r in rows]

            axes[0].plot(x, cond_vals, marker="o", label="Condition score")
            axes[0].set_title("Condition Progress")
            axes[0].set_ylabel("Score (0-100)")
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

            axes[0].legend(loc="best")

            ratio_series = _history_ratio_series(rows)
            r_labels = ratio_series["labels"]
            r_sw = ratio_series["shoulder_to_waist"]
            r_cw = ratio_series["chest_to_waist"]
            rx = np.arange(len(r_labels))
            if len(r_labels) >= 2:
                axes[1].plot(rx, [v if v is not None else np.nan for v in r_sw], marker="o", label="Shoulder/Waist")
                axes[1].plot(rx, [v if v is not None else np.nan for v in r_cw], marker="o", label="Chest/Waist")
                axes[1].set_title("Ratio Progress")
                axes[1].set_ylabel("Ratio")
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xticks(rx)
                axes[1].set_xticklabels(r_labels, rotation=35, ha="right", fontsize=8)
                axes[1].legend(loc="best")
            else:
                axes[1].axis("off")
                axes[1].text(0.1, 0.5, "Not enough ratio history.", fontsize=12)
        else:
            for ax in axes:
                ax.axis("off")
            axes[0].text(0.1, 0.5, "Not enough historical sessions for progress trends.", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Federation/category scoring and feedback
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        y = 0.95
        fed = criteria_bundle.get("federation_id") or "-"
        fig.text(0.05, y, "Criteria Feedback", fontsize=18, weight="bold")
        y -= 0.035
        fig.text(0.05, y, f"Federation mapping: {fed}")
        y -= 0.03
        results = criteria_bundle.get("results") or []
        if not results:
            fig.text(0.05, y, "No criteria score available for this session/profile mapping.")
        else:
            for res in results:
                cat = res.get("category_id", "-")
                overall = _safe_float(res.get("overall_score"))
                conf = str(res.get("confidence") or "-")
                overall_txt = f"{overall:.1f}" if overall is not None else "N/A"
                y -= 0.01
                fig.text(0.05, y, f"Category: {cat} | Overall: {overall_txt} | Confidence: {conf}", fontsize=11, weight="bold")
                y -= 0.024

                axis_scores = res.get("axis_scores") or {}
                axis_items = []
                for k, v in axis_scores.items():
                    fv = _safe_float(v)
                    axis_items.append(f"{k}={fv:.1f}" if fv is not None else f"{k}=N/A")
                line = "Axis: " + ", ".join(axis_items)
                y = _fig_text_wrapped(fig, 0.06, y, line, width=96, fontsize=9, step=0.02)
                y -= 0.006

                missing_axes = res.get("missing_axes") or []
                if missing_axes:
                    y = _fig_text_wrapped(
                        fig,
                        0.06,
                        y,
                        "Missing axes: " + ", ".join(str(a) for a in missing_axes),
                        width=96,
                        fontsize=9,
                        step=0.02,
                    )
                    y -= 0.004

                res_flags = res.get("flags") or []
                if res_flags:
                    y = _fig_text_wrapped(
                        fig,
                        0.06,
                        y,
                        "Confidence reasons: " + ", ".join(str(f) for f in res_flags),
                        width=96,
                        fontsize=9,
                        step=0.02,
                    )
                    y -= 0.004

                strengths = ", ".join(res.get("strengths") or [])
                priorities = ", ".join(res.get("priorities") or [])
                y = _fig_text_wrapped(fig, 0.06, y, f"Strengths: {strengths}", width=96, fontsize=9, step=0.02)
                y = _fig_text_wrapped(fig, 0.06, y, f"Priorities: {priorities}", width=96, fontsize=9, step=0.02)
                y -= 0.004

                for fb in (res.get("feedback") or [])[:2]:
                    msg = str(fb.get("message") or "")
                    action = fb.get("action") or {}
                    tp = fb.get("training_prescription") or {}
                    muscles = ", ".join(tp.get("muscle_groups") or [])
                    y = _fig_text_wrapped(fig, 0.07, y, f"Feedback: {msg}", width=94, fontsize=9, step=0.02)
                    if muscles:
                        y = _fig_text_wrapped(fig, 0.08, y, f"Target muscles: {muscles}", width=92, fontsize=9, step=0.02)
                    ex_menu = tp.get("exercise_menu") or {}
                    ex_lines = []
                    for grp, exs in ex_menu.items():
                        if not exs:
                            continue
                        ex_lines.append(f"{grp}: {', '.join([str(e) for e in exs[:3]])}")
                    for ex_line in ex_lines[:3]:
                        y = _fig_text_wrapped(fig, 0.08, y, ex_line, width=90, fontsize=8, step=0.018)
                    tr = action.get("training")
                    if tr:
                        y = _fig_text_wrapped(fig, 0.08, y, f"Training cue: {tr}", width=92, fontsize=8, step=0.018)
                    po = action.get("posing")
                    if po:
                        y = _fig_text_wrapped(fig, 0.08, y, f"Posing cue: {po}", width=92, fontsize=8, step=0.018)
                    y -= 0.004
                if y < 0.12:
                    break

            y -= 0.01
            fig.text(0.05, y, "References used", fontsize=11, weight="bold")
            y -= 0.024
            for c in (criteria_bundle.get("citations") or [])[:10]:
                title = str(c.get("title") or "")
                url = str(c.get("url") or "")
                y = _fig_text_wrapped(fig, 0.06, y, f"- {title}: {url}", width=96, fontsize=8, step=0.018)
                if y < 0.08:
                    break

        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: RGB thumbnails
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        angles = [0, 90, 180, 270]
        for ax, angle in zip(axes.flat, angles):
            img_path = session_dir / "media" / f"rgb_{angle:03d}.jpg"
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Missing image", ha="center", va="center")
            ax.set_title(f"RGB View {angle:03d}°")
            ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Sensor/debug captures (RGB + mask + depth)
        fig, axes = plt.subplots(4, 3, figsize=(8.5, 11))
        angles = [0, 90, 180, 270]
        for r, angle in enumerate(angles):
            rgb_path = session_dir / "media" / f"rgb_{angle:03d}.jpg"
            mask_path = next((session_dir / "raw" / f"view_{angle:03d}").glob("*_mask.png"), None)
            depth_path = next((session_dir / "raw" / f"view_{angle:03d}").glob("*_depth.png"), None)

            rgb = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
            if rgb is not None:
                axes[r, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            else:
                axes[r, 0].text(0.5, 0.5, "Missing", ha="center", va="center")
            axes[r, 0].set_title(f"{angle:03d} RGB")
            axes[r, 0].axis("off")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
            if mask is not None:
                axes[r, 1].imshow(mask, cmap="gray")
            else:
                axes[r, 1].text(0.5, 0.5, "Missing", ha="center", va="center")
            axes[r, 1].set_title(f"{angle:03d} Mask")
            axes[r, 1].axis("off")

            dep = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path else None
            if dep is not None:
                if dep.ndim == 3:
                    dep = cv2.cvtColor(dep, cv2.COLOR_BGR2RGB)
                    axes[r, 2].imshow(dep)
                else:
                    axes[r, 2].imshow(dep, cmap="inferno")
            else:
                axes[r, 2].text(0.5, 0.5, "Missing", ha="center", va="center")
            axes[r, 2].set_title(f"{angle:03d} Depth")
            axes[r, 2].axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: Point cloud preview (2D projections)
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        pcd_path = _pick_pcd_preview_path(session_dir)
        pts = _read_ascii_pcd_points(pcd_path, max_points=50000) if pcd_path else None
        if pts is None or pts.size == 0:
            for ax in axes:
                ax.axis("off")
            axes[0].text(0.1, 0.5, "Point cloud preview not available.", fontsize=12)
        else:
            c0 = pts[:, 1]
            axes[0].scatter(pts[:, 0], pts[:, 2], c=c0, s=0.3, cmap="viridis", alpha=0.7)
            axes[0].set_title(f"Point Cloud X-Z Projection ({pcd_path.name})")
            axes[0].set_xlabel("X (m)")
            axes[0].set_ylabel("Z (m)")
            axes[0].grid(True, alpha=0.2)
            axes[0].set_aspect("equal", adjustable="box")

            c1 = pts[:, 2]
            axes[1].scatter(pts[:, 0], pts[:, 1], c=c1, s=0.3, cmap="plasma", alpha=0.7)
            axes[1].set_title("Point Cloud X-Y Projection")
            axes[1].set_xlabel("X (m)")
            axes[1].set_ylabel("Y (m)")
            axes[1].grid(True, alpha=0.2)
            axes[1].set_aspect("equal", adjustable="box")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 7: Width/circumference profiles
        views = _view_map(metrics)
        front = views.get(0) or {}
        widths = np.array(front.get("width_profile_m") or [])
        heights = np.array(front.get("height_bins") or [])
        circum = np.array(metrics.get("circumference_profile_m") or [])
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))

        if widths.size and heights.size:
            axes[0].plot(widths, heights, color="#1f77b4")
            axes[0].invert_yaxis()
            axes[0].set_xlabel("Width (m)")
            axes[0].set_ylabel("Height (normalized)")
            axes[0].set_title("Front View Width Profile")
            axes[0].grid(True, alpha=0.25)
        else:
            axes[0].axis("off")
            axes[0].text(0.1, 0.5, "Width profile not available.", fontsize=12)

        if circum.size and heights.size and circum.size == heights.size:
            axes[1].plot(circum, heights, color="#d62728")
            axes[1].invert_yaxis()
            axes[1].set_xlabel("Estimated Circumference (m)")
            axes[1].set_ylabel("Height (normalized)")
            axes[1].set_title("Estimated Circumference Profile")
            axes[1].grid(True, alpha=0.25)
        else:
            axes[1].axis("off")
            axes[1].text(0.1, 0.5, "Circumference profile not available.", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate scan reports for a session.")
    parser.add_argument("--session_dir", required=True, help="Path to scan3d session directory")
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.exists():
        print(f"[make_report] Session dir not found: {session_dir}")
        return 2

    derived_dir = session_dir / "derived"
    meta = _load_json(derived_dir / "meta.json") or {}
    metrics = _load_json(derived_dir / "metrics.json")
    if metrics is None:
        print("[make_report] Missing derived/metrics.json")
        return 2

    condition = _load_json(derived_dir / "condition.json")
    if condition is None:
        condition = compute_condition_for_session(session_dir, meta, metrics)
        _save_json(derived_dir / "condition.json", condition)

    ratios = _key_ratios(metrics)
    widths_cm = _key_widths_cm(metrics)
    condition_score = float(condition.get("condition_score", 0.0))
    confidence = str(condition.get("confidence", "low"))
    scan_quality = _scan_quality_gates(session_dir, meta, metrics)
    quality_gates = _load_or_compute_quality_gates(session_dir, derived_dir, meta, metrics, scan_quality)
    category_hint = _category_hint(ratios)

    user, date, time_tag, pose_mode = _resolve_session_info(session_dir, meta)
    median_v_taper = _median_v_taper(metrics)
    _update_index(
        session_dir,
        date=date,
        time_tag=time_tag,
        pose_mode=pose_mode,
        condition_score=condition_score,
        median_v_taper=median_v_taper,
    )

    history_rows = _load_scan_history(session_dir)
    baseline = _find_last_baseline(session_dir, pose_mode)
    last_delta = None
    prev_scorecard = None
    if baseline:
        last_delta = {
            "condition_score": condition_score - float(baseline.get("condition_score", 0.0)),
            "median_v_taper": median_v_taper - float(baseline.get("median_v_taper", 0.0)),
        }
        prev_dir = Path(str(baseline.get("session_dir", "")))
        if prev_dir.exists():
            prev_scorecard = _load_scorecard_for_session(prev_dir)

    shape_metrics = compute_shape_metrics(metrics)

    scorecard = {
        "session_id": session_dir.name,
        "pose_mode": pose_mode,
        "ratios": ratios,
        "ratio_meanings": _RATIO_MEANINGS,
        "key_widths_cm": widths_cm,
        "condition_score": condition_score,
        "confidence": confidence,
    }
    deltas = _scorecard_deltas(scorecard, prev_scorecard)
    scorecard["deltas"] = deltas

    criteria_bundle = _build_criteria_bundle(
        session_dir=session_dir,
        user=user,
        category_hint=category_hint,
        metrics=metrics,
        condition=condition,
        quality_gates=quality_gates,
    )

    session_summary = {
        "session_dir": str(session_dir),
        "scan_quality": scan_quality,
        "quality_gates": quality_gates,
        "key_ratios": ratios,
        "key_widths_cm": widths_cm,
        "condition_score": condition_score,
        "confidence": confidence,
        "category_hint": category_hint,
        "last_delta": last_delta,
        "history_count": len(history_rows),
        "shape_metrics": shape_metrics,
        "criteria": {
            "federation_id": criteria_bundle.get("federation_id"),
            "category_ids": criteria_bundle.get("category_ids"),
            "result_count": len(criteria_bundle.get("results") or []),
        },
    }

    _save_json(derived_dir / "session_summary.json", session_summary)
    _save_json(derived_dir / "scorecard.json", scorecard)

    strengths, priorities = _strengths_and_priorities(ratios, condition_score)
    reports_dir = session_dir / "reports"
    _make_compact_report(
        reports_dir / "report_compact.pdf",
        session_summary=session_summary,
        scorecard=scorecard,
        strengths=strengths,
        priorities=priorities,
        last_delta=last_delta,
        deltas=deltas,
    )
    _make_full_report(
        reports_dir / "report_full.pdf",
        session_dir=session_dir,
        session_summary=session_summary,
        scorecard=scorecard,
        metrics=metrics,
        condition=condition,
        history_rows=history_rows,
        criteria_bundle=criteria_bundle,
        deltas=deltas,
        shape_metrics=shape_metrics,
        quality_gates=quality_gates,
    )

    print(f"[make_report] Wrote reports to: {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
