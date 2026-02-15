from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .poses.library import POSES
from .storage.session import SessionStore


@dataclass(frozen=True)
class CapturePick:
    date_tag: str
    file_path: Path
    score: float


def _norm_text(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return " ".join(cleaned.split())


def _resolve_pose_key(pose_arg: str) -> str:
    if pose_arg in POSES:
        return pose_arg
    target = _norm_text(pose_arg)
    for key, pose in POSES.items():
        if target == _norm_text(pose.display):
            return key
    raise ValueError(f"Unknown pose '{pose_arg}'. Use a pose key like: {', '.join(POSES.keys())}")


def _list_date_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    out = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.isdigit() and len(name) == 8:
            out.append(p)
            continue
        if len(name) == 10 and name[4] == "-" and name[7] == "-":
            out.append(p)
    return sorted(out)


def _best_capture_for_date(date_dir: Path, pose_key: str, variant: str) -> Optional[CapturePick]:
    poses_dir = date_dir / "poses"
    if poses_dir.exists():
        picks = []
        for session_dir in poses_dir.iterdir():
            if not session_dir.is_dir():
                continue
            meta_path = session_dir / "derived" / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(meta.get("pose") or meta.get("pose_id") or "") != pose_key:
                continue
            score = meta.get("pose_score")
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0
            media = meta.get("media") or {}
            rel = media.get(variant)
            if rel:
                path = session_dir / rel
            else:
                media_dir = session_dir / "media"
                path = next(media_dir.glob(f"*_{variant}.jpg"), None)
            if not path or not path.exists():
                continue
            picks.append((score_f, path))
        if not picks:
            return None
        best_score, best_path = max(picks, key=lambda x: x[0])
        return CapturePick(date_tag=date_dir.name, file_path=best_path, score=best_score)

    # Legacy fallback: sessions/captures/<profile>/<YYYYMMDD>/index.json
    index_path = date_dir / "index.json"
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    picks = []
    for entry in data:
        if entry.get("pose") != pose_key:
            continue
        if entry.get("variant") != variant:
            continue
        score = entry.get("pose_score")
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        picks.append((score_f, entry.get("file")))
    if not picks:
        return None
    best_score, best_file = max(picks, key=lambda x: x[0])
    if not best_file:
        return None
    path = date_dir / str(best_file)
    if not path.exists():
        return None
    return CapturePick(date_tag=date_dir.name, file_path=path, score=best_score)


def _pick_dates(available: list[str], date_a: Optional[str], date_b: Optional[str]) -> Optional[Tuple[str, str]]:
    if date_a and date_b:
        return date_a, date_b
    if date_a:
        others = [d for d in available if d != date_a]
        return (date_a, others[-1]) if others else None
    if date_b:
        others = [d for d in available if d != date_b]
        return (others[-1], date_b) if others else None
    if len(available) >= 2:
        return available[-2], available[-1]
    return None


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    scale = height / float(h)
    nw = max(1, int(w * scale))
    return cv2.resize(img, (nw, height))


def run_compare(
    profile_name: str,
    pose: str,
    date_a: Optional[str] = None,
    date_b: Optional[str] = None,
    variant: str = "cutout",
    out_path: Optional[str] = None,
) -> None:
    pose_key = _resolve_pose_key(pose)
    sessions = SessionStore.default()
    base = sessions.root / profile_name
    date_dirs = _list_date_dirs(base)
    if not date_dirs:
        base = sessions.root / "captures" / profile_name
        date_dirs = _list_date_dirs(base)
    if not date_dirs:
        print("No capture sessions found.")
        return

    available = []
    picks: dict[str, CapturePick] = {}
    for d in date_dirs:
        pick = _best_capture_for_date(d, pose_key, variant)
        if pick is None:
            continue
        available.append(d.name)
        picks[d.name] = pick

    pair = _pick_dates(available, date_a, date_b)
    if pair is None:
        print("Not enough captures for that pose/variant.")
        return

    a_tag, b_tag = pair
    pick_a = picks.get(a_tag)
    pick_b = picks.get(b_tag)
    if pick_a is None or pick_b is None:
        print("Could not find captures for the selected dates.")
        return

    img_a = cv2.imread(str(pick_a.file_path))
    img_b = cv2.imread(str(pick_b.file_path))
    if img_a is None or img_b is None:
        print("Could not read capture images.")
        return

    target_h = 720
    img_a = _resize_to_height(img_a, target_h)
    img_b = _resize_to_height(img_b, target_h)

    gap = 24
    label_h = 72
    h = target_h + label_h
    w = img_a.shape[1] + img_b.shape[1] + gap
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (10, 10, 10)

    canvas[label_h : label_h + target_h, : img_a.shape[1]] = img_a
    canvas[label_h : label_h + target_h, img_a.shape[1] + gap :] = img_b

    pose_label = POSES[pose_key].display
    cv2.putText(canvas, pose_label, (18, 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    left_label = f"{a_tag} | Score {pick_a.score:.1f}"
    right_label = f"{b_tag} | Score {pick_b.score:.1f}"
    cv2.putText(canvas, left_label, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    right_x = img_a.shape[1] + gap + 18
    cv2.putText(canvas, right_label, (right_x, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    if out_path:
        out = Path(out_path)
    else:
        out = base / f"compare_{pose_key}_{a_tag}_vs_{b_tag}_{variant}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"Saved comparison: {out}")
