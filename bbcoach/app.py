from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Dict, Tuple

import cv2
import numpy as np

from .vision.source import OpenCVCameraSource

from .profile import UserProfile
from .vision.pose import PoseBackend
from .vision.overlay import draw_pose_overlay, draw_mask_outline
from .poses.library import POSES, ROUTINES
from .poses.scoring import score_pose
from .metrics.proportions import compute_from_mask
from .federations.library import cycle_federation, RULES
from .planning.contest_plan import build_prep_summary
from .storage.session import SessionStore
from .utils.time import parse_date, days_until


def _put_text(img, text, y, scale=0.6, colour=(255, 255, 255)):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def _fmt_num(x: float | None, nd: int = 2) -> str:
    """Format optional numbers for on-screen display."""
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _safe_current_weight(profile: UserProfile) -> float | None:
    if not profile.bodyweight_log:
        return None
    try:
        return float(profile.bodyweight_log[-1]["weight_kg"])
    except Exception:
        return None


def _suggest_category(profile: UserProfile, props) -> tuple[str, str]:
    """Very light heuristic: prioritise Men's Physique if V-taper looks strong.

    This is *not* a judge. It only helps decide what routine to start with.
    """
    if props.shoulder_to_waist and props.shoulder_to_waist >= 1.35:
        return ("Mens Physique", "Strong shoulder-to-waist ratio suggests Men's Physique as a good starting point.")
    if props.upper_to_lower_area and props.upper_to_lower_area < 0.95:
        return ("Mens Physique", "Upper body silhouette looks behind lower body; Men's Physique lets you focus on upper aesthetics.")
    return ("Classic", "Proportions look balanced; Classic can be a good long-term path if you enjoy mandatory poses.")


def run_live(profile: UserProfile, camera: int | str = 0, width: int = 1280, height: int = 720) -> None:
    # Video source (webcam-first). This is intentionally abstracted so we can
    # later drop in Kinect v2 without rewriting the coach logic.
    source = OpenCVCameraSource(device=camera, width=width, height=height)
    try:
        source.open()
    except RuntimeError as e:
        # Friendly diagnostics for Linux webcam indexing quirks.
        from pathlib import Path

        vids = sorted(str(p) for p in Path("/dev").glob("video*"))
        print(f"[bbcoach] {e}")
        if vids:
            print("[bbcoach] Available V4L2 nodes:")
            for v in vids:
                print(f"  - {v}")
            print("[bbcoach] Try: python -m bbcoach live --profile <name> --camera /dev/video2")
        return

    try:
        pose = PoseBackend()
    except ModuleNotFoundError:
        print("[bbcoach] Missing dependency: mediapipe")
        print("[bbcoach] Activate your venv and install requirements:")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        source.close()
        return
    except RuntimeError as e:
        print(f"[bbcoach] {e}")
        source.close()
        return
    sessions = SessionStore.default()

    # Categories may contain multiple selections; pick the first for live routine.
    cat = profile.selected_categories[0] if profile.selected_categories else "Mens Physique"
    routine = ROUTINES.get(cat, ROUTINES["Mens Physique"])
    pose_i = 0

    # Snapshot template capture buffer
    stability_buf: list[dict] = []

    print("Live coach started. Press Q to quit.")

    while True:
        ok, frame, _ts = source.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        res = pose.process_bgr(frame)

        # Proportions from segmentation mask (if available)
        props = None
        if res.mask is not None:
            props = compute_from_mask(res.mask)

        # Suggest category on first usable mask
        if props is not None and (not profile.plan.target_categories):
            sug_cat, why = _suggest_category(profile, props)
            profile.plan.target_categories = [sug_cat]
            if sug_cat not in profile.selected_categories:
                profile.selected_categories = [sug_cat]
            cat = sug_cat
            routine = ROUTINES.get(cat, ROUTINES["Mens Physique"])
            pose_i = 0

        pose_key = routine[pose_i]
        pose_def = POSES[pose_key]

        # Score pose
        tpl = profile.templates.get(pose_key)
        tpl_feats = tpl.get("features") if tpl else None
        ps = score_pose(res.landmarks, pose_def.target, pose_def.tolerance, template_override=tpl_feats)

        # Map feature ok flags to joint/line ok flags (simple default)
        joint_ok = {k: True for k in res.landmarks.keys()}
        line_ok = {}

        # We highlight shoulders and elbows when key features fail
        if ps.ok_flags:
            if not ps.ok_flags.get("shoulder_level", True):
                joint_ok["left_shoulder"] = False
                joint_ok["right_shoulder"] = False
            if not ps.ok_flags.get("elbow_sym", True) or not ps.ok_flags.get("elbow_height", True):
                joint_ok["left_elbow"] = False
                joint_ok["right_elbow"] = False

        out = draw_pose_overlay(frame, res.landmarks, joint_ok=joint_ok, line_ok=line_ok)
        if res.mask is not None:
            out = draw_mask_outline(out, res.mask)

        # UI text
        fed_rules = RULES[profile.federation]
        comp_d = parse_date(profile.plan.competition_date or "")
        dts = days_until(comp_d)

        prep = build_prep_summary(
            competition_date=profile.plan.competition_date,
            mode=profile.prep.mode,
            start_weight=profile.prep.start_weight_kg,
            current_weight=_safe_current_weight(profile),
            start_date=profile.prep.start_date,
        )

        y = 26
        _put_text(out, f"Federation: {fed_rules.display_name} | First Timers: {'Yes' if profile.first_timers else 'No'}", y)
        y += 26
        _put_text(out, f"Category: {cat} | Pose: {pose_def.display} ({pose_i+1}/{len(routine)})", y)
        y += 26
        _put_text(out, f"Pose score: {ps.score_0_100:.1f}/100", y)
        y += 26

        if dts is not None:
            _put_text(out, f"Countdown: {dts} days to show", y)
            y += 26

        # Proportions display
        if props is not None:
            sw = _fmt_num(props.shoulder_to_waist)
            cw = _fmt_num(props.chest_to_waist)
            hw = _fmt_num(props.hip_to_waist)
            ul = _fmt_num(props.upper_to_lower_area)
            _put_text(out, f"S/W: {sw} | C/W: {cw} | H/W: {hw} | Upper/Lower area: {ul}", y)
            y += 26

        # Guidance lines
        guide = pose_def.guidance[:2]
        for g in guide:
            _put_text(out, g, y, scale=0.55, colour=(220, 220, 220))
            y += 22

        cv2.imshow("BB Coach (Webcam)", out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

        if key in (ord('n'), ord('N')):
            pose_i = (pose_i + 1) % len(routine)

        if key in (ord('p'), ord('P')):
            pose_i = (pose_i - 1) % len(routine)

        if key in (ord('f'), ord('F')):
            profile.federation = cycle_federation(profile.federation)  # type: ignore

        if key in (ord('t'), ord('T')):
            profile.first_timers = not profile.first_timers
            profile.plan.first_timers = profile.first_timers

        if key in (ord('c'), ord('C')):
            # Cycle a small list; user can add more via profile editor
            cycle = ["Mens Physique", "Classic", "Bodybuilding"]
            cur = cycle.index(cat) if cat in cycle else 0
            cat = cycle[(cur + 1) % len(cycle)]
            if cat not in profile.selected_categories:
                profile.selected_categories.insert(0, cat)
            routine = ROUTINES.get(cat, ROUTINES["Mens Physique"])
            pose_i = 0

        if key == ord(' '):
            # Auto-capture a stable template: we buffer 20 frames and pick the one
            # with the most consistent shoulders/hips.
            if res.landmarks:
                stability_buf.append({"lm": res.landmarks, "score": ps.score_0_100})
            if len(stability_buf) >= 20:
                # pick the frame with highest score
                best = max(stability_buf, key=lambda d: float(d["score"]))
                profile.templates[pose_key] = {
                    "captured": date.today().isoformat(),
                    "features": compute_template_features(best["lm"]),
                }
                stability_buf.clear()

        if key in (ord('s'), ord('S')):
            snap = {
                "profile": profile.name,
                "federation": profile.federation,
                "category": cat,
                "pose": pose_key,
                "pose_score": ps.score_0_100,
                "pose_features": ps.per_feature,
                "proportions": asdict(props) if props is not None else None,
                "competition_date": profile.plan.competition_date,
                "first_timers": profile.first_timers,
                "prep_mode": profile.prep.mode,
            }
            outp = sessions.save_snapshot(profile.name, snap)
            print(f"Saved snapshot: {outp}")

    source.close()
    cv2.destroyAllWindows()


def compute_template_features(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    # Store *absolute* feature values from the captured frame.
    from .metrics.pose_features import compute_features, PoseLandmarks

    feats = compute_features(PoseLandmarks(pts=landmarks))
    return feats
