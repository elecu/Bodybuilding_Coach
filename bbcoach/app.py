from __future__ import annotations

from collections import deque
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from .vision.source import OpenCVCameraSource

from .profile import UserProfile
from .vision.pose import PoseBackend
from .vision.overlay import draw_pose_overlay, draw_mask_outline, draw_pose_guide
from .poses.library import POSES, ROUTINES, POSE_GUIDES
from .poses.scoring import score_pose
from .metrics.proportions import compute_from_mask
from .federations.library import cycle_federation, RULES
from .planning.contest_plan import build_prep_summary
from .storage.session import SessionStore
from .utils.time import parse_date, days_until
from .voice.commands import VoiceCommandConfig, VoiceCommandListener


_TEXT_FONT = cv2.FONT_HERSHEY_TRIPLEX
_TEXT_COLOUR = (235, 235, 235)
_TEXT_ACCENT = (0, 230, 255)
_TEXT_BG = (8, 8, 10)


def _draw_text_bg(img, x: int, y: int, w: int, h: int, pad: int = 6, alpha: float = 0.55) -> None:
    y0 = max(0, y - h - pad)
    x0 = max(0, x - pad)
    x1 = min(img.shape[1] - 1, x + w + pad)
    y1 = min(img.shape[0] - 1, y + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _TEXT_BG, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _put_text(
    img,
    text,
    y,
    scale=0.6,
    colour=_TEXT_COLOUR,
    x: int = 12,
    bg: bool = True,
    thickness: int = 1,
):
    (tw, th), _ = cv2.getTextSize(text, _TEXT_FONT, scale, thickness)
    if bg:
        _draw_text_bg(img, x, y, tw, th)
    cv2.putText(img, text, (x, y), _TEXT_FONT, scale, colour, thickness, cv2.LINE_AA)


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


@dataclass
class AutoCaptureConfig:
    enabled: bool = True
    min_score: float = 70.0
    stable_frames: int = 12
    motion_threshold: float = 0.004
    cooldown_frames: int = 6
    top_k: int = 3
    settle_frames: int = 10
    pose_limit_mb: float = 100.0
    pose_cleanup_batch: int = 10


@dataclass
class UIButton:
    key: str
    label: str
    rect: Tuple[int, int, int, int]
    active: bool = False


_CATEGORY_NOTES = {
    "Mens Physique": "Focus on V-taper, shoulders, and tight waist.",
    "Classic": "Focus on balanced proportions and classic lines.",
    "Bodybuilding": "Focus on mass, conditioning, and symmetry.",
}


def _avg_motion(prev: Dict[str, Tuple[float, float]] | None, cur: Dict[str, Tuple[float, float]] | None) -> float:
    if not prev or not cur:
        return 1.0
    keys = prev.keys() & cur.keys()
    if not keys:
        return 1.0
    diffs = []
    for k in keys:
        dx = cur[k][0] - prev[k][0]
        dy = cur[k][1] - prev[k][1]
        diffs.append((dx * dx + dy * dy) ** 0.5)
    return float(np.mean(diffs)) if diffs else 1.0


def _apply_cutout(frame: np.ndarray, mask: Optional[np.ndarray], bg_colour=(0, 0, 0)) -> np.ndarray:
    if mask is None:
        return frame
    blur = cv2.GaussianBlur(mask, (9, 9), 0)
    alpha = blur.astype(np.float32) / 255.0
    alpha = alpha[..., None]
    bg = np.zeros_like(frame, dtype=np.float32)
    bg[:] = bg_colour
    out = frame.astype(np.float32) * alpha + bg * (1 - alpha)
    return out.astype(np.uint8)


def _subject_bbox(
    mask: Optional[np.ndarray],
    landmarks: Optional[Dict[str, Tuple[float, float]]],
    w: int,
    h: int,
) -> Optional[Tuple[int, int, int, int]]:
    if mask is not None:
        ys, xs = np.where(mask > 0)
        if xs.size and ys.size:
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    if landmarks:
        xs = [int(pt[0] * w) for pt in landmarks.values()]
        ys = [int(pt[1] * h) for pt in landmarks.values()]
        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)
    return None


def _crop_to_story(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    landmarks: Optional[Dict[str, Tuple[float, float]]],
) -> np.ndarray:
    target_w, target_h = 1080, 1920
    h, w = frame.shape[:2]
    aspect = target_w / target_h

    bbox = _subject_bbox(mask, landmarks, w, h)
    if bbox:
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
    else:
        cx = w / 2.0
        cy = h / 2.0

    if w / h >= aspect:
        crop_h = h
        crop_w = int(h * aspect)
        cx = max(crop_w / 2, min(w - crop_w / 2, cx))
        x0 = int(cx - crop_w / 2)
        y0 = 0
    else:
        crop_w = w
        crop_h = int(w / aspect)
        cy = max(crop_h / 2, min(h - crop_h / 2, cy))
        x0 = 0
        y0 = int(cy - crop_h / 2)

    crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, (target_w, target_h))


def _draw_story_hud(canvas: np.ndarray, meta: Dict[str, str]) -> None:
    h, w = canvas.shape[:2]
    top_h = 170
    bottom_h = 170
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), _TEXT_BG, -1)
    cv2.rectangle(overlay, (0, h - bottom_h), (w, h), _TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    pose = meta.get("pose", "").upper()
    category = meta.get("category", "").upper()
    score = meta.get("score", "")
    date_tag = meta.get("date", "")
    fed = meta.get("federation", "").upper()
    first_timers = meta.get("first_timers", "").upper()

    cv2.putText(canvas, pose, (40, 70), _TEXT_FONT, 1.0, _TEXT_COLOUR, 2, cv2.LINE_AA)
    cv2.putText(canvas, category, (40, 120), _TEXT_FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    right_x = w - 40
    if date_tag:
        (tw, _), _meta = cv2.getTextSize(date_tag, _TEXT_FONT, 0.7, 1)
        cv2.putText(canvas, date_tag, (right_x - tw, 120), _TEXT_FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    score_text = f"SCORE {score}"
    cv2.putText(canvas, score_text, (40, h - 90), _TEXT_FONT, 1.1, _TEXT_ACCENT, 2, cv2.LINE_AA)
    subline = " | ".join([t for t in [fed, first_timers] if t])
    if subline:
        cv2.putText(canvas, subline, (40, h - 35), _TEXT_FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def _build_story_frame(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    landmarks: Optional[Dict[str, Tuple[float, float]]],
    meta: Dict[str, str],
) -> np.ndarray:
    story = _crop_to_story(frame, mask, landmarks)
    _draw_story_hud(story, meta)
    return story


def _serialize_landmarks(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, list[float]]:
    return {k: [float(v[0]), float(v[1])] for k, v in landmarks.items()}


def _deserialize_landmarks(data: Optional[Dict[str, list[float]]]) -> Optional[Dict[str, Tuple[float, float]]]:
    if not data:
        return None
    out: Dict[str, Tuple[float, float]] = {}
    for k, v in data.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = (float(v[0]), float(v[1]))
    return out or None


def _draw_button(img: np.ndarray, btn: UIButton) -> None:
    x, y, w, h = btn.rect
    overlay = img.copy()
    base = (20, 20, 24)
    on = (30, 160, 200)
    color = on if btn.active else base
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    (tw, th), _ = cv2.getTextSize(btn.label, _TEXT_FONT, 0.5, 1)
    tx = x + max(6, (w - tw) // 2)
    ty = y + h - max(6, (h - th) // 2)
    cv2.putText(img, btn.label, (tx, ty), _TEXT_FONT, 0.5, _TEXT_COLOUR, 1, cv2.LINE_AA)


def _in_rect(pt: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
    x, y = pt
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def run_live(
    profile: UserProfile,
    camera: int | str = 0,
    width: int = 1280,
    height: int = 720,
    voice: bool = False,
    voice_model: Optional[str] = None,
) -> None:
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
    voice_listener: Optional[VoiceCommandListener] = None
    voice_enabled = False
    voice_last_cmd = ""
    voice_error = ""

    def _resolve_voice_model(path_arg: Optional[str]) -> Optional[Path]:
        candidates = []
        if path_arg:
            candidates.append(Path(path_arg))
        root = Path(__file__).resolve().parent.parent
        candidates.append(root / "models" / "vosk")
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                return cand
        return None

    if voice:
        model_path = _resolve_voice_model(voice_model)
        if model_path is None:
            print("[bbcoach] Voice enabled but no Vosk model found.")
            print("[bbcoach] Place a model under models/vosk or pass --voice-model /path/to/model.")
        else:
            try:
                voice_listener = VoiceCommandListener(
                    VoiceCommandConfig(model_path=str(model_path))
                )
                voice_listener.start()
                voice_enabled = True
                print("[bbcoach] Voice commands enabled (say: 'next pose' / 'siguiente pose').")
            except Exception as exc:
                voice_error = str(exc)
                print(f"[bbcoach] Voice init failed: {voice_error}")

    window_name = "BB Coach (Webcam)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    mouse = {"click": None}

    def _on_mouse(event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)

    cv2.setMouseCallback(window_name, _on_mouse)
    cv2.resizeWindow(window_name, width, height)

    # Categories may contain multiple selections; pick the first for live routine.
    cat = profile.selected_categories[0] if profile.selected_categories else "Mens Physique"
    routine = ROUTINES.get(cat, ROUTINES["Mens Physique"])
    pose_i = 0

    # Snapshot template capture buffer
    stability_buf: list[dict] = []
    auto_cfg = AutoCaptureConfig()
    frame_idx = 0
    last_landmarks: Optional[Dict[str, Tuple[float, float]]] = None
    motion_buf: deque[float] = deque(maxlen=auto_cfg.stable_frames)
    last_capture_frame = -9999
    pose_switch_frame = 0
    pose_captures: Dict[str, list[dict]] = {}
    guide_enabled = True
    cutout_enabled = False
    show_info = False
    show_category_menu = False
    fullscreen = False
    window_size: Optional[Tuple[int, int]] = None

    def _finalize_pose(pose_key: str) -> None:
        shots = pose_captures.get(pose_key, [])
        if len(shots) <= auto_cfg.top_k:
            return
        shots.sort(key=lambda s: s.get("score", 0.0), reverse=True)
        keep = shots[: auto_cfg.top_k]
        drop = shots[auto_cfg.top_k :]
        for entry in drop:
            _cleanup_capture(entry)
        pose_captures[pose_key] = keep

    def _set_pose_index(new_idx: int) -> None:
        nonlocal pose_i, pose_switch_frame
        if routine:
            _finalize_pose(routine[pose_i])
        pose_i = new_idx % len(routine)
        pose_switch_frame = frame_idx
        motion_buf.clear()

    def _set_category(new_cat: str) -> None:
        nonlocal cat, routine, pose_i, show_category_menu, pose_switch_frame
        if routine:
            _finalize_pose(routine[pose_i])
        cat = new_cat
        if cat not in profile.selected_categories:
            profile.selected_categories.insert(0, cat)
        profile.plan.target_categories = list(profile.selected_categories)
        routine = ROUTINES.get(cat, ROUTINES["Mens Physique"])
        pose_i = 0
        pose_switch_frame = frame_idx
        show_category_menu = False

    def _apply_window_state() -> None:
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            if window_size:
                cv2.resizeWindow(window_name, window_size[0], window_size[1])

    def _build_snapshot_payload(pose_key: str, score: float, props_obj) -> dict:
        return {
            "profile": profile.name,
            "federation": profile.federation,
            "category": cat,
            "pose": pose_key,
            "pose_score": float(score),
            "pose_features": ps.per_feature,
            "proportions": asdict(props_obj) if props_obj is not None else None,
            "competition_date": profile.plan.competition_date,
            "first_timers": profile.first_timers,
            "prep_mode": profile.prep.mode,
        }

    def _cleanup_capture(entry: dict) -> None:
        for key in ("full_path", "cutout_path", "story_path"):
            path = entry.get(key)
            if isinstance(path, Path):
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
                index_path = path.parent / "index.json"
                if index_path.exists():
                    try:
                        data = json.loads(index_path.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            data = [row for row in data if row.get("file") != path.name]
                            index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    except Exception:
                        pass

    def _save_auto_capture(
        pose_key: str,
        score: float,
        frame_bgr: np.ndarray,
        cutout_bgr: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        landmarks: Optional[Dict[str, Tuple[float, float]]],
        props_obj,
        pose_display: str,
    ) -> dict:
        payload = _build_snapshot_payload(pose_key, score, props_obj)
        capture_id = uuid.uuid4().hex
        payload["capture_id"] = capture_id
        entry = {"score": float(score)}
        entry["full_path"] = sessions.save_capture(profile.name, payload, frame_bgr, variant="full")
        if cutout_bgr is not None:
            entry["cutout_path"] = sessions.save_capture(profile.name, payload, cutout_bgr, variant="cutout")

        story_src = cutout_bgr if cutout_bgr is not None else frame_bgr
        meta = {
            "pose": pose_display,
            "category": cat,
            "score": f"{score:.1f}",
            "date": date.today().strftime("%Y-%m-%d"),
            "federation": RULES[profile.federation].display_name,
            "first_timers": f"FIRST TIMERS {'YES' if profile.first_timers else 'NO'}",
        }
        story = _build_story_frame(story_src, mask, landmarks, meta)
        entry["story_path"] = sessions.save_capture(profile.name, payload, story, variant="story")
        return entry

    print("Live coach started. Press Q to quit.")

    while True:
        ok, frame, _ts = source.read()
        if not ok:
            break

        frame_idx += 1
        frame = cv2.flip(frame, 1)
        res = pose.process_bgr(frame)

        if res.landmarks:
            if last_landmarks:
                motion_buf.append(_avg_motion(last_landmarks, res.landmarks))
            last_landmarks = res.landmarks
        else:
            last_landmarks = None
            motion_buf.clear()

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
            _set_category(sug_cat)

        pose_key = routine[pose_i]
        pose_def = POSES[pose_key]

        # Score pose
        tpl = profile.templates.get(pose_key)
        tpl_feats = tpl.get("features") if tpl else None
        tpl_landmarks = _deserialize_landmarks(tpl.get("landmarks")) if tpl else None
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

        cutout_frame = _apply_cutout(frame, res.mask, bg_colour=(10, 10, 10)) if res.mask is not None else None
        display_frame = cutout_frame if cutout_enabled and cutout_frame is not None else frame
        guide_landmarks = None
        if guide_enabled:
            guide_landmarks = tpl_landmarks or POSE_GUIDES.get(pose_key)

        out = display_frame
        if guide_landmarks:
            out = draw_pose_guide(out, guide_landmarks)
        out = draw_pose_overlay(out, res.landmarks, joint_ok=joint_ok, line_ok=line_ok)
        if res.mask is not None and not cutout_enabled:
            out = draw_mask_outline(out, res.mask)

        if not fullscreen:
            size = (out.shape[1], out.shape[0])
            if window_size != size:
                window_size = size
                cv2.resizeWindow(window_name, size[0], size[1])

        stable = False
        if len(motion_buf) >= auto_cfg.stable_frames:
            stable = float(np.mean(motion_buf)) < auto_cfg.motion_threshold

        can_capture = (
            auto_cfg.enabled
            and stable
            and bool(res.landmarks)
            and ps.score_0_100 >= auto_cfg.min_score
            and (frame_idx - last_capture_frame) >= auto_cfg.cooldown_frames
            and (frame_idx - pose_switch_frame) >= auto_cfg.settle_frames
        )
        if can_capture:
            entry = _save_auto_capture(
                pose_key,
                ps.score_0_100,
                frame,
                cutout_frame,
                res.mask,
                res.landmarks,
                props,
                pose_def.display,
            )
            last_capture_frame = frame_idx
            pose_captures.setdefault(pose_key, []).append(entry)
            removed = sessions.enforce_pose_storage_limit(
                profile.name,
                pose_key,
                limit_mb=auto_cfg.pose_limit_mb,
                batch_size=auto_cfg.pose_cleanup_batch,
            )
            if removed:
                removed_names = {p.name for p in removed}
                updated = []
                for shot in pose_captures.get(pose_key, []):
                    drop = False
                    for key in ("full_path", "cutout_path", "story_path"):
                        path = shot.get(key)
                        if isinstance(path, Path) and path.name in removed_names:
                            drop = True
                            break
                    if not drop:
                        updated.append(shot)
                pose_captures[pose_key] = updated

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
        ft = "YES" if profile.first_timers else "NO"
        _put_text(
            out,
            f"FEDERATION {fed_rules.display_name} | FIRST TIMERS {ft}",
            y,
            scale=0.5,
            thickness=1,
        )
        y += 22
        _put_text(
            out,
            f"CATEGORY {cat.upper()} | POSE {pose_def.display.upper()} ({pose_i+1}/{len(routine)})",
            y,
            scale=0.52,
            thickness=1,
        )
        y += 24
        _put_text(out, f"SCORE {ps.score_0_100:.1f}", y, scale=0.7, colour=_TEXT_ACCENT, thickness=2)
        y += 30
        auto_status = "On" if auto_cfg.enabled else "Off"
        guide_status = "On" if guide_enabled else "Off"
        cutout_status = "On" if cutout_enabled else "Off"
        captures = len(pose_captures.get(pose_key, []))
        _put_text(
            out,
            f"AUTO {auto_status.upper()} | GUIDE {guide_status.upper()} | CUTOUT {cutout_status.upper()}",
            y,
            scale=0.5,
            thickness=1,
        )
        y += 22
        voice_status = "On" if voice_enabled else "Off"
        voice_line = f"Voice: {voice_status}"
        if voice_last_cmd:
            voice_line += f" | Last: {voice_last_cmd}"
        if voice_error:
            voice_line += " | Error"
        _put_text(out, voice_line.upper(), y, scale=0.48, colour=(200, 200, 200), thickness=1)
        y += 20
        _put_text(
            out,
            f"AUTO CAPTURES {captures} | KEEP BEST {auto_cfg.top_k} ON NEXT POSE | STABLE {'YES' if stable else 'NO'}",
            y,
            scale=0.48,
            colour=(200, 200, 200),
            thickness=1,
        )
        y += 22

        if dts is not None:
            _put_text(out, f"COUNTDOWN {dts} DAYS TO SHOW", y, scale=0.5, thickness=1)
            y += 22

        # Proportions display
        if props is not None:
            sw = _fmt_num(props.shoulder_to_waist)
            cw = _fmt_num(props.chest_to_waist)
            hw = _fmt_num(props.hip_to_waist)
            ul = _fmt_num(props.upper_to_lower_area)
            _put_text(
                out,
                f"S/W {sw} | C/W {cw} | H/W {hw} | U/L AREA {ul}",
                y,
                scale=0.5,
                thickness=1,
            )
            y += 22

        # Guidance lines
        guide = pose_def.guidance[:2]
        for g in guide:
            _put_text(out, g, y, scale=0.48, colour=(190, 190, 190), thickness=1)
            y += 20

        def build_buttons() -> list[UIButton]:
            buttons: list[UIButton] = []
            btn_h = 26
            btn_gap = 8
            btn_y = out.shape[0] - 34
            btn_x = 12
            auto_label = "On" if auto_cfg.enabled else "Off"
            cutout_label = "On" if cutout_enabled else "Off"
            guide_label = "On" if guide_enabled else "Off"

            def add(key: str, label: str, active: bool = False) -> None:
                nonlocal btn_x
                (tw, _), _meta = cv2.getTextSize(label, _TEXT_FONT, 0.5, 1)
                w = tw + 18
                buttons.append(UIButton(key=key, label=label, rect=(btn_x, btn_y, w, btn_h), active=active))
                btn_x += w + btn_gap

            add("prev_pose", "PREV")
            add("next_pose", "NEXT")
            add("auto", f"AUTO {auto_label.upper()}", active=auto_cfg.enabled)
            add("cutout", f"CUTOUT {cutout_label.upper()}", active=cutout_enabled)
            add("guide", f"GUIDE {guide_label.upper()}", active=guide_enabled)

            btn_y2 = btn_y - (btn_h + 8)
            btn_x2 = 12

            def add_row2(key: str, label: str, active: bool = False) -> None:
                nonlocal btn_x2
                (tw, _), _meta = cv2.getTextSize(label, _TEXT_FONT, 0.5, 1)
                w = tw + 18
                buttons.append(UIButton(key=key, label=label, rect=(btn_x2, btn_y2, w, btn_h), active=active))
                btn_x2 += w + btn_gap

            add_row2("category_menu", f"CATEGORY {cat.upper()}", active=show_category_menu)
            add_row2("info", "INFO", active=show_info)
            if voice_listener is not None:
                voice_label = "On" if voice_enabled else "Off"
                add_row2("voice", f"VOICE {voice_label.upper()}", active=voice_enabled)
            full_label = "On" if fullscreen else "Off"
            add_row2("fullscreen", f"FULL {full_label.upper()}", active=fullscreen)

            if show_category_menu:
                menu_y = btn_y2 - (btn_h + 8)
                for cat_key in ROUTINES.keys():
                    (tw, _), _meta = cv2.getTextSize(cat_key, _TEXT_FONT, 0.5, 1)
                    w = tw + 18
                    buttons.append(
                        UIButton(
                            key=f"cat:{cat_key}",
                            label=cat_key,
                            rect=(12, menu_y, w, btn_h),
                            active=(cat_key == cat),
                        )
                    )
                    menu_y -= btn_h + 6

            return buttons

        buttons = build_buttons()
        click = mouse.get("click")
        if click:
            for btn in buttons:
                if _in_rect(click, btn.rect):
                    if btn.key == "prev_pose":
                        _set_pose_index(pose_i - 1)
                    elif btn.key == "next_pose":
                        _set_pose_index(pose_i + 1)
                    elif btn.key == "auto":
                        auto_cfg.enabled = not auto_cfg.enabled
                    elif btn.key == "cutout":
                        cutout_enabled = not cutout_enabled
                    elif btn.key == "guide":
                        guide_enabled = not guide_enabled
                    elif btn.key == "category_menu":
                        show_category_menu = not show_category_menu
                    elif btn.key == "info":
                        show_info = not show_info
                    elif btn.key == "voice":
                        if voice_listener is not None:
                            voice_enabled = not voice_enabled
                    elif btn.key == "fullscreen":
                        fullscreen = not fullscreen
                        _apply_window_state()
                    elif btn.key.startswith("cat:"):
                        _set_category(btn.key.split(":", 1)[1])
                    break
            mouse["click"] = None
            buttons = build_buttons()

        for btn in buttons:
            _draw_button(out, btn)

        if show_info:
            routine_names = [POSES[p].display for p in routine]
            info_lines = [
                f"{cat} info",
                _CATEGORY_NOTES.get(cat, "Category info unavailable."),
                f"First Timers: {'Yes' if profile.first_timers else 'No'}",
                "Routine:",
            ] + [f"- {name}" for name in routine_names]
            panel_w = min(520, out.shape[1] - 24)
            panel_h = 20 * len(info_lines) + 16
            panel_x = max(12, out.shape[1] - panel_w - 12)
            panel_y = 18
            overlay = out.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
            ty = panel_y + 24
            for line in info_lines:
                cv2.putText(out, line, (panel_x + 12, ty), _TEXT_FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                ty += 20

        cv2.imshow(window_name, out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

        if key in (ord('n'), ord('N')):
            _set_pose_index(pose_i + 1)

        if key in (ord('p'), ord('P')):
            _set_pose_index(pose_i - 1)

        if key in (ord('f'), ord('F')):
            profile.federation = cycle_federation(profile.federation)  # type: ignore

        if key in (ord('t'), ord('T')):
            profile.first_timers = not profile.first_timers
            profile.plan.first_timers = profile.first_timers

        if key in (ord('c'), ord('C')):
            # Cycle a small list; user can add more via profile editor
            cycle = ["Mens Physique", "Classic", "Bodybuilding"]
            cur = cycle.index(cat) if cat in cycle else 0
            _set_category(cycle[(cur + 1) % len(cycle)])

        if key in (ord('a'), ord('A')):
            auto_cfg.enabled = not auto_cfg.enabled

        if key in (ord('g'), ord('G')):
            guide_enabled = not guide_enabled

        if key in (ord('b'), ord('B')):
            cutout_enabled = not cutout_enabled

        if key in (ord('i'), ord('I')):
            show_info = not show_info

        if key in (ord('m'), ord('M')):
            show_category_menu = not show_category_menu

        if key in (ord('v'), ord('V')) and voice_listener is not None:
            voice_enabled = not voice_enabled

        if key in (ord('z'), ord('Z')):
            fullscreen = not fullscreen
            _apply_window_state()

        if voice_listener is not None:
            if voice_listener.error() and not voice_error:
                voice_error = voice_listener.error() or "voice error"
                voice_enabled = False
            if voice_enabled:
                cmd = voice_listener.pop_command()
                if cmd:
                    voice_last_cmd = cmd
                    if cmd == "next_pose":
                        _set_pose_index(pose_i + 1)
                    elif cmd == "prev_pose":
                        _set_pose_index(pose_i - 1)

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
                    "landmarks": _serialize_landmarks(best["lm"]),
                }
                stability_buf.clear()

        if key in (ord('s'), ord('S')):
            snap = _build_snapshot_payload(pose_key, ps.score_0_100, props)
            outp = sessions.save_snapshot(profile.name, snap)
            print(f"Saved snapshot: {outp}")

    if routine:
        _finalize_pose(routine[pose_i])

    if voice_listener is not None:
        voice_listener.stop()

    source.close()
    cv2.destroyAllWindows()


def compute_template_features(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    # Store *absolute* feature values from the captured frame.
    from .metrics.pose_features import compute_features, PoseLandmarks

    feats = compute_features(PoseLandmarks(pts=landmarks))
    return feats
