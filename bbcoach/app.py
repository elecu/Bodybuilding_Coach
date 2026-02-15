from __future__ import annotations

from collections import deque
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from .vision.source import open_source

from .profile import UserProfile, ProfileStore, VideoConfig
from .vision.pose import PoseBackend
from .vision.overlay import draw_pose_overlay, draw_mask_outline, draw_pose_guide
from .vision.depth_seg import segment_person_depth
from .poses.library import POSES, ROUTINES, POSE_GUIDES, routine_for
from .poses.scoring import PoseScore, score_pose
from .metrics.proportions import compute_from_mask
from .app_state import get_global_app_state
from .federations.library import cycle_federation, RULES
from .federations.pose_checklist import PoseChecklistItem, build_pose_checklist, selected_division_labels
from .federations.specs import SelectedDivisionRef, load_federation_specs
from .planning.contest_plan import build_prep_summary
from .storage.session import SessionStore
from .ui.tabs import MetricsTab
from .utils.time import parse_date, days_until
from .voice.commands import VoiceCommandConfig, VoiceCommandListener
from .voice.tts import TTSSpeaker


_TEXT_FONT = cv2.FONT_HERSHEY_TRIPLEX
_TEXT_COLOUR = (235, 235, 235)
_TEXT_ACCENT = (0, 230, 255)
_TEXT_BG = (8, 8, 10)

_COACH_VOICE_LINES = {
    "Keep shoulders level (no shrugging).": "Keep shoulders level.",
    "Keep hips level.": "Keep hips level.",
    "Align hips with shoulders.": "Align hips with shoulders.",
    "Stand tall; avoid leaning.": "Stand tall.",
    "Match elbow spacing.": "Match elbow spacing.",
    "Match elbow height.": "Match elbow height.",
    "Turn slightly more or less.": "Adjust your turn.",
    "Square your hips to the pose.": "Square your hips.",
    "Adjust foot width.": "Adjust foot width.",
    "Level your knees/legs.": "Level your knees.",
    "Lift chest and keep waist tight.": "Chest up. Keep waist tight.",
    "Open lats; keep waist tight.": "Open lats. Keep waist tight.",
    "Keep midsection tight and hips set.": "Keep midsection tight.",
    "Balance upper and lower body posture.": "Balance upper and lower body.",
}
_COACH_VOICE_SUCCESS = "Good. Hold it."


def _draw_text_bg(img, x: int, y: int, w: int, h: int, pad: int = 6, alpha: float = 0.55) -> None:
    y0 = max(0, y - h - pad)
    x0 = max(0, x - pad)
    x1 = min(img.shape[1] - 1, x + w + pad)
    y1 = min(img.shape[0] - 1, y + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _TEXT_BG, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _depth_to_vis(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if depth is None or depth.size == 0:
        return None
    d = depth.astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return None
    vals = d[valid]
    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))
    if vmax <= vmin:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
    if vmax <= vmin:
        return None
    scaled = (d - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def _ir_to_vis(ir: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if ir is None or ir.size == 0:
        return None
    arr = ir.astype(np.float32, copy=False)
    if not np.isfinite(arr).any():
        return None
    vmin = float(np.percentile(arr, 5))
    vmax = float(np.percentile(arr, 95))
    if vmax <= vmin:
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
    if vmax <= vmin:
        return None
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _mask_to_vis(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None or mask.size == 0:
        return None
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def _overlay_kinect_previews(
    img: np.ndarray,
    depth: Optional[np.ndarray],
    ir: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    rgb_aligned: Optional[np.ndarray] = None,
    rgb_raw: Optional[np.ndarray] = None,
    show_raw: bool = False,
) -> None:
    h, w = img.shape[:2]
    tiles: list[tuple[str, np.ndarray]] = []
    dvis = _depth_to_vis(depth)
    if dvis is not None:
        tiles.append(("DEPTH", dvis))
    irvis = _ir_to_vis(ir)
    if irvis is not None:
        tiles.append(("IR", irvis))
    if show_raw:
        rvis = _mask_to_vis(rgb_raw)
        if rvis is not None:
            tiles.append(("RGB-RAW", rvis))
    else:
        rvis = _mask_to_vis(rgb_aligned)
        if rvis is not None:
            tiles.append(("RGB-ALIGNED", rvis))
        else:
            mvis = _mask_to_vis(mask)
            if mvis is not None:
                tiles.append(("MASK", mvis))

    if not tiles:
        return
    tiles = tiles[:3]
    ntiles = len(tiles)
    top_reserved = 90
    bottom_pad = 12
    avail_h = max(60, h - top_reserved - bottom_pad)
    gap = 10
    tile_h = min(max(90, int(w * 0.18 * 0.75)), int((avail_h - gap * (ntiles - 1)) / ntiles))
    tile_w = max(140, int(tile_h / 0.75))
    if tile_w > w - 24:
        tile_w = w - 24
        tile_h = max(90, int(tile_w * 0.75))
    x = w - tile_w - 12
    stack_h = (tile_h * ntiles) + (gap * (ntiles - 1))
    y = max(top_reserved, h - stack_h - bottom_pad)

    for label, tile in tiles:
        thumb = cv2.resize(tile, (tile_w, tile_h))
        if thumb.shape[2] == 4:
            thumb = thumb[:, :, :3]
        if y + tile_h > h:
            break
        img[y : y + tile_h, x : x + tile_w] = thumb
        bar_h = 24
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + tile_w, y + bar_h), _TEXT_BG, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        _put_text(img, label, y + 17, scale=0.48, colour=_TEXT_COLOUR, x=x + 8, bg=False)
        y += tile_h + 10


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


def _get_screen_size() -> Optional[Tuple[int, int]]:
    try:
        import tkinter as tk
    except Exception:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w and h:
            return int(w), int(h)
    except Exception:
        return None
    return None


def _get_window_size(window_name: str) -> Optional[Tuple[int, int]]:
    try:
        _x, _y, w, h = cv2.getWindowImageRect(window_name)
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        return None
    return None


def _resize_cover(frame: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame, (0.0, 0.0, 1.0)
    scale = max(target_w / float(w), target_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    x0 = max(0, int((new_w - target_w) / 2))
    y0 = max(0, int((new_h - target_h) / 2))
    cropped = resized[y0 : y0 + target_h, x0 : x0 + target_w]
    # Negative offsets indicate a cropped source region for click mapping.
    return cropped, (-float(x0), -float(y0), scale)


def _map_click_to_source(pt: Tuple[int, int], transform: Tuple[float, float, float]) -> Tuple[int, int]:
    x, y = pt
    x0, y0, scale = transform
    if scale <= 0:
        return x, y
    src_x = int((x - x0) / scale)
    src_y = int((y - y0) / scale)
    return src_x, src_y


def _resize_contain_with_bars(
    frame: np.ndarray, target_w: int, target_h: int
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame, (0.0, 0.0, 1.0)
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x = max(0, (target_w - new_w) // 2)
    off_y = max(0, (target_h - new_h) // 2)
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas, (float(off_x), float(off_y), scale)


def _resize_contain(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame
    scale = min(max_w / float(w), max_h / float(h))
    scale = max(scale, 0.01)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h))


def _load_pose_example(
    pose_key: str,
    cache: Dict[str, Optional[np.ndarray]],
    base_dir: Path,
) -> Optional[np.ndarray]:
    if pose_key in cache:
        return cache[pose_key]
    for ext in (".jpg", ".jpeg", ".png"):
        path = base_dir / f"{pose_key}{ext}"
        if path.exists():
            img = cv2.imread(str(path))
            cache[pose_key] = img
            return img
    cache[pose_key] = None
    return None


def run_live(
    profile: UserProfile,
    camera: int | str = 0,
    width: int = 1280,
    height: int = 720,
    voice: bool = False,
    voice_model: Optional[str] = None,
    coach_voice: Optional[bool] = None,
    tts_backend: Optional[str] = None,
    source_kind: str = "v4l2",
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    mic: Optional[str] = None,
    pose_every_n: Optional[int] = None,
) -> None:
    # Video source (webcam-first). This is intentionally abstracted so we can
    # later drop in Kinect v2 without rewriting the coach logic.
    source_kind_norm = (source_kind or "v4l2").strip().lower()
    cam_index = None if source_kind_norm in ("kinect2", "kinect") else camera
    if isinstance(cam_index, str) and cam_index.isdigit():
        cam_index = int(cam_index)

    def _webcam_candidates(preferred: int | str | None) -> list[int | str]:
        candidates: list[int | str] = []
        seen: set[str] = set()

        def _add(value: int | str | None) -> None:
            if value in (None, ""):
                return
            candidate: int | str
            if isinstance(value, str) and value.isdigit():
                candidate = int(value)
            else:
                candidate = value
            key = f"{type(candidate).__name__}:{candidate}"
            if key in seen:
                return
            seen.add(key)
            candidates.append(candidate)

        _add(preferred)
        if isinstance(preferred, str) and preferred.startswith("/dev/video"):
            suffix = preferred.replace("/dev/video", "")
            if suffix.isdigit():
                _add(int(suffix))
        for node in sorted(Path("/dev").glob("video*"), key=lambda p: str(p)):
            node_text = str(node)
            _add(node_text)
            if node_text.startswith("/dev/video"):
                suffix = node_text.replace("/dev/video", "")
                if suffix.isdigit():
                    _add(int(suffix))
        return candidates

    source = None
    if source_kind_norm == "v4l2":
        preferred = cam_index if cam_index is not None else 0
        candidates = _webcam_candidates(preferred)
        open_errors: list[str] = []
        for candidate in candidates:
            print(f"[bbcoach] Trying webcam candidate: {candidate!r}")
            candidate_source = open_source(
                kind=source_kind_norm,
                cam_index=candidate,
                width=width,
                height=height,
                fps=30,
                depth_align=True,
                depth_scale="meters",
            )
            try:
                candidate_source.start()
                source = candidate_source
                if candidate != preferred:
                    print(f"[bbcoach] Webcam fallback selected: {candidate!r}")
                break
            except (RuntimeError, ValueError) as e:
                open_errors.append(f"{candidate!r}: {e}")
                try:
                    candidate_source.stop()
                except Exception:
                    pass
        if source is None:
            if open_errors:
                print(f"[bbcoach] {open_errors[-1]}")
            vids = sorted(str(p) for p in Path("/dev").glob("video*"))
            if vids:
                print("[bbcoach] Available V4L2 nodes:")
                for v in vids:
                    print(f"  - {v}")
                print("[bbcoach] Try: python -m bbcoach live --profile <name> --cam-index /dev/video2")
            return
    else:
        source = open_source(
            kind=source_kind_norm,
            cam_index=cam_index,
            width=width,
            height=height,
            fps=30,
            depth_align=True,
            depth_scale="meters",
        )
        try:
            source.start()
        except (RuntimeError, ValueError) as e:
            print(f"[bbcoach] {e}")
            return

    if depth_min is not None or depth_max is not None:
        if getattr(profile, "video", None) is None:
            profile.video = VideoConfig()
        if depth_min is not None:
            profile.video.depth_min_m = depth_min
        if depth_max is not None:
            profile.video.depth_max_m = depth_max

    try:
        pose = PoseBackend()
    except ModuleNotFoundError:
        print("[bbcoach] Missing dependency: mediapipe")
        print("[bbcoach] Activate your venv and install requirements:")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        source.stop()
        return
    except RuntimeError as e:
        print(f"[bbcoach] {e}")
        source.stop()
        return
    sessions = SessionStore.default()
    voice_listener: Optional[VoiceCommandListener] = None
    voice_enabled = False
    voice_last_cmd = ""
    voice_error = ""
    coach_voice_requested = coach_voice if coach_voice is not None else getattr(profile, "coach_voice", False)
    coach_voice_backend = (tts_backend or getattr(profile, "tts_backend", "piper_bin") or "piper_bin").lower()
    if coach_voice_backend == "auto":
        coach_voice_backend = "piper_bin"
    tts_speaker: Optional[TTSSpeaker] = None
    coach_voice_enabled = False
    coach_voice_error = ""
    coach_voice_warning = ""
    coach_voice_warning_until = 0.0
    last_spoken_line = ""
    last_speak_ts = 0.0
    score_window: deque[float] = deque(maxlen=8)
    last_advice_active = False
    template_cleared_until = 0.0

    def _missing_tts_assets() -> list[str]:
        root = Path(__file__).resolve().parent.parent
        assets = [
            root / "vendor" / "piper" / "linux_x86_64" / "piper",
            root / "data" / "tts" / "en_GB-alan-medium" / "voice.onnx",
            root / "data" / "tts" / "en_GB-alan-medium" / "voice.onnx.json",
        ]
        return [str(p) for p in assets if not p.exists()]

    def _set_coach_voice(enabled: bool) -> None:
        nonlocal coach_voice_enabled, coach_voice_error, coach_voice_warning, coach_voice_warning_until, tts_speaker
        if enabled:
            if missing_tts and coach_voice_backend == "piper_bin":
                coach_voice_error = "missing Alan assets"
                coach_voice_warning = "Coach voice disabled: missing vendored Alan assets: " + ", ".join(missing_tts)
                coach_voice_warning_until = time.time() + 6.0
                coach_voice_enabled = False
                return
            if tts_speaker is None:
                try:
                    tts_speaker = TTSSpeaker(backend=coach_voice_backend)
                    tts_speaker.start()
                except Exception as exc:
                    coach_voice_error = str(exc)
                    coach_voice_enabled = False
                    tts_speaker = None
                    return
            coach_voice_error = ""
            coach_voice_enabled = True
        else:
            coach_voice_enabled = False
            if tts_speaker is not None:
                tts_speaker.stop()
                tts_speaker = None

    def _vosk_model_valid(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        has_model_conf = (path / "conf" / "model.conf").exists() or (path / "model.conf").exists()
        if not has_model_conf:
            return False
        if not (path / "am" / "final.mdl").exists():
            return False
        return True

    def _resolve_voice_model(path_arg: Optional[str]) -> Optional[Path]:
        candidates = []
        if path_arg:
            candidates.append(Path(path_arg))
        root = Path(__file__).resolve().parent.parent
        candidates.append(root / "models" / "vosk")
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                if _vosk_model_valid(cand):
                    return cand
        return None

    def _init_voice_listener() -> None:
        nonlocal voice_listener, voice_enabled, voice_error
        if voice_listener is not None:
            return
        model_path = _resolve_voice_model(voice_model)
        if model_path is None:
            root = Path(__file__).resolve().parent.parent
            cand = Path(voice_model) if voice_model else (root / "models" / "vosk")
            if cand.exists():
                voice_error = "invalid vosk model"
                print("[bbcoach] Vosk model folder exists but is missing required files (conf/model.conf or model.conf, am/final.mdl).")
                print("[bbcoach] Reinstall it once: scripts/vendor_vosk_model_small_en_us.sh")
            else:
                voice_error = "missing vosk model"
                print("[bbcoach] Voice enabled but no Vosk model found.")
                print("[bbcoach] Install it once: scripts/vendor_vosk_model_small_en_us.sh")
            return
        try:
            mic_prefer = mic or getattr(getattr(profile, "audio", None), "mic_prefer", None)
            mic_fallback = getattr(getattr(profile, "audio", None), "mic_fallback", None)
            if not mic_prefer and source_kind_norm == "kinect2":
                mic_prefer = "Xbox NUI Sensor"
            voice_listener = VoiceCommandListener(
                VoiceCommandConfig(
                    model_path=str(model_path),
                    mic_prefer=mic_prefer,
                    mic_fallback=mic_fallback,
                    source_kind=source_kind_norm,
                )
            )
            voice_listener.start()
            voice_enabled = True
            voice_error = ""
            print("[bbcoach] Voice commands enabled (say: 'next pose' / 'siguiente pose').")
        except Exception as exc:
            raw_msg = str(exc)
            if "Missing optional dependencies for voice commands" in raw_msg:
                voice_error = "voice deps missing"
                print("[bbcoach] Voice deps missing. Install base deps once:")
                print("[bbcoach]   pip install -r requirements.txt")
            else:
                voice_error = raw_msg
            print(f"[bbcoach] Voice init failed: {raw_msg}")

    missing_tts = _missing_tts_assets()
    if not missing_tts:
        coach_voice_backend = "piper_bin"
    if missing_tts:
        print(
            "[bbcoach] Missing vendored TTS assets. Run: "
            "scripts/vendor_piper_linux_x86_64.sh && scripts/vendor_voice_alan.sh"
        )

    if voice:
        _init_voice_listener()

    if coach_voice_requested:
        _set_coach_voice(True)

    screen_size = _get_screen_size()
    window_name = "BB Coach (Webcam)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    mouse = {"click": None}

    def _on_mouse(event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)

    cv2.setMouseCallback(window_name, _on_mouse)
    cv2.resizeWindow(window_name, width, height)
    # Force a sane initial window size (maximized windowed).
    if screen_size:
        window_size = screen_size
        cv2.resizeWindow(window_name, screen_size[0], screen_size[1])
        cv2.moveWindow(window_name, 0, 0)
    else:
        window_size = (width, height)
    # Ensure the window is actually realized before first size query.
    cv2.waitKey(1)

    # Categories may contain multiple selections; pick the first for live routine.
    cat = profile.selected_categories[0] if profile.selected_categories else "Mens Physique"
    routine = routine_for(cat, profile.federation)
    app_state = get_global_app_state()
    specs_library = load_federation_specs()
    selected_div_refs: list[SelectedDivisionRef] = list(app_state.state.selected_divisions)
    selected_pose_checklist: list[PoseChecklistItem] = build_pose_checklist(specs_library, selected_div_refs)
    if selected_pose_checklist:
        routine = [item.pose_key or "mp_front" for item in selected_pose_checklist]
    selected_refs_hash = tuple(ref.key() for ref in selected_div_refs)
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
    guide_use_template = False
    active_tab = "posing"
    metrics_tab = MetricsTab(profile.name)
    fullscreen = False
    _window_state_applied = False
    display_transform: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    should_exit = False
    pose_example_until = time.time() + 5.0
    example_cache: Dict[str, Optional[np.ndarray]] = {}
    example_dir = Path(__file__).resolve().parent.parent / "assets" / "pose_examples"
    pose_timer_enabled = True
    pose_timer_seconds = 20
    pose_timer_started_at = time.time()
    pose_quality_scores: Dict[str, float] = {}

    def _current_pose_item() -> Optional[PoseChecklistItem]:
        if not selected_pose_checklist:
            return None
        if not routine:
            return None
        idx = max(0, min(len(selected_pose_checklist) - 1, pose_i))
        return selected_pose_checklist[idx]

    def _quality_key(pose_name: str, item: Optional[PoseChecklistItem]) -> str:
        if item is None:
            return f"{profile.federation}:{cat}:{pose_name}"
        class_part = item.class_id or "open"
        return f"{item.federation_id}:{item.division_id}:{class_part}:{pose_name}"

    def _sync_state_metadata() -> None:
        app_state.set_session_metadata(
            selected_divisions=[ref.to_dict() for ref in selected_div_refs],
            selected_division_labels=selected_division_labels(specs_library, selected_div_refs),
            pose_quality_scores=dict(pose_quality_scores),
        )

    def _refresh_selected_divisions() -> None:
        nonlocal selected_div_refs, selected_pose_checklist, routine, pose_i, pose_switch_frame
        nonlocal selected_refs_hash, pose_example_until, pose_timer_started_at
        latest_refs = list(app_state.state.selected_divisions)
        latest_hash = tuple(ref.key() for ref in latest_refs)
        if latest_hash == selected_refs_hash:
            return
        selected_div_refs = latest_refs
        selected_refs_hash = latest_hash
        selected_pose_checklist = build_pose_checklist(specs_library, selected_div_refs)
        if selected_pose_checklist:
            routine = [item.pose_key or "mp_front" for item in selected_pose_checklist]
            pose_i = 0
            pose_switch_frame = frame_idx
            pose_example_until = time.time() + 5.0
            pose_timer_started_at = time.time()
        else:
            routine = routine_for(cat, profile.federation)
            pose_i = min(pose_i, max(0, len(routine) - 1))
        _sync_state_metadata()

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
        nonlocal pose_i, pose_switch_frame, pose_example_until, pose_timer_started_at
        if not routine:
            return
        try:
            _finalize_pose(routine[pose_i])
        except Exception as exc:
            print(f"[bbcoach] Pose finalize error: {exc}")
        try:
            pose_i = new_idx % len(routine)
        except Exception:
            pose_i = 0
        pose_switch_frame = frame_idx
        motion_buf.clear()
        pose_example_until = time.time() + 5.0
        pose_timer_started_at = time.time()

    def _set_category(new_cat: str) -> None:
        nonlocal cat, routine, pose_i, show_category_menu, pose_switch_frame, pose_example_until
        if routine:
            try:
                _finalize_pose(routine[pose_i])
            except Exception as exc:
                print(f"[bbcoach] Pose finalize error: {exc}")
        cat = new_cat
        if cat not in profile.selected_categories:
            profile.selected_categories.insert(0, cat)
        profile.plan.target_categories = list(profile.selected_categories)
        if selected_pose_checklist:
            routine = [item.pose_key or "mp_front" for item in selected_pose_checklist]
        else:
            routine = routine_for(cat, profile.federation)
        pose_i = 0
        pose_switch_frame = frame_idx
        show_category_menu = False
        pose_example_until = time.time() + 5.0

    def _apply_metrics_overrides() -> None:
        nonlocal routine
        fed_override = metrics_tab.pop_federation_override() if hasattr(metrics_tab, "pop_federation_override") else None
        if fed_override and fed_override in RULES and fed_override != profile.federation:
            profile.federation = fed_override
            if selected_pose_checklist:
                routine = [item.pose_key or "mp_front" for item in selected_pose_checklist]
            else:
                routine = routine_for(cat, profile.federation)
            _set_pose_index(0)

        cat_override = metrics_tab.pop_category_override() if hasattr(metrics_tab, "pop_category_override") else None
        if cat_override and cat_override != cat:
            _set_category(cat_override)

    def _apply_voice_command(cmd: str) -> None:
        nonlocal voice_last_cmd, active_tab, should_exit
        voice_last_cmd = cmd
        if cmd == "next_pose":
            _set_pose_index(pose_i + 1)
            return
        if cmd == "prev_pose":
            _set_pose_index(pose_i - 1)
            return
        if cmd == "open_metrics":
            active_tab = "metrics"
            return
        if cmd == "open_posing":
            active_tab = "posing"
            return
        if cmd in ("exit_app", "exit"):
            should_exit = True
            return
        metrics_tab.handle_voice_command(cmd)

    def _apply_window_state() -> None:
        if fullscreen:
            if screen_size:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, screen_size[0], screen_size[1])
                cv2.moveWindow(window_name, 0, 0)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            if window_size:
                cv2.resizeWindow(window_name, window_size[0], window_size[1])

    def _build_tab_buttons(out: np.ndarray) -> list[UIButton]:
        buttons: list[UIButton] = []
        btn_h = 24
        btn_gap = 8
        btn_y = 8
        btn_x = out.shape[1] - 12

        def add_tab(key: str, label: str, active: bool = False) -> None:
            nonlocal btn_x
            (tw, _), _meta = cv2.getTextSize(label, _TEXT_FONT, 0.48, 1)
            w = tw + 16
            btn_x -= w
            buttons.append(UIButton(key=key, label=label, rect=(btn_x, btn_y, w, btn_h), active=active))
            btn_x -= btn_gap

        add_tab("tab:exit", "EXIT", active=False)
        add_tab("tab:metrics", "METRICS", active=(active_tab == "metrics"))
        add_tab("tab:posing", "POSING", active=(active_tab == "posing"))
        buttons.reverse()
        return buttons

    def _handle_tab_button(key: str) -> bool:
        nonlocal active_tab, should_exit
        if key == "tab:exit":
            should_exit = True
            return True
        if key == "tab:metrics":
            active_tab = "metrics"
            return True
        if key == "tab:posing":
            active_tab = "posing"
            return True
        return False

    def _window_closed() -> bool:
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            return True

    def _build_snapshot_payload(
        pose_key: str,
        score: float,
        props_obj,
        pose_label: Optional[str] = None,
        manual_quality_score: Optional[float] = None,
    ) -> dict:
        return {
            "profile": profile.name,
            "federation": profile.federation,
            "category": cat,
            "pose": pose_key,
            "pose_label": pose_label or pose_key,
            "pose_score": float(score),
            "manual_pose_quality_score": manual_quality_score,
            "pose_features": ps.per_feature,
            "proportions": asdict(props_obj) if props_obj is not None else None,
            "competition_date": profile.plan.competition_date,
            "first_timers": profile.first_timers,
            "prep_mode": profile.prep.mode,
            "selected_divisions": [ref.to_dict() for ref in selected_div_refs],
            "selected_division_labels": selected_division_labels(specs_library, selected_div_refs),
            "pose_quality_scores": dict(pose_quality_scores),
        }

    def _cleanup_capture(entry: dict) -> None:
        for key in (
            "full_path",
            "cutout_path",
            "story_path",
            "depth_path",
            "ir_path",
            "rgb_aligned_path",
            "rgb_raw_path",
        ):
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
        manual_quality_score: Optional[float] = None,
        depth: Optional[np.ndarray] = None,
        ir: Optional[np.ndarray] = None,
        rgb_aligned: Optional[np.ndarray] = None,
        rgb_raw: Optional[np.ndarray] = None,
    ) -> dict:
        payload = _build_snapshot_payload(
            pose_key,
            score,
            props_obj,
            pose_label=pose_display,
            manual_quality_score=manual_quality_score,
        )
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

        depth_vis = _depth_to_vis(depth)
        if depth_vis is not None:
            entry["depth_path"] = sessions.save_capture(profile.name, payload, depth_vis, variant="depth")
        ir_vis = _ir_to_vis(ir)
        if ir_vis is not None:
            entry["ir_path"] = sessions.save_capture(profile.name, payload, ir_vis, variant="ir")
        if rgb_aligned is not None:
            entry["rgb_aligned_path"] = sessions.save_capture(
                profile.name, payload, rgb_aligned, variant="rgb_aligned"
            )
        if rgb_raw is not None:
            entry["rgb_raw_path"] = sessions.save_capture(profile.name, payload, rgb_raw, variant="rgb_raw")
        return entry

    def _maybe_flip_packet(packet: dict) -> dict:
        rgb = packet.get("rgb")
        rgb_aligned = packet.get("rgb_aligned")
        rgb_raw = packet.get("rgb_raw")
        depth = packet.get("depth")
        ir = packet.get("ir")
        if rgb is not None:
            rgb = cv2.flip(rgb, 1)
        if rgb_aligned is not None:
            rgb_aligned = cv2.flip(rgb_aligned, 1)
        if rgb_raw is not None:
            rgb_raw = cv2.flip(rgb_raw, 1)
        if depth is not None:
            depth = cv2.flip(depth, 1)
        if ir is not None:
            ir = cv2.flip(ir, 1)
        packet["rgb"], packet["rgb_aligned"], packet["rgb_raw"] = rgb, rgb_aligned, rgb_raw
        packet["depth"], packet["ir"] = depth, ir
        return packet

    _sync_state_metadata()
    print("Live coach started. Press Q to quit.")

    pose_every = pose_every_n if pose_every_n and pose_every_n > 0 else (3 if source_kind_norm == "kinect2" else 1)
    seg_every = 2 if source_kind_norm == "kinect2" else 1
    empty_frame_limit = 45 if source_kind_norm == "kinect2" else 10
    empty_frames = 0
    last_pose_res = None
    last_seg_mask = None
    depth_stats_printed = False

    while True:
        packet = _maybe_flip_packet(source.read())
        main_is_aligned = False
        frame = packet.get("rgb")
        if source_kind_norm == "kinect2":
            aligned_rgb = packet.get("rgb_aligned")
            if aligned_rgb is not None:
                frame = aligned_rgb
                main_is_aligned = True
        depth = packet.get("depth")
        if source_kind_norm == "kinect2" and frame is not None and not main_is_aligned:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
        if depth is not None and not depth_stats_printed:
            valid = depth[np.isfinite(depth) & (depth > 0)]
            if valid.size:
                print(
                    f"[bbcoach] depth(frame): shape={tuple(depth.shape)} dtype={depth.dtype} "
                    f"min={float(valid.min()):.2f} max={float(valid.max()):.2f}"
                )
                depth_stats_printed = True
        if frame is None:
            empty_frames += 1
            if empty_frames > empty_frame_limit:
                break
            continue
        empty_frames = 0
        _refresh_selected_divisions()

        frame_idx += 1
        pose_frame = frame
        if source_kind_norm == "kinect2":
            h, w = frame.shape[:2]
            pose_frame = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)))
        if (frame_idx % pose_every) == 0 or last_pose_res is None:
            res = pose.process_bgr(pose_frame)
            if pose_frame is not frame and res.mask is not None:
                res.mask = cv2.resize(res.mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            last_pose_res = res
        else:
            res = last_pose_res

        if depth is not None:
            depth_min_val = depth_min
            depth_max_val = depth_max
            if depth_min_val is None:
                depth_min_val = getattr(getattr(profile, "video", None), "depth_min_m", None)
            if depth_max_val is None:
                depth_max_val = getattr(getattr(profile, "video", None), "depth_max_m", None)
            pose_mask = res.mask
            do_seg = (frame_idx % seg_every) == 0 or last_seg_mask is None
            if do_seg:
                seg = segment_person_depth(
                    depth_m=depth,
                    rgb_bgr=frame,
                    landmarks=res.landmarks if res else None,
                    depth_min=float(depth_min_val) if depth_min_val is not None else 0.4,
                    depth_max=float(depth_max_val) if depth_max_val is not None else 4.0,
                )
                last_seg_mask = seg.get("mask")
            if last_seg_mask is not None:
                mask = last_seg_mask
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                if pose_mask is not None and pose_mask.shape[:2] == mask.shape[:2]:
                    overlap = float(np.count_nonzero((pose_mask > 0) & (mask > 0)))
                    pose_area = float(np.count_nonzero(pose_mask > 0)) or 1.0
                    if (overlap / pose_area) < 0.15:
                        res.mask = pose_mask
                    else:
                        res.mask = mask
                else:
                    res.mask = mask

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

        if not routine:
            routine = ["mp_front"]
            pose_i = 0
        pose_key = routine[pose_i]
        current_pose_item = _current_pose_item()
        pose_def = POSES.get(pose_key, POSES["mp_front"])
        pose_display_name = current_pose_item.pose_name if current_pose_item is not None else pose_def.display
        pose_is_mapped = current_pose_item is None or current_pose_item.pose_key is not None
        pose_can_score = pose_key in POSES
        pose_is_proxy_score = (not pose_is_mapped) and pose_can_score
        pose_quality_key = _quality_key(pose_display_name, current_pose_item)
        pose_quality_value = pose_quality_scores.get(pose_quality_key)
        pose_timer_remaining: Optional[int] = None
        if pose_timer_enabled and active_tab == "posing":
            elapsed = max(0.0, time.time() - pose_timer_started_at)
            remaining = int(np.ceil(max(0.0, float(pose_timer_seconds) - elapsed)))
            pose_timer_remaining = remaining
            if remaining <= 0 and routine:
                _set_pose_index(pose_i + 1)
                pose_key = routine[pose_i]
                current_pose_item = _current_pose_item()
                pose_def = POSES.get(pose_key, POSES["mp_front"])
                pose_display_name = current_pose_item.pose_name if current_pose_item is not None else pose_def.display
                pose_is_mapped = current_pose_item is None or current_pose_item.pose_key is not None
                pose_can_score = pose_key in POSES
                pose_is_proxy_score = (not pose_is_mapped) and pose_can_score
                pose_quality_key = _quality_key(pose_display_name, current_pose_item)
                pose_quality_value = pose_quality_scores.get(pose_quality_key)
                pose_timer_remaining = pose_timer_seconds

        if active_tab == "metrics":
            target_size: Optional[Tuple[int, int]] = None
            if fullscreen and screen_size:
                target_size = screen_size
            else:
                win_size = _get_window_size(window_name)
                if win_size and win_size[0] >= 800 and win_size[1] >= 600:
                    window_size = win_size
                    target_size = win_size
                elif window_size:
                    target_size = window_size
                elif screen_size:
                    target_size = screen_size

            out = metrics_tab.render(
                frame=frame,
                mask=res.mask,
                props=props,
                profile=profile,
                category=cat,
                pose_key=pose_key,
                landmarks=res.landmarks,
                depth=depth,
                intrinsics=(packet.get("meta") or {}).get("intrinsics"),
                depth_aligned=(packet.get("meta") or {}).get("depth_aligned"),
                depth_units=(packet.get("meta") or {}).get("depth_units"),
                target_size=target_size,
            )

            tab_buttons = _build_tab_buttons(out)
            click = mouse.get("click")
            if click:
                tab_hit = False
                for btn in tab_buttons:
                    if _in_rect(click, btn.rect):
                        tab_hit = _handle_tab_button(btn.key)
                        break
                if not tab_hit:
                    metrics_tab.handle_click(click)
                    _apply_metrics_overrides()
                mouse["click"] = None

            for btn in tab_buttons:
                _draw_button(out, btn)

            if not _window_state_applied:
                _apply_window_state()
                _window_state_applied = True

            if source_kind_norm == "kinect2":
                _overlay_kinect_previews(
                    out,
                    depth,
                    packet.get("ir"),
                    res.mask if res else None,
                    packet.get("rgb_aligned"),
                    packet.get("rgb_raw"),
                    show_raw=main_is_aligned,
                )
            cv2.imshow(window_name, out)

            countdown = metrics_tab.pop_scan_countdown()
            if countdown is not None and coach_voice_enabled and tts_speaker is not None:
                tts_speaker.say(str(countdown))
            scan4_tts_reset = metrics_tab.pop_scan4_tts_reset() if hasattr(metrics_tab, "pop_scan4_tts_reset") else False
            if scan4_tts_reset and tts_speaker is not None:
                tts_speaker.clear_pending()
            scan4_announce = metrics_tab.pop_scan4_announce() if hasattr(metrics_tab, "pop_scan4_announce") else None
            if scan4_announce:
                if tts_speaker is None:
                    try:
                        tts_speaker = TTSSpeaker(backend=coach_voice_backend)
                        tts_speaker.start()
                    except Exception:
                        tts_speaker = None
                if tts_speaker is not None:
                    tts_speaker.say_priority(scan4_announce, clear_pending=True)

            key = cv2.waitKey(1) & 0xFF
            if _window_closed():
                should_exit = True
                break
            if key in (ord('1'),):
                active_tab = "posing"
            if key in (ord('2'),):
                active_tab = "metrics"
            if key == 9:
                active_tab = "metrics" if active_tab == "posing" else "posing"
            if key in (ord('z'), ord('Z')):
                fullscreen = not fullscreen
                _apply_window_state()

            metrics_tab.handle_key(key)
            _apply_metrics_overrides()

            if voice_listener is not None:
                if voice_listener.error() and not voice_error:
                    voice_error = voice_listener.error() or "voice error"
                    voice_enabled = False
                if voice_enabled:
                    cmd = voice_listener.pop_command()
                    if cmd:
                        _apply_voice_command(cmd)

            if should_exit:
                break
            continue

        # Score pose
        tpl = profile.templates.get(pose_key) if pose_is_mapped else None
        tpl_feats = tpl.get("features") if tpl else None
        tpl_landmarks = _deserialize_landmarks(tpl.get("landmarks")) if tpl else None
        if pose_can_score:
            ps = score_pose(
                res.landmarks,
                pose_def.target,
                pose_def.tolerance,
                template_override=tpl_feats,
                weights=pose_def.weights,
                props=props,
            )
        else:
            ps = PoseScore(
                score_0_100=0.0,
                per_feature={},
                ok_flags={},
                advice=["Manual quality score required for this pose/division."],
            )

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
            if guide_use_template and tpl_landmarks:
                guide_landmarks = tpl_landmarks
            else:
                guide_landmarks = POSE_GUIDES.get(pose_key)

        out = display_frame
        if guide_landmarks:
            out = draw_pose_guide(out, guide_landmarks)
        out = draw_pose_overlay(out, res.landmarks, joint_ok=joint_ok, line_ok=line_ok)
        if res.mask is not None and not cutout_enabled:
            out = draw_mask_outline(out, res.mask)

        if window_size is None:
            window_size = (out.shape[1], out.shape[0])

        now = time.time()
        show_example = now < pose_example_until
        example_img = None
        if show_example:
            example_img = _load_pose_example(pose_key, example_cache, example_dir)

        stable = False
        if len(motion_buf) >= auto_cfg.stable_frames:
            stable = float(np.mean(motion_buf)) < auto_cfg.motion_threshold

        score_window.append(ps.score_0_100)
        pose_stable_for_voice = False
        if len(score_window) == score_window.maxlen:
            avg_score = float(np.mean(score_window))
            pose_stable_for_voice = abs(ps.score_0_100 - avg_score) < 1.0

        speak_line = ""
        if ps.advice:
            top_advice = ps.advice[0]
            speak_line = _COACH_VOICE_LINES.get(top_advice, top_advice)

        advice_active = bool(ps.advice) and (ps.score_0_100 < 95.0 or any(not ok for ok in ps.ok_flags.values()))
        scan4_active = metrics_tab.is_scan4_active() if hasattr(metrics_tab, "is_scan4_active") else False

        if coach_voice_enabled and tts_speaker is not None and pose_stable_for_voice and not show_example and not scan4_active:
            now_ts = time.time()
            if advice_active and speak_line:
                adv_changed = speak_line != last_spoken_line
                time_since = now_ts - last_speak_ts
                if adv_changed or time_since > 6.0:
                    tts_speaker.say(speak_line)
                    last_spoken_line = speak_line
                    last_speak_ts = now_ts
                last_advice_active = True
            else:
                if last_advice_active and (now_ts - last_speak_ts) > 1.0:
                    tts_speaker.say(_COACH_VOICE_SUCCESS)
                    last_spoken_line = _COACH_VOICE_SUCCESS
                    last_speak_ts = now_ts
                last_advice_active = False
        elif scan4_active:
            last_advice_active = False
        elif not coach_voice_enabled:
            last_advice_active = advice_active

        can_capture = (
            auto_cfg.enabled
            and stable
            and bool(res.landmarks)
            and pose_is_mapped
            and ps.score_0_100 >= auto_cfg.min_score
            and (frame_idx - last_capture_frame) >= auto_cfg.cooldown_frames
            and (frame_idx - pose_switch_frame) >= auto_cfg.settle_frames
            and not show_example
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
                pose_display_name,
                manual_quality_score=pose_quality_value,
                depth=depth,
                ir=packet.get("ir"),
                rgb_aligned=packet.get("rgb_aligned"),
                rgb_raw=packet.get("rgb_raw"),
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

        target_size: Optional[Tuple[int, int]] = None
        if fullscreen and screen_size:
            target_size = screen_size
        else:
            win_size = _get_window_size(window_name)
            # Ignore bogus/tiny window rects that some OpenCV backends return at startup
            if win_size and win_size[0] >= 800 and win_size[1] >= 600:
                window_size = win_size
                target_size = win_size
            elif window_size:
                target_size = window_size
            elif screen_size:
                target_size = screen_size

        if target_size:
            display_frame, display_transform = _resize_cover(out, target_size[0], target_size[1])
        else:
            display_transform = (0.0, 0.0, 1.0)

        out = display_frame

        if not _window_state_applied:
            _apply_window_state()
            _window_state_applied = True

        if show_example and example_img is not None:
            max_w = int(out.shape[1] * 0.36)
            max_h = int(out.shape[0] * 0.55)
            ex = _resize_contain(example_img, max_w, max_h)
            ex_h, ex_w = ex.shape[:2]
            panel_pad = 10
            label_h = 26
            panel_w = ex_w + panel_pad * 2
            panel_h = ex_h + panel_pad * 2 + label_h
            panel_x1 = out.shape[1] - 18
            panel_x0 = max(18, panel_x1 - panel_w)
            panel_y0 = 18
            panel_y1 = min(out.shape[0] - 18, panel_y0 + panel_h)
            overlay = out.copy()
            cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), _TEXT_BG, -1)
            cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
            label = f"EXAMPLE {int(max(0, pose_example_until - now))}s"
            cv2.putText(
                out,
                label,
                (panel_x0 + panel_pad, panel_y0 + 20),
                _TEXT_FONT,
                0.5,
                _TEXT_ACCENT,
                1,
                cv2.LINE_AA,
            )
            img_y0 = panel_y0 + label_h + panel_pad
            img_x0 = panel_x0 + panel_pad
            out[img_y0 : img_y0 + ex_h, img_x0 : img_x0 + ex_w] = ex

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

        y = 28
        ft = "YES" if profile.first_timers else "NO"
        _put_text(
            out,
            f"FEDERATION {fed_rules.display_name} | FIRST TIMERS {ft}",
            y,
            scale=0.5,
            thickness=1,
        )
        y += 22
        if current_pose_item is not None:
            _put_text(
                out,
                f"POSE CHECKLIST | {current_pose_item.division_label.upper()} | POSE {pose_display_name.upper()} ({pose_i+1}/{len(routine)})",
                y,
                scale=0.47,
                thickness=1,
            )
        else:
            _put_text(
                out,
                f"CATEGORY {cat.upper()} | POSE {pose_display_name.upper()} ({pose_i+1}/{len(routine)})",
                y,
                scale=0.52,
                thickness=1,
            )
        y += 24
        if pose_is_mapped:
            _put_text(out, f"SCORE {ps.score_0_100:.1f}", y, scale=0.7, colour=_TEXT_ACCENT, thickness=2)
        elif pose_is_proxy_score:
            _put_text(out, f"GUIDE SCORE {ps.score_0_100:.1f}", y, scale=0.62, colour=_TEXT_ACCENT, thickness=2)
        else:
            _put_text(out, "MANUAL QUALITY MODE", y, scale=0.62, colour=_TEXT_ACCENT, thickness=2)
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
        timer_status = "ON" if pose_timer_enabled else "OFF"
        timer_remaining_txt = str(pose_timer_remaining) if pose_timer_remaining is not None else "-"
        _put_text(
            out,
            f"POSE TIMER {timer_status} | {timer_remaining_txt}S",
            y,
            scale=0.48,
            colour=(200, 200, 200),
            thickness=1,
        )
        y += 20
        quality_txt = "-" if pose_quality_value is None else f"{pose_quality_value:.1f}/5"
        _put_text(
            out,
            f"POSE QUALITY {quality_txt} | PRESS 3-7 TO SCORE",
            y,
            scale=0.48,
            colour=(200, 200, 200),
            thickness=1,
        )
        y += 20
        voice_status = "On" if voice_enabled else "Off"
        voice_line = f"Mic: {voice_status}"
        if voice_last_cmd:
            voice_line += f" | Last: {voice_last_cmd}"
        if voice_listener is not None:
            heard = voice_listener.last_text().strip()
            if heard:
                heard_short = heard if len(heard) <= 24 else ("..." + heard[-24:])
                voice_line += f" | Heard: {heard_short}"
        if voice_error:
            voice_line += " | Error"
        _put_text(out, voice_line.upper(), y, scale=0.48, colour=(200, 200, 200), thickness=1)
        y += 20
        if voice_error and not voice_enabled:
            if voice_error == "missing vosk model":
                msg = "MIC ERROR: install Vosk model in ./models/vosk or pass --voice-model"
            else:
                msg = f"MIC ERROR: {voice_error}"
            _put_text(
                out,
                msg.upper(),
                y,
                scale=0.46,
                colour=(255, 190, 120),
                thickness=1,
            )
            y += 20
        coach_status = "Error" if coach_voice_error else ("On" if coach_voice_enabled else "Off")
        coach_line = f"Coach Voice: {coach_status}"
        _put_text(out, coach_line.upper(), y, scale=0.48, colour=(200, 200, 200), thickness=1)
        y += 20
        tts_label = "TTS: disabled"
        if tts_speaker is not None:
            tts_label = f"TTS: {tts_speaker.backend_label()}"
        _put_text(out, tts_label.upper(), y, scale=0.48, colour=(200, 200, 200), thickness=1)
        y += 20
        if tts_speaker is not None:
            warn = tts_speaker.pop_warning()
            if warn:
                coach_voice_warning = warn
                coach_voice_warning_until = time.time() + 6.0
        if coach_voice_warning and time.time() < coach_voice_warning_until:
            _put_text(out, coach_voice_warning.upper(), y, scale=0.48, colour=(255, 190, 120), thickness=1)
            y += 20
        if template_cleared_until and time.time() < template_cleared_until:
            _put_text(out, "TEMPLATE CLEARED", y, scale=0.5, colour=(180, 230, 180), thickness=1)
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

        # Guidance lines: 1 static line + dynamic advice when needed.
        guide_lines: list[str] = []
        if pose_def.guidance:
            guide_lines.append(pose_def.guidance[0])

        needs_advice = ps.score_0_100 < 95.0 or any(not ok for ok in ps.ok_flags.values())
        if needs_advice and ps.advice:
            guide_lines.extend(ps.advice[:2])

        for g in guide_lines:
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
            timer_label = "On" if pose_timer_enabled else "Off"

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
            add("timer", f"TIMER {timer_label.upper()}", active=pose_timer_enabled)

            btn_y2 = btn_y - (btn_h + 8)
            btn_x2 = 12

            def add_row2(key: str, label: str, active: bool = False) -> None:
                nonlocal btn_x2
                (tw, _), _meta = cv2.getTextSize(label, _TEXT_FONT, 0.5, 1)
                w = tw + 18
                buttons.append(UIButton(key=key, label=label, rect=(btn_x2, btn_y2, w, btn_h), active=active))
                btn_x2 += w + btn_gap

            if selected_pose_checklist:
                add_row2("category_menu", f"CHECKLIST {len(selected_div_refs)} DIVS", active=False)
            else:
                add_row2("category_menu", f"CATEGORY {cat.upper()}", active=show_category_menu)
            add_row2("federation", f"FED {profile.federation}", active=False)
            guide_src_label = "TEMPLATE" if guide_use_template else "OFFICIAL"
            add_row2("guide_source", f"GUIDE SOURCE {guide_src_label}", active=guide_use_template)
            coach_label = "Error" if coach_voice_error else ("On" if coach_voice_enabled else "Off")
            add_row2("coach_voice", f"COACH VOICE {coach_label.upper()}", active=coach_voice_enabled)
            add_row2("info", "INFO", active=show_info)
            voice_label = "On" if voice_enabled else "Off"
            if voice_error:
                voice_label = "Err"
            add_row2("voice", f"MIC {voice_label.upper()}", active=voice_enabled)
            full_label = "On" if fullscreen else "Off"
            add_row2("fullscreen", f"FULL {full_label.upper()}", active=fullscreen)
            add_row2("exit", "EXIT", active=False)

            if show_category_menu and not selected_pose_checklist:
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
        tab_buttons = _build_tab_buttons(out)
        click = mouse.get("click")
        if click:
            tab_hit = False
            for btn in tab_buttons:
                if _in_rect(click, btn.rect):
                    tab_hit = _handle_tab_button(btn.key)
                    break
            if not tab_hit:
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
                        elif btn.key == "timer":
                            pose_timer_enabled = not pose_timer_enabled
                            pose_timer_started_at = time.time()
                        elif btn.key == "category_menu":
                            if not selected_pose_checklist:
                                show_category_menu = not show_category_menu
                        elif btn.key == "info":
                            show_info = not show_info
                        elif btn.key == "federation":
                            profile.federation = cycle_federation(profile.federation)
                            if selected_pose_checklist:
                                routine = [item.pose_key or "mp_front" for item in selected_pose_checklist]
                            else:
                                routine = routine_for(cat, profile.federation)
                            _set_pose_index(0)
                        elif btn.key == "guide_source":
                            guide_use_template = not guide_use_template
                        elif btn.key == "coach_voice":
                            _set_coach_voice(not coach_voice_enabled)
                        elif btn.key == "voice":
                            if voice_listener is None:
                                _init_voice_listener()
                            if voice_listener is not None:
                                voice_enabled = not voice_enabled
                        elif btn.key == "fullscreen":
                            fullscreen = not fullscreen
                            _apply_window_state()
                        elif btn.key == "exit":
                            should_exit = True
                        elif btn.key.startswith("cat:"):
                            _set_category(btn.key.split(":", 1)[1])
                        break
            mouse["click"] = None
            buttons = build_buttons()

        for btn in buttons:
            _draw_button(out, btn)
        for btn in tab_buttons:
            _draw_button(out, btn)

        if source_kind_norm == "kinect2":
            _overlay_kinect_previews(
                out,
                depth,
                packet.get("ir"),
                res.mask if res else None,
                packet.get("rgb_aligned"),
                packet.get("rgb_raw"),
                show_raw=main_is_aligned,
            )

        if show_info:
            if selected_pose_checklist:
                division_pose_map: Dict[str, list[str]] = {}
                for item in selected_pose_checklist:
                    division_pose_map.setdefault(item.division_label, []).append(item.pose_name)
                info_lines = [
                    "Selected divisions pose checklist",
                    f"First Timers: {'Yes' if profile.first_timers else 'No'}",
                ]
                for division_label, pose_names in division_pose_map.items():
                    info_lines.append(division_label)
                    for pose_name in pose_names:
                        info_lines.append(f"- {pose_name}")
            else:
                routine_names = [POSES[p].display for p in routine]
                info_lines = [
                    f"{cat} info",
                    _CATEGORY_NOTES.get(cat, "Category info unavailable."),
                    f"First Timers: {'Yes' if profile.first_timers else 'No'}",
                    "Routine:",
                ] + [f"- {name}" for name in routine_names]
            if len(info_lines) > 22:
                info_lines = info_lines[:22]
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
        if _window_closed():
            should_exit = True
            break
        if key in (ord('1'),):
            active_tab = "posing"
        if key in (ord('2'),):
            active_tab = "metrics"
        if key == 9:
            active_tab = "metrics" if active_tab == "posing" else "posing"
        if key in (ord('n'), ord('N')):
            _set_pose_index(pose_i + 1)

        if key in (ord('p'), ord('P')):
            _set_pose_index(pose_i - 1)

        if key in (ord('f'), ord('F')):
            profile.federation = cycle_federation(profile.federation)  # type: ignore
            if selected_pose_checklist:
                routine = [item.pose_key or "mp_front" for item in selected_pose_checklist]
            else:
                routine = routine_for(cat, profile.federation)
            _set_pose_index(0)

        if key in (ord('t'), ord('T')):
            profile.first_timers = not profile.first_timers
            profile.plan.first_timers = profile.first_timers

        if key in (ord('c'), ord('C')):
            if not selected_pose_checklist:
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
            if not selected_pose_checklist:
                show_category_menu = not show_category_menu

        if key in (ord('o'), ord('O')):
            pose_timer_enabled = not pose_timer_enabled
            pose_timer_started_at = time.time()

        quality_key_map = {
            ord('3'): 1.0,
            ord('4'): 2.0,
            ord('5'): 3.0,
            ord('6'): 4.0,
            ord('7'): 5.0,
        }
        if key in quality_key_map:
            pose_quality_scores[pose_quality_key] = quality_key_map[key]
            _sync_state_metadata()

        if key in (ord('v'), ord('V')):
            if voice_listener is None:
                _init_voice_listener()
            if voice_listener is not None:
                voice_enabled = not voice_enabled

        if key in (ord('k'), ord('K')):
            _set_coach_voice(not coach_voice_enabled)

        if key in (ord('x'), ord('X')):
            if pose_key in profile.templates:
                try:
                    del profile.templates[pose_key]
                    ProfileStore.default().save(profile)
                    template_cleared_until = time.time() + 2.5
                except Exception:
                    pass

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
                    _apply_voice_command(cmd)

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
            snap = _build_snapshot_payload(
                pose_key,
                ps.score_0_100,
                props,
                pose_label=pose_display_name,
                manual_quality_score=pose_quality_value,
            )
            outp = sessions.save_snapshot(profile.name, snap)
            print(f"Saved snapshot: {outp}")

        if should_exit:
            break

    if routine:
        _finalize_pose(routine[pose_i])

    if voice_listener is not None:
        voice_listener.stop()

    if tts_speaker is not None:
        tts_speaker.stop()

    source.stop()
    cv2.destroyAllWindows()


def compute_template_features(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    # Store *absolute* feature values from the captured frame.
    from .metrics.pose_features import compute_features, PoseLandmarks

    feats = compute_features(PoseLandmarks(pts=landmarks))
    return feats
