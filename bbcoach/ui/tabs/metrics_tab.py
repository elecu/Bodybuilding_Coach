from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...app_state import get_global_app_state
from ...athlete_profiles import AthleteProfile
from ...federations.coaching_targets import load_coaching_targets
from ...federations.config_loader import load_federation_config
from ...federations.eligibility import evaluate_division_eligibility
from ...federations.library import RULES, cycle_federation
from ...federations.pose_checklist import selected_division_labels
from ...federations.specs import (
    FederationSpecsLibrary,
    SelectedDivisionRef,
    load_federation_specs,
)
from ...metrics.body_compare import compare_metrics
from ...metrics.body_proxy import compute_body_proxy
from ...metrics.body_scoring import score_metrics
from ...poses.library import ROUTINES
from ...storage.body_sessions import BodySessionStore
from ...storage.session_paths import SessionPaths
from ...vision.pointcloud import depth_to_pointcloud
from ...core.scan_capture import FourViewScanCapture
from ...core.metrics import compute_metrics_for_scan
from ...core.registration import auto_merge
from ...core.manual_align import manual_align
from ...vision.overlay import draw_mask_outline


_TEXT_FONT = cv2.FONT_HERSHEY_TRIPLEX
_TEXT_COLOUR = (235, 235, 235)
_TEXT_ACCENT = (0, 230, 255)
_TEXT_BG = (8, 8, 10)
_TEMP_DISABLED_FEDERATION_IDS = {"nabba"}


@dataclass
class UIButton:
    key: str
    label: str
    rect: Tuple[int, int, int, int]
    active: bool = False
    disabled: bool = False


def _draw_text_bg(img: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 6, alpha: float = 0.55) -> None:
    y0 = max(0, y - h - pad)
    x0 = max(0, x - pad)
    x1 = min(img.shape[1] - 1, x + w + pad)
    y1 = min(img.shape[0] - 1, y + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _TEXT_BG, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _put_text(
    img: np.ndarray,
    text: str,
    y: int,
    scale: float = 0.55,
    colour: Tuple[int, int, int] = _TEXT_COLOUR,
    x: int = 12,
    bg: bool = True,
    thickness: int = 1,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, _TEXT_FONT, scale, thickness)
    if bg:
        _draw_text_bg(img, x, y, tw, th)
    cv2.putText(img, text, (x, y), _TEXT_FONT, scale, colour, thickness, cv2.LINE_AA)


def _draw_button(img: np.ndarray, btn: UIButton) -> None:
    x, y, w, h = btn.rect
    overlay = img.copy()
    base = (20, 20, 24)
    on = (30, 160, 200)
    disabled = (50, 50, 60)
    color = disabled if btn.disabled else (on if btn.active else base)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    (tw, th), _ = cv2.getTextSize(btn.label, _TEXT_FONT, 0.5, 1)
    tx = x + max(6, (w - tw) // 2)
    ty = y + h - max(6, (h - th) // 2)
    text_colour = (140, 140, 150) if btn.disabled else _TEXT_COLOUR
    cv2.putText(img, btn.label, (tx, ty), _TEXT_FONT, 0.5, text_colour, 1, cv2.LINE_AA)


def _draw_countdown(img: np.ndarray, seconds_left: int) -> None:
    h, w = img.shape[:2]
    label = str(max(0, int(seconds_left)))
    scale = 3.2 if w >= 1200 else 2.4
    thickness = 4
    (tw, th), _ = cv2.getTextSize(label, _TEXT_FONT, scale, thickness)
    x = max(0, (w - tw) // 2)
    y = max(th + 20, (h + th) // 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 30, y - th - 30), (x + tw + 30, y + 30), _TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, label, (x, y), _TEXT_FONT, scale, _TEXT_ACCENT, thickness, cv2.LINE_AA)


def _in_rect(pt: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
    x, y = pt
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


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
    return cropped, (-float(x0), -float(y0), scale)


def _fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _front_width(landmarks: Optional[Dict[str, Tuple[float, float]]]) -> Optional[float]:
    if not landmarks:
        return None
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    nose = landmarks.get("nose")
    if not ls or not rs or not nose:
        return None
    width = abs(float(rs[0]) - float(ls[0]))
    if width < 0.12:
        return None
    minx = min(float(ls[0]), float(rs[0]))
    maxx = max(float(ls[0]), float(rs[0]))
    if not (minx <= float(nose[0]) <= maxx):
        return None
    return width


class MetricsTab:
    def __init__(self, profile_name: str) -> None:
        self.profile_name = profile_name
        self.store = BodySessionStore.default()
        self._body_profile: Optional[Dict[str, Any]] = None
        self.height_cm: Optional[float] = None
        self.weight_kg: Optional[float] = None
        self._input_mode: Optional[str] = None
        self._input_value: str = ""
        self._input_error: str = ""
        self._status_msg: str = ""
        self._status_until: float = 0.0
        self._capture_active: bool = False
        self._capture_started: float = 0.0
        self._sessions: List[Dict[str, Any]] = []
        self._last_refresh: float = 0.0
        self._pose_filter: Optional[str] = None
        self._pose_filter_options: List[Optional[str]] = []
        self._selected_a: Optional[str] = None
        self._selected_b: Optional[str] = None
        self._buttons: List[UIButton] = []
        self._session_hits: List[Tuple[Tuple[int, int, int, int], str]] = []
        self._last_frame: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None
        self._last_props: Optional[object] = None
        self._last_category: Optional[str] = None
        self._last_pose_key: Optional[str] = None
        self._last_federation: Optional[str] = None
        self._selected_category: Optional[str] = None
        self._selected_federation: Optional[str] = None
        self._category_suggestion: Optional[str] = None
        self._category_suggestion_reason: str = ""
        self._pending_category_override: Optional[str] = None
        self._pending_federation_override: Optional[str] = None
        self._last_landmarks: Optional[Dict[str, Tuple[float, float]]] = None
        self._depth_available: bool = False
        self._depth_aligned: bool = False
        self._depth_units: Optional[str] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_intrinsics: Optional[Dict[str, Any]] = None
        self._scan_active: bool = False
        self._scan_frames: int = 0
        self._scan_prep_active: bool = False
        self._scan_prep_started_ts: Optional[float] = None
        self._scan_prep_remaining_s: Optional[int] = None
        self._scan_prep_duration_s: float = 5.0
        self._scan_started_ts: Optional[float] = None
        self._scan_duration_s: float = 30.0
        self._scan_min_frames: int = 90
        self._scan_capture_hz: float = 5.0
        self._scan_last_capture_ts: Optional[float] = None
        self._scan_remaining_s: Optional[int] = None
        self._scan_countdown_last: Optional[int] = None
        self._scan_countdown_pending: Optional[int] = None
        self._scan_points: List[np.ndarray] = []
        self._scan_colors: List[np.ndarray] = []
        self._last_scan: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
        self._scan4: Optional[FourViewScanCapture] = None
        self._scan4_status: Optional[str] = None
        self._scan4_dir: Optional[Path] = None
        self._open3d_available = False
        self._open3d_last_check: float = 0.0
        self._open3d_error: str = ""
        self._refresh_open3d_status(force=True)
        self._scan4_active: bool = False
        self._scan4_prep_active: bool = False
        self._scan4_prep_started_monotonic: Optional[float] = None
        self._scan4_prep_duration_seconds: int = 5
        self._scan4_prep_countdown_last: Optional[int] = None
        self._scan4_step: int = 0
        self._scan4_step_deadline_monotonic: Optional[float] = None
        self._scan4_countdown_seconds: int = 5
        self._scan4_ready_hold_seconds: float = 0.6
        self._scan4_ready_since_monotonic: Optional[float] = None
        self._scan4_intro_announced_step: Optional[int] = None
        self._scan4_side_ref_sign: Optional[float] = None
        self._scan4_countdown_remaining: Optional[int] = None
        self._scan4_countdown_last: Optional[int] = None
        self._scan4_countdown_pending: Optional[str] = None
        self._scan4_capture_fired: bool = False
        self._scan4_tts_reset_pending: bool = False
        self._scan4_post_capture_cooldown_until: Optional[float] = None
        self._scan4_guide_enabled: bool = True
        self._scan4_metrics_snapshot: Optional[Dict[str, Any]] = None
        self._app_state = get_global_app_state()
        self._spec_library: FederationSpecsLibrary = load_federation_specs()
        self._apply_temp_disabled_federations()
        self._coaching_targets = load_coaching_targets()
        self._ui_mode: str = "metrics"
        self._browse_ref: Optional[SelectedDivisionRef] = None
        self._browse_tree_hits: List[Tuple[Tuple[int, int, int, int], SelectedDivisionRef]] = []
        self._selection_hits: List[Tuple[Tuple[int, int, int, int], SelectedDivisionRef, bool, List[str]]] = []
        self._selection_cursor_index: int = 0
        self._selection_scroll_index: int = 0
        self._comparison_scroll_chars: int = 0
        self._selection_refs: list[SelectedDivisionRef] = list(self._app_state.state.selected_divisions)
        self._selection_refs = self._filter_disabled_selected_refs(self._selection_refs)
        self._app_state.set_selected_divisions(self._selection_refs)
        if not self._browse_ref:
            refs = self._spec_library.iter_division_refs()
            if refs:
                self._browse_ref = refs[0]

    def _apply_temp_disabled_federations(self) -> None:
        for fed_id in _TEMP_DISABLED_FEDERATION_IDS:
            self._spec_library.federations.pop(str(fed_id).lower(), None)

    def _filter_disabled_selected_refs(self, refs: list[SelectedDivisionRef]) -> list[SelectedDivisionRef]:
        allowed = set(self._spec_library.federations.keys())
        return [ref for ref in refs if str(ref.federation_id or "").lower() in allowed]

    def _available_categories(self) -> List[str]:
        cats = [str(k) for k in ROUTINES.keys()]
        if not cats:
            return ["Mens Physique", "Classic", "Bodybuilding"]
        return cats

    def _set_context_defaults(self, profile_federation: str, category: str) -> None:
        if self._selected_federation not in RULES:
            self._selected_federation = profile_federation if profile_federation in RULES else "WNBF_UK"

        categories = self._available_categories()
        if self._selected_category not in categories:
            self._selected_category = category if category in categories else categories[0]

        self._last_federation = self._selected_federation
        self._last_category = self._selected_category

    def _cycle_federation(self) -> None:
        current = self._selected_federation if self._selected_federation in RULES else "WNBF_UK"
        nxt = cycle_federation(current)  # type: ignore[arg-type]
        self._selected_federation = nxt
        self._last_federation = nxt
        self._pending_federation_override = nxt
        self._set_status(f"Federation set to {nxt}.")

    def _cycle_category(self) -> None:
        categories = self._available_categories()
        if not categories:
            return
        cur = self._selected_category if self._selected_category in categories else categories[0]
        idx = categories.index(cur)
        nxt = categories[(idx + 1) % len(categories)]
        self._selected_category = nxt
        self._last_category = nxt
        self._pending_category_override = nxt
        self._set_status(f"Classification set to {nxt}.")

    def _suggest_category(self, metrics: Optional[Dict[str, Any]]) -> None:
        props = (metrics or {}).get("proportions") or {}
        sw = self._safe_float(props.get("shoulder_to_waist"))
        ul = self._safe_float(props.get("upper_to_lower_area"))
        chest_ratio = self._safe_float(props.get("chest_to_waist"))

        suggestion = "Classic"
        reason = "Balanced shape profile."
        if sw is not None and sw >= 1.35:
            suggestion = "Mens Physique"
            reason = "Strong shoulder-to-waist ratio."
        elif sw is not None and sw < 1.20 and chest_ratio is not None and chest_ratio < 1.10:
            suggestion = "Bodybuilding"
            reason = "More development runway for upper-body dominance."
        elif ul is not None and ul < 0.95:
            suggestion = "Mens Physique"
            reason = "Upper/lower balance favors physique presentation."

        if suggestion not in self._available_categories():
            suggestion = self._available_categories()[0]

        self._category_suggestion = suggestion
        self._category_suggestion_reason = reason
        self._selected_category = suggestion
        self._last_category = suggestion
        self._pending_category_override = suggestion
        self._set_status(f"Suggested class: {suggestion} ({reason})", seconds=4.5)

    def pop_federation_override(self) -> Optional[str]:
        val = self._pending_federation_override
        self._pending_federation_override = None
        return val

    def pop_category_override(self) -> Optional[str]:
        val = self._pending_category_override
        self._pending_category_override = None
        return val

    def _ensure_body_profile(self) -> None:
        if self._body_profile is None:
            self._body_profile = self.store.load_body_profile(self.profile_name)
            if self._body_profile:
                self.height_cm = self._safe_float(self._body_profile.get("height_cm"))
                self.weight_kg = self._safe_float(self._body_profile.get("weight_kg"))
        if self.height_cm is None:
            self._start_input("height")

    def _start_input(self, mode: str) -> None:
        if self._input_mode:
            return
        self._input_mode = mode
        self._input_value = ""
        self._input_error = ""

    def _save_body_profile(self) -> None:
        data = {
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.store.save_body_profile(self.profile_name, data)
        self._body_profile = data

    def _refresh_sessions(self) -> None:
        self._sessions = self.store.list_sessions(self.profile_name)
        pose_ids = sorted({s.get("pose_id") for s in self._sessions if s.get("pose_id")})
        self._pose_filter_options = [None] + pose_ids
        if self._pose_filter not in self._pose_filter_options:
            self._pose_filter = None
        self._last_refresh = time.time()

    def _filtered_sessions(self) -> List[Dict[str, Any]]:
        if self._pose_filter:
            return [s for s in self._sessions if s.get("pose_id") == self._pose_filter]
        return list(self._sessions)

    def _set_status(self, msg: str, seconds: float = 3.0) -> None:
        self._status_msg = msg
        self._status_until = time.time() + seconds

    def _set_ui_mode(self, mode: str) -> None:
        allowed = {"metrics", "browse_rules", "select_categories", "comparison_table"}
        self._ui_mode = mode if mode in allowed else "metrics"

    def update_live_context(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray],
        props: Optional[object],
        profile: Any,
        category: str,
        pose_key: str,
        landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict[str, Any]] = None,
        depth_aligned: Optional[bool] = None,
        depth_units: Optional[str] = None,
    ) -> None:
        """Update internal live state without drawing overlay controls on the frame."""
        self._last_frame = frame
        self._last_mask = mask
        self._last_props = props
        self._last_pose_key = pose_key
        self._last_landmarks = landmarks
        self._last_depth = depth
        self._last_intrinsics = intrinsics
        self._depth_aligned = bool(depth_aligned) if depth_aligned is not None else False
        self._depth_units = depth_units
        self._depth_available = bool(depth is not None and depth.size > 0 and np.isfinite(depth).any())
        profile_fed = str(getattr(profile, "federation", "") or "")
        self._set_context_defaults(profile_fed, category)
        self._ensure_body_profile()
        if time.time() - self._last_refresh > 3.0:
            self._refresh_sessions()
        self._update_scan()
        self._update_scan4()
        state_selected = list(self._app_state.state.selected_divisions)
        if [ref.key() for ref in state_selected] != [ref.key() for ref in self._selection_refs]:
            self._selection_refs = state_selected

    def latest_status_message(self) -> str:
        if self._status_msg and time.time() < self._status_until:
            return self._status_msg
        if self._scan4_status:
            return self._scan4_status
        return ""

    def toggle_metrics_session(self) -> bool:
        if self._capture_active:
            self._stop_capture()
            return False
        self._start_capture()
        return self._capture_active

    def start_scan4_session(self) -> None:
        self._start_scan4()

    def stop_scan4_session(self) -> None:
        if self._scan4_active or self._scan4_prep_active:
            self._scan4_active = False
            self._scan4_prep_active = False
            self._scan4_prep_started_monotonic = None
            self._scan4_prep_countdown_last = None
            self._scan4_countdown_remaining = None
            self._scan4_countdown_last = None
            self._scan4_step_deadline_monotonic = None
            self._set_status("4-side scan stopped.")

    def load_scan4_session(self) -> None:
        self._load_scan4_from_folder()

    def set_scan4_dir(self, scan_dir: Path) -> bool:
        resolved = self._scan4_resolve_session_dir(scan_dir)
        if resolved is None:
            return False
        self._scan4 = None
        self._scan4_dir = resolved
        self._scan4_active = False
        self._scan4_prep_active = False
        self._scan4_prep_started_monotonic = None
        self._scan4_prep_countdown_last = None
        self._scan4_step = 0
        self._scan4_step_deadline_monotonic = None
        self._scan4_ready_since_monotonic = None
        self._scan4_intro_announced_step = None
        self._scan4_side_ref_sign = None
        self._scan4_countdown_remaining = None
        self._scan4_countdown_last = None
        self._scan4_countdown_pending = None
        self._scan4_capture_fired = False
        self._scan4_tts_reset_pending = False
        self._scan4_post_capture_cooldown_until = None
        self._scan4_status = f"Loaded scan: {resolved.name}"
        self._scan4_reload_metrics_snapshot()
        self._set_status(f"Loaded existing scan folder: {resolved}", seconds=4.5)
        return True

    def compute_scan4_metrics(self) -> bool:
        return self._scan4_compute_metrics()

    def auto_merge_scan4(self) -> None:
        self._scan4_auto_merge()

    def manual_align_scan4(self) -> None:
        self._scan4_manual_align()

    def export_scan4_report(self) -> None:
        self._scan4_export()

    def current_scan_dir(self) -> Optional[Path]:
        return self._scan4_dir

    def _active_athlete_profile(self, profile: Any) -> AthleteProfile:
        active = deepcopy(self._app_state.state.active_profile) if self._app_state.state.active_profile else None
        if active is None:
            active = AthleteProfile(
                profile_id=f"live-{self.profile_name.lower().replace(' ', '_')}",
                profile_name=self.profile_name,
                sex_category="Other",
                age_years=None,
                height_cm=float(self.height_cm or 0.0),
                weight_kg=float(self.weight_kg or 0.0),
                bodyfat_percent=None,
                drug_tested_preference=True,
                competition_date=None,
            )
        active.profile_name = self.profile_name
        if self.height_cm is not None:
            active.height_cm = float(self.height_cm)
        if self.weight_kg is not None:
            active.weight_kg = float(self.weight_kg)
        if self.weight_kg is None:
            active.weight_kg = 0.0
        if self.height_cm is None:
            active.height_cm = 0.0
        plan = getattr(profile, "plan", None)
        if plan is not None:
            comp_date = getattr(plan, "competition_date", None)
            if comp_date:
                active.competition_date = str(comp_date)
        self._app_state.set_active_profile(active)
        return active

    def _is_selected_ref(self, ref: SelectedDivisionRef) -> bool:
        return any(item.key() == ref.key() for item in self._selection_refs)

    def _set_selected_refs(self, refs: list[SelectedDivisionRef]) -> None:
        dedup: dict[str, SelectedDivisionRef] = {}
        for ref in refs:
            dedup[ref.key()] = ref
        self._selection_refs = self._filter_disabled_selected_refs(list(dedup.values()))
        self._app_state.set_selected_divisions(self._selection_refs)

    def _toggle_selected_ref(self, ref: SelectedDivisionRef) -> None:
        if self._is_selected_ref(ref):
            kept = [item for item in self._selection_refs if item.key() != ref.key()]
            self._set_selected_refs(kept)
            return
        self._set_selected_refs(self._selection_refs + [ref])

    def _candidate_class_specs(self, division: Any) -> list[Any]:
        classes = list(getattr(division, "classes", []) or [])
        if classes:
            return classes
        classes.extend(getattr(getattr(division, "eligibility", None), "height_classes", []) or [])
        classes.extend(getattr(getattr(division, "eligibility", None), "weight_classes", []) or [])
        dedup: dict[str, Any] = {}
        for cls in classes:
            cls_id = str(getattr(cls, "class_id", "") or "")
            if not cls_id:
                continue
            dedup[cls_id] = cls
        return list(dedup.values())

    def _matches_profile_sex(self, athlete: AthleteProfile, division: Any) -> bool:
        allowed = [str(item).strip().lower() for item in (division.eligibility.sex_categories or []) if str(item).strip()]
        if not allowed:
            return True
        profile_sex = str(athlete.sex_category or "").strip().lower()
        if not profile_sex:
            return False
        return profile_sex in allowed

    def _matches_profile_sex_for_class(self, athlete: AthleteProfile, class_spec: Any) -> bool:
        profile_sex = str(athlete.sex_category or "").strip().lower()
        if not profile_sex:
            return False
        class_name = str(getattr(class_spec, "class_name", "") or "")
        class_notes = str(getattr(class_spec, "notes", "") or "")
        class_text = class_name.lower()
        notes_text = class_notes.lower()
        women_name_hint = bool(re.search(r"\b(women|women'?s|female|females|ladies|woman)\b", class_text))
        men_name_hint = bool(re.search(r"\b(men|men'?s|male|males|man)\b", class_text))
        women_notes_hint = bool(re.search(r"\b(women|women'?s|female|females|ladies|woman)\b", notes_text))
        men_notes_hint = bool(re.search(r"\b(men|men'?s|male|males|man)\b", notes_text))
        if women_name_hint != men_name_hint:
            women_hint = women_name_hint
            men_hint = men_name_hint
        else:
            women_hint = women_name_hint or women_notes_hint
            men_hint = men_name_hint or men_notes_hint
        if women_hint and not men_hint:
            return profile_sex == "female"
        if men_hint and not women_hint:
            return profile_sex == "male"
        return True

    def _passes_height_suggestion_filter(self, athlete: AthleteProfile, class_spec: Any) -> bool:
        """User-facing suggestion rule: do not suggest classes with max height above athlete height."""
        try:
            athlete_height = float(athlete.height_cm)
        except Exception:
            return True
        max_height_raw = getattr(class_spec, "max_height_cm", None)
        if max_height_raw is None:
            return True
        try:
            class_max_height = float(max_height_raw)
        except Exception:
            return True
        return class_max_height <= (athlete_height + 1e-6)

    def _selection_rows(self, athlete: AthleteProfile) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fed in self._spec_library.list_federations():
            for division in fed.divisions:
                if not self._matches_profile_sex(athlete, division):
                    continue
                class_specs = self._candidate_class_specs(division)
                if class_specs:
                    for cls in class_specs:
                        if not self._matches_profile_sex_for_class(athlete, cls):
                            continue
                        if not self._passes_height_suggestion_filter(athlete, cls):
                            continue
                        ref = SelectedDivisionRef(
                            federation_id=fed.federation_id,
                            division_id=division.division_id,
                            class_id=cls.class_id,
                        )
                        result = evaluate_division_eligibility(athlete, division, class_spec=cls)
                        rows.append(
                            {
                                "federation": fed,
                                "division": division,
                                "class_spec": cls,
                                "ref": ref,
                                "result": result,
                            }
                        )
                else:
                    ref = SelectedDivisionRef(federation_id=fed.federation_id, division_id=division.division_id)
                    result = evaluate_division_eligibility(athlete, division, class_spec=None)
                    rows.append(
                        {
                            "federation": fed,
                            "division": division,
                            "class_spec": None,
                            "ref": ref,
                            "result": result,
                        }
                    )
        return rows

    def _render_browse_rules(self, out: np.ndarray, athlete: AthleteProfile, y_start: int) -> int:
        self._browse_tree_hits = []
        rows = self._selection_rows(athlete)
        if not rows:
            _put_text(out, "No federation spec files loaded.", y_start, scale=0.5, colour=(255, 190, 120), thickness=1)
            return y_start + 20

        if self._browse_ref is None:
            self._browse_ref = rows[0]["ref"]

        h, w = out.shape[:2]
        tree_x = 12
        detail_x = max(int(w * 0.45), 420)
        y = y_start
        _put_text(out, "FEDERATIONS/CATEGORIES -> BROWSE RULES", y, scale=0.5, colour=_TEXT_ACCENT, thickness=1, x=tree_x)
        y += 22

        current_fed = ""
        current_div = ""
        line_h = 18
        max_tree_bottom = h - 120
        for row in rows:
            fed = row["federation"]
            div = row["division"]
            cls = row["class_spec"]
            ref = row["ref"]
            if y > max_tree_bottom:
                break
            if current_fed != fed.federation_id:
                current_fed = fed.federation_id
                _put_text(out, fed.federation_name, y, scale=0.46, colour=(200, 230, 255), thickness=1, x=tree_x)
                y += line_h
                current_div = ""
            if current_div != div.division_id:
                current_div = div.division_id
                div_label = f"  - {div.division_name}"
                selected_div = (
                    self._browse_ref is not None
                    and self._browse_ref.federation_id == ref.federation_id
                    and self._browse_ref.division_id == ref.division_id
                    and self._browse_ref.class_id is None
                )
                div_colour = (180, 255, 180) if selected_div else _TEXT_COLOUR
                _put_text(out, div_label, y, scale=0.43, colour=div_colour, thickness=1, x=tree_x)
                (tw, th), _meta = cv2.getTextSize(div_label, _TEXT_FONT, 0.43, 1)
                self._browse_tree_hits.append(
                    (
                        (tree_x, y - th - 4, tw + 6, th + 10),
                        SelectedDivisionRef(ref.federation_id, ref.division_id),
                    )
                )
                y += line_h
            if cls is not None:
                cls_label = f"      * {cls.class_name}"
                selected_cls = self._browse_ref is not None and self._browse_ref.key() == ref.key()
                cls_colour = (180, 255, 180) if selected_cls else (200, 200, 200)
                _put_text(out, cls_label, y, scale=0.41, colour=cls_colour, thickness=1, x=tree_x)
                (tw, th), _meta = cv2.getTextSize(cls_label, _TEXT_FONT, 0.41, 1)
                self._browse_tree_hits.append(((tree_x, y - th - 4, tw + 6, th + 10), ref))
                y += line_h

        detail_ref = self._browse_ref
        if detail_ref is None:
            return y

        fed = self._spec_library.get_federation(detail_ref.federation_id)
        div = self._spec_library.get_division(detail_ref)
        cls = self._spec_library.get_division_class(detail_ref)
        if fed is None or div is None:
            return y

        eval_result = evaluate_division_eligibility(athlete, div, class_spec=cls)
        dy = y_start
        _put_text(out, "DETAILS", dy, scale=0.5, colour=_TEXT_ACCENT, thickness=1, x=detail_x)
        dy += 22
        label = f"{fed.federation_name} / {div.division_name}"
        if cls is not None:
            label += f" / {cls.class_name}"
        _put_text(out, label, dy, scale=0.44, thickness=1, x=detail_x)
        dy += 20
        elig_line = f"Eligibility: {'eligible' if eval_result.eligible else 'ineligible'} | confidence {eval_result.confidence}"
        _put_text(
            out,
            elig_line,
            dy,
            scale=0.42,
            colour=(180, 230, 180) if eval_result.eligible else (255, 190, 120),
            thickness=1,
            x=detail_x,
        )
        dy += 18
        if eval_result.needs_user_confirmation:
            _put_text(out, "needs_user_confirmation=true", dy, scale=0.42, colour=(255, 190, 120), thickness=1, x=detail_x)
            dy += 18

        req_lines: list[str] = []
        if div.eligibility.sex_categories:
            req_lines.append("Sex: " + ", ".join(div.eligibility.sex_categories))
        if div.eligibility.age_min_years is not None or div.eligibility.age_max_years is not None:
            req_lines.append(
                f"Age: {div.eligibility.age_min_years if div.eligibility.age_min_years is not None else '-inf'}"
                f" to {div.eligibility.age_max_years if div.eligibility.age_max_years is not None else '+inf'}"
            )
        if div.eligibility.age_rule_text:
            req_lines.append(div.eligibility.age_rule_text)
        if div.eligibility.novice_restriction_text:
            req_lines.append(div.eligibility.novice_restriction_text)
        if cls is not None:
            cls_bits = []
            if cls.min_height_cm is not None or cls.max_height_cm is not None:
                cls_bits.append(f"height {cls.min_height_cm or '-inf'} to {cls.max_height_cm or '+inf'} cm")
            if cls.min_weight_kg is not None or cls.max_weight_kg is not None:
                cls_bits.append(f"weight {cls.min_weight_kg or '-inf'} to {cls.max_weight_kg or '+inf'} kg")
            if cls_bits:
                req_lines.append("Class limits: " + "; ".join(cls_bits))

        if not req_lines:
            req_lines.append("Eligibility requirements: null")

        _put_text(out, "Eligibility requirements:", dy, scale=0.43, colour=(200, 230, 255), thickness=1, x=detail_x)
        dy += 18
        for line in req_lines[:6]:
            _put_text(out, line, dy, scale=0.4, thickness=1, x=detail_x)
            dy += 16

        _put_text(out, "Rounds:", dy, scale=0.43, colour=(200, 230, 255), thickness=1, x=detail_x)
        dy += 18
        if div.rounds:
            for rnd in div.rounds[:5]:
                text = rnd.round_name
                if rnd.judged_on:
                    text += f" - {rnd.judged_on}"
                if rnd.routine_time_limit_seconds is not None:
                    text += f" ({rnd.routine_time_limit_seconds}s)"
                _put_text(out, text, dy, scale=0.4, thickness=1, x=detail_x)
                dy += 16
        else:
            _put_text(out, "null", dy, scale=0.4, thickness=1, x=detail_x)
            dy += 16

        _put_text(out, "Mandatory poses:", dy, scale=0.43, colour=(200, 230, 255), thickness=1, x=detail_x)
        dy += 18
        if div.mandatory_poses:
            for pose in div.mandatory_poses[:10]:
                _put_text(out, f"- {pose}", dy, scale=0.4, thickness=1, x=detail_x)
                dy += 16
        else:
            _put_text(out, "null", dy, scale=0.4, thickness=1, x=detail_x)
            dy += 16

        _put_text(out, "Attire:", dy, scale=0.43, colour=(200, 230, 255), thickness=1, x=detail_x)
        dy += 18
        _put_text(out, div.attire or "null", dy, scale=0.4, thickness=1, x=detail_x)
        dy += 16

        _put_text(out, "Sources:", dy, scale=0.43, colour=(200, 230, 255), thickness=1, x=detail_x)
        dy += 18
        for src in (div.sources.source_urls + div.sources.source_files)[:6]:
            _put_text(out, src, dy, scale=0.38, thickness=1, x=detail_x)
            dy += 14

        if div.needs_verification:
            _put_text(out, "needs_verification=true", dy, scale=0.4, colour=(255, 190, 120), thickness=1, x=detail_x)
            dy += 16

        return max(y, dy)

    def _render_select_categories(self, out: np.ndarray, athlete: AthleteProfile, y_start: int) -> int:
        self._selection_hits = []
        rows = self._selection_rows(athlete)
        if not rows:
            _put_text(out, "No federation spec files loaded.", y_start, scale=0.5, colour=(255, 190, 120), thickness=1)
            return y_start + 20

        _put_text(out, "FEDERATIONS/CATEGORIES -> SELECT CATEGORIES", y_start, scale=0.5, colour=_TEXT_ACCENT, thickness=1)
        y = y_start + 22
        _put_text(out, "Only eligible items are selectable. Ineligible rows are greyed out.", y, scale=0.4, thickness=1)
        y += 18

        h, _w = out.shape[:2]
        max_rows = max(6, (h - y - 130) // 18)
        self._selection_cursor_index = int(np.clip(self._selection_cursor_index, 0, max(0, len(rows) - 1)))
        if self._selection_cursor_index < self._selection_scroll_index:
            self._selection_scroll_index = self._selection_cursor_index
        if self._selection_cursor_index >= self._selection_scroll_index + max_rows:
            self._selection_scroll_index = self._selection_cursor_index - max_rows + 1

        start = self._selection_scroll_index
        end = min(len(rows), start + max_rows)
        for idx in range(start, end):
            row = rows[idx]
            ref: SelectedDivisionRef = row["ref"]
            fed = row["federation"]
            div = row["division"]
            cls = row["class_spec"]
            result = row["result"]

            checked = self._is_selected_ref(ref)
            label = f"[{'x' if checked else ' '}] {fed.federation_name} / {div.division_name}"
            if cls is not None:
                label += f" / {cls.class_name}"
            if not result.eligible:
                label += " (ineligible)"

            selected = idx == self._selection_cursor_index
            colour = _TEXT_COLOUR
            if not result.eligible:
                colour = (130, 130, 140)
            if selected:
                colour = (180, 255, 180) if result.eligible else (255, 190, 120)

            _put_text(out, label, y, scale=0.41, colour=colour, thickness=1)
            (tw, th), _meta = cv2.getTextSize(label, _TEXT_FONT, 0.41, 1)
            self._selection_hits.append(((12, y - th - 4, tw + 6, th + 10), ref, bool(result.eligible), list(result.reasons)))
            y += 18

        current = rows[self._selection_cursor_index]
        cur_result = current["result"]
        y += 8
        _put_text(out, "Current eligibility reasons:", y, scale=0.44, colour=(200, 230, 255), thickness=1)
        y += 18
        for reason in (cur_result.reasons or [])[:5]:
            _put_text(out, reason, y, scale=0.39, thickness=1)
            y += 16

        y += 8
        _put_text(out, "Selected divisions:", y, scale=0.44, colour=(200, 230, 255), thickness=1)
        y += 18
        labels = selected_division_labels(self._spec_library, self._selection_refs)
        if not labels:
            _put_text(out, "None selected.", y, scale=0.39, thickness=1)
            y += 16
        else:
            for label in labels[:6]:
                _put_text(out, f"- {label}", y, scale=0.39, thickness=1)
                y += 16

        return y

    def _render_comparison_table(self, out: np.ndarray, athlete: AthleteProfile, y_start: int) -> int:
        rows = self._selection_rows(athlete)
        _put_text(out, "COMPARISON TABLE", y_start, scale=0.5, colour=_TEXT_ACCENT, thickness=1)
        y = y_start + 20
        _put_text(out, "Horizontal scroll: '[' left, ']' right", y, scale=0.4, thickness=1)
        y += 18

        header = "Federation | Division | Eligibility | Rounds | Mandatory poses | Attire | Notes"
        table_lines = [header]
        for row in rows:
            fed = row["federation"]
            div = row["division"]
            cls = row["class_spec"]
            res = row["result"]
            division_label = div.division_name + (f" / {cls.class_name}" if cls is not None else "")
            eligibility_text = "Eligible" if res.eligible else "Ineligible"
            if res.needs_user_confirmation:
                eligibility_text += " (needs confirmation)"
            round_text = "; ".join(
                [
                    (rnd.round_name + (f": {rnd.judged_on}" if rnd.judged_on else ""))
                    + (f" [{rnd.routine_time_limit_seconds}s]" if rnd.routine_time_limit_seconds is not None else "")
                    for rnd in div.rounds
                ]
            )
            pose_text = "; ".join(div.mandatory_poses or ["null"])
            attire_text = div.attire or "null"
            notes_text = div.notes_keywords or "null"
            table_lines.append(
                " | ".join(
                    [
                        fed.federation_name,
                        division_label,
                        eligibility_text,
                        round_text,
                        pose_text,
                        attire_text,
                        notes_text,
                    ]
                )
            )

        max_chars = max(20, int((out.shape[1] - 24) / 7.4))
        self._comparison_scroll_chars = max(0, self._comparison_scroll_chars)
        for line in table_lines[: min(len(table_lines), 28)]:
            start = min(self._comparison_scroll_chars, max(0, len(line) - 1))
            clipped = line[start : start + max_chars]
            _put_text(out, clipped, y, scale=0.39, thickness=1)
            y += 16

        return y
    def _refresh_open3d_status(self, force: bool = False) -> bool:
        now = time.time()
        if (not force) and (now - self._open3d_last_check) < 2.0:
            return self._open3d_available
        self._open3d_last_check = now
        try:
            import open3d  # noqa: F401

            self._open3d_available = True
            self._open3d_error = ""
        except Exception as exc:
            self._open3d_available = False
            self._open3d_error = str(exc)
        return self._open3d_available

    def _open3d_missing_msg(self) -> str:
        hint = "Open3D not available. Install it in this env (e.g. `pip install open3d`) and restart."
        if self._open3d_error:
            return f"{hint} ({self._open3d_error})"
        return hint

    def _scan4_snapshot_from_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        views = metrics.get("views") or []
        by_angle: Dict[int, Dict[str, Any]] = {}
        for v in views:
            try:
                ang = int(v.get("angle_deg", -1))
            except Exception:
                ang = -1
            by_angle[ang] = v
        front = by_angle.get(0) or (views[0] if views else {})
        key_widths_m = (front or {}).get("key_widths_m") or {}

        def to_float(v: Any) -> float:
            try:
                return float(v)
            except Exception:
                return 0.0

        widths_cm = {k: round(to_float(v) * 100.0, 2) for k, v in key_widths_m.items()}
        shoulders = to_float(key_widths_m.get("shoulders"))
        chest = to_float(key_widths_m.get("chest"))
        waist = to_float(key_widths_m.get("waist"))
        hips = to_float(key_widths_m.get("hips"))
        thigh = to_float(key_widths_m.get("thigh"))

        def ratio(a: float, b: float) -> Optional[float]:
            if b <= 0:
                return None
            return round(a / b, 3)

        ratios = {
            "shoulder_to_waist": ratio(shoulders, waist),
            "chest_to_waist": ratio(chest, waist),
            "hip_to_waist": ratio(hips, waist),
            "thigh_to_waist": ratio(thigh, waist),
        }
        return {
            "key_widths_cm": widths_cm,
            "ratios": ratios,
        }

    def _scan4_reload_metrics_snapshot(self) -> None:
        self._scan4_metrics_snapshot = None
        if not self._scan4_dir:
            return
        derived = self._scan4_dir / "derived"
        scorecard_path = derived / "scorecard.json"
        if scorecard_path.exists():
            try:
                scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
                ratios = scorecard.get("ratios") or {}
                widths = scorecard.get("key_widths_cm") or {}
                self._scan4_metrics_snapshot = {
                    "ratios": ratios,
                    "key_widths_cm": widths,
                }
                return
            except Exception:
                pass
        metrics_path = derived / "metrics.json"
        if not metrics_path.exists():
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self._scan4_metrics_snapshot = self._scan4_snapshot_from_metrics(metrics)
        except Exception:
            self._scan4_metrics_snapshot = None

    def _apply_cutout(self, frame: np.ndarray, mask: Optional[np.ndarray], bg_colour=(0, 0, 0)) -> np.ndarray:
        if mask is None:
            return frame
        blur = cv2.GaussianBlur(mask, (9, 9), 0)
        alpha = blur.astype(np.float32) / 255.0
        alpha = alpha[..., None]
        bg = np.zeros_like(frame, dtype=np.float32)
        bg[:] = bg_colour
        out = frame.astype(np.float32) * alpha + bg * (1 - alpha)
        return out.astype(np.uint8)

    def _start_capture(self) -> None:
        if self.height_cm is None:
            self._set_status("Set height before capturing.")
            self._start_input("height")
            return
        self._capture_active = True
        self._capture_started = time.time()
        self._set_status("Metrics session started.")

    def _stop_capture(self) -> None:
        if not self._capture_active:
            return
        if self._last_frame is None:
            self._set_status("No frame available to save.")
            self._capture_active = False
            return
        metrics = compute_body_proxy(self._last_mask, self.height_cm or 0.0, self.weight_kg, self._last_props)
        if not metrics:
            self._set_status("No mask available for metrics.")
            self._capture_active = False
            return
        meta = {
            "profile": self.profile_name,
            "federation_id": self._last_federation,
            "category_id": self._last_category,
            "pose_id": self._last_pose_key,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "started_at": datetime.fromtimestamp(self._capture_started).isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "capture_mode": "2d",
        }
        frame_bgr = self._last_frame.copy()
        cutout = self._apply_cutout(frame_bgr, self._last_mask)
        self.store.create_session(
            self.profile_name,
            meta,
            frame_bgr,
            cutout if self._last_mask is not None else None,
            metrics,
        )
        self._capture_active = False
        self._refresh_sessions()
        self._set_status("Metrics session saved.")

    def _depth_ok(self) -> bool:
        return bool(
            self._last_depth is not None
            and self._last_depth.size > 0
            and np.isfinite(self._last_depth).any()
        )

    def _select_scan_session_id(self) -> str:
        if self._selected_a:
            return str(self._selected_a)
        if self._sessions:
            return str(self._sessions[0].get("session_id") or "")
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _start_scan_prep(self) -> None:
        self._scan_prep_active = True
        self._scan_prep_started_ts = time.time()
        self._scan_prep_remaining_s = int(self._scan_prep_duration_s)
        self._scan_active = False
        self._scan_frames = 0
        self._scan_started_ts = None
        self._scan_last_capture_ts = None
        self._scan_countdown_last = None
        self._scan_countdown_pending = None
        self._scan_points = []
        self._scan_colors = []
        self._set_status("3D scan starts in 5s.")

    def _start_scan4(self) -> None:
        mode = "locked"
        paths = SessionPaths.default()
        scan_dir = paths.new_scan3d_session_dir(user=self.profile_name, mode=mode)
        self._scan4 = FourViewScanCapture(
            self.profile_name,
            scan_dir,
            pose_name=self._last_pose_key or "pose",
            pose_mode=mode,
            rgb_burst_frames=1,
        )
        self._scan4_status = "4-side scan starts in 5s."
        self._scan4_dir = None
        self._scan4_active = False
        self._scan4_prep_active = True
        self._scan4_prep_started_monotonic = time.monotonic()
        self._scan4_prep_countdown_last = int(self._scan4_prep_duration_seconds)
        self._scan4_step = 0
        self._scan4_step_deadline_monotonic = None
        self._scan4_ready_since_monotonic = None
        self._scan4_intro_announced_step = None
        self._scan4_side_ref_sign = None
        self._scan4_countdown_remaining = int(self._scan4_prep_duration_seconds)
        self._scan4_countdown_last = None
        self._scan4_countdown_pending = str(self._scan4_prep_duration_seconds)
        self._scan4_capture_fired = False
        self._scan4_tts_reset_pending = True
        self._scan4_post_capture_cooldown_until = None
        self._scan4_metrics_snapshot = None
        self._set_status("4-side scan starts in 5s.")

    def _scan4_pick_folder(self) -> Optional[Path]:
        sessions_root = SessionPaths.default().root
        initial_dir = sessions_root / self.profile_name
        if not initial_dir.exists():
            initial_dir = sessions_root
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass
            selected = filedialog.askdirectory(
                title="Select existing 4-side scan folder",
                initialdir=str(initial_dir),
            )
            root.destroy()
        except Exception as exc:
            self._set_status(f"Folder picker unavailable: {exc}", seconds=5.0)
            return None
        if not selected:
            return None
        return Path(selected).resolve()

    def _scan4_resolve_session_dir(self, folder: Path) -> Optional[Path]:
        scan_dir = folder
        if folder.name in {"raw", "derived", "reports", "exports", "media"}:
            candidate = folder.parent
            if (candidate / "raw").exists():
                scan_dir = candidate
        if not scan_dir.exists() or not scan_dir.is_dir():
            return None
        raw_dir = scan_dir / "raw"
        if raw_dir.exists() and any(raw_dir.glob("view_*")):
            return scan_dir
        # Fallback: old layouts where view_* lives directly under selected folder.
        if any(scan_dir.glob("view_*")):
            return scan_dir
        return None

    def _load_scan4_from_folder(self) -> None:
        picked = self._scan4_pick_folder()
        if picked is None:
            return
        scan_dir = self._scan4_resolve_session_dir(picked)
        if scan_dir is None:
            self._set_status("Selected folder is not a valid scan session.", seconds=4.5)
            return
        self._scan4 = None
        self._scan4_dir = scan_dir
        self._scan4_active = False
        self._scan4_prep_active = False
        self._scan4_prep_started_monotonic = None
        self._scan4_prep_countdown_last = None
        self._scan4_step = 0
        self._scan4_step_deadline_monotonic = None
        self._scan4_ready_since_monotonic = None
        self._scan4_intro_announced_step = None
        self._scan4_side_ref_sign = None
        self._scan4_countdown_remaining = None
        self._scan4_countdown_last = None
        self._scan4_countdown_pending = None
        self._scan4_capture_fired = False
        self._scan4_tts_reset_pending = False
        self._scan4_post_capture_cooldown_until = None
        self._scan4_status = f"Loaded scan: {scan_dir.name}"
        self._scan4_reload_metrics_snapshot()
        self._set_status(f"Loaded existing scan folder: {scan_dir}", seconds=4.5)

    def pop_scan4_announce(self) -> Optional[str]:
        val = self._scan4_countdown_pending
        self._scan4_countdown_pending = None
        return val

    def pop_scan4_tts_reset(self) -> bool:
        val = self._scan4_tts_reset_pending
        self._scan4_tts_reset_pending = False
        return val

    def is_scan4_active(self) -> bool:
        return bool(self._scan4_active or self._scan4_prep_active)

    def _update_scan4(self) -> None:
        if self._scan4 is None or (not self._scan4_active and not self._scan4_prep_active):
            return
        if self._scan4_prep_active:
            now_mono = time.monotonic()
            if self._scan4_prep_started_monotonic is None:
                self._scan4_prep_started_monotonic = now_mono
            elapsed = now_mono - float(self._scan4_prep_started_monotonic or now_mono)
            remaining = max(0, int(np.ceil(float(self._scan4_prep_duration_seconds) - elapsed)))
            self._scan4_countdown_remaining = remaining
            self._scan4_status = f"4-side scan starts in {remaining}s."
            if remaining != self._scan4_prep_countdown_last and remaining > 0:
                self._scan4_prep_countdown_last = remaining
                self._scan4_countdown_pending = str(remaining)
            if remaining <= 0:
                self._scan4_prep_active = False
                self._scan4_prep_started_monotonic = None
                self._scan4_prep_countdown_last = None
                self._scan4_active = True
                self._scan4_countdown_remaining = None
                self._scan4_countdown_last = None
                self._scan4_step_deadline_monotonic = None
                self._scan4_ready_since_monotonic = None
                self._scan4_intro_announced_step = None
                self._scan4_capture_fired = False
                self._scan4_tts_reset_pending = True
                self._scan4_countdown_pending = "Start"
                self._scan4_status = "Align FRONT. Center and level shoulders."
            return
        if self._last_depth is None or self._last_intrinsics is None:
            self._scan4_status = "Depth/intrinsics missing."
            return
        now_wall = time.time()
        now_mono = time.monotonic()
        checks = self._scan4_pose_checks()
        ready = bool(checks.get("ready"))
        if self._scan4_step >= 4:
            self._scan4_active = False
            self._scan4_dir = self._scan4.scan_dir
            self._scan4_status = "Saved ✅ (4/4)"
            self._scan4_countdown_remaining = None
            self._scan4_countdown_last = None
            self._scan4_step_deadline_monotonic = None
            self._scan4_ready_since_monotonic = None
            self._set_status("Scan complete. Saved all 4 sides.")
            return

        if (
            self._scan4_post_capture_cooldown_until is not None
            and now_mono < self._scan4_post_capture_cooldown_until
        ):
            self._scan4_countdown_remaining = None
            self._scan4_countdown_last = None
            return
        self._scan4_post_capture_cooldown_until = None

        if self._scan4_step > 0 and self._scan4_intro_announced_step != self._scan4_step:
            self._scan4_countdown_pending = "Turn right"
            self._scan4_intro_announced_step = self._scan4_step

        if ready:
            if self._scan4_ready_since_monotonic is None:
                self._scan4_ready_since_monotonic = now_mono
        else:
            self._scan4_ready_since_monotonic = None

        if self._scan4_step_deadline_monotonic is None:
            self._scan4_countdown_remaining = None
            self._scan4_countdown_last = None
            self._scan4_capture_fired = False
            label = self._scan4_step_label()
            if self._scan4_step == 0:
                self._scan4_status = f"Align {label}. Center and level shoulders."
            else:
                self._scan4_status = f"Turn right to {label}. Center and level shoulders."
            if (
                ready
                and self._scan4_ready_since_monotonic is not None
                and (now_mono - self._scan4_ready_since_monotonic) >= float(self._scan4_ready_hold_seconds)
            ):
                self._scan4_step_deadline_monotonic = now_mono + float(self._scan4_countdown_seconds)
                self._scan4_countdown_last = self._scan4_countdown_seconds
                self._scan4_countdown_remaining = self._scan4_countdown_seconds
                self._scan4_capture_fired = False
                self._scan4_countdown_pending = str(self._scan4_countdown_seconds)
                self._scan4_tts_reset_pending = True
                self._scan4_status = f"Freeze… ({self._scan4_step + 1}/4)"
            return

        if not ready:
            self._scan4_step_deadline_monotonic = None
            self._scan4_countdown_remaining = None
            self._scan4_countdown_last = None
            self._scan4_capture_fired = False
            label = self._scan4_step_label()
            if self._scan4_step == 0:
                self._scan4_status = f"Re-align {label}. Countdown paused."
            else:
                self._scan4_status = f"Hold {label}. Countdown paused."
            return

        remaining = max(0, int(np.ceil((self._scan4_step_deadline_monotonic or now_mono) - now_mono)))
        self._scan4_countdown_remaining = remaining
        if remaining != self._scan4_countdown_last:
            self._scan4_countdown_last = remaining
            if remaining > 0:
                self._scan4_countdown_pending = str(remaining)

        if remaining == 0 and not self._scan4_capture_fired:
            step_before_capture = int(self._scan4_step)
            self._scan4_countdown_pending = "Now"
            ok = self._scan4.capture_now(
                depth_m=self._last_depth,
                intrinsics=self._last_intrinsics,
                mask=self._last_mask,
                rgb_bgr=self._last_frame,
                timestamp=now_wall,
            )
            self._scan4_capture_fired = True
            if ok:
                side_sign = checks.get("side_sign")
                if (
                    step_before_capture == 1
                    and side_sign is not None
                    and abs(float(side_sign)) >= 0.015
                ):
                    self._scan4_side_ref_sign = float(side_sign)
                self._scan4_step += 1
                self._scan4_status = f"Captured ✅ ({self._scan4_step}/4)"
                self._set_status(f"Saved view {self._scan4_step}/4.")
                self._scan4_step_deadline_monotonic = None
                self._scan4_ready_since_monotonic = None
                self._scan4_countdown_remaining = None
                self._scan4_countdown_last = None
                self._scan4_post_capture_cooldown_until = now_mono + 0.45
                if self._scan4_step >= 4:
                    self._scan4_active = False
                    self._scan4_dir = self._scan4.scan_dir
                    self._scan4_status = "Saved ✅ (4/4)"
                    self._scan4_countdown_remaining = None
                    self._scan4_step_deadline_monotonic = None
                    self._scan4_ready_since_monotonic = None
                    self._scan4_tts_reset_pending = True
                    self._scan4_countdown_pending = "Scan complete"
                    self._set_status("Scan complete. Saved all 4 sides.")
            else:
                self._scan4_status = "Capture failed… hold still."
                self._scan4_step_deadline_monotonic = now_mono + float(self._scan4_countdown_seconds)
                self._scan4_countdown_last = None
                self._scan4_countdown_remaining = self._scan4_countdown_seconds
                self._scan4_capture_fired = False

    def _scan4_compute_metrics(self) -> bool:
        if not self._scan4_dir:
            self._set_status("No scan to compute.")
            return False
        views = sorted((self._scan4_dir / "raw").glob("view_*"))
        if len(views) < 4:
            views = sorted(self._scan4_dir.glob("view_*"))
        if len(views) < 4:
            self._set_status("Missing views for metrics.")
            return False
        intr = self._last_intrinsics or {}
        meta = None
        for meta_path in (self._scan4_dir / "derived" / "meta.json", self._scan4_dir / "meta.json"):
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                views_meta = meta.get("views") or meta.get("captures") or []
                if views_meta:
                    intr = views_meta[0].get("intrinsics", intr) or intr
                break
            except Exception:
                continue
        masks = []
        depths = []
        points = []
        for v in views[:4]:
            mask_path = next(v.glob("*_mask.png"), None)
            depth_path = next(v.glob("*_depth.npy"), None)
            pcd_path = next(v.glob("*.pcd"), None)
            if not mask_path or not depth_path or not pcd_path:
                self._set_status("Missing debug artifacts for metrics.")
                return False
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            depth = np.load(str(depth_path))
            masks.append(mask)
            depths.append(depth)
            points.append(None)
        metrics = compute_metrics_for_scan(self._scan4_dir, masks, depths, points, intr, meta=meta)
        if metrics:
            self._scan4_metrics_snapshot = self._scan4_snapshot_from_metrics(metrics)
            self._set_status("Metrics computed from 4 sides.")
            return True
        return False

    def _scan4_auto_merge(self) -> None:
        if not self._scan4_dir:
            self._set_status("No scan to merge.")
            return
        if not self._refresh_open3d_status(force=True):
            self._set_status(self._open3d_missing_msg(), seconds=5.5)
            return
        res = auto_merge(self._scan4_dir)
        if not res.ok and "Open3D not available" in (res.message or ""):
            self._set_status(self._open3d_missing_msg(), seconds=5.5)
            return
        self._set_status(res.message)

    def _scan4_manual_align(self) -> None:
        if not self._scan4_dir:
            self._set_status("No scan to align.")
            return
        if not self._refresh_open3d_status(force=True):
            self._set_status(self._open3d_missing_msg(), seconds=5.5)
            return
        try:
            res = manual_align(self._scan4_dir)
            self._set_status(res.message)
        except Exception as exc:
            raw_msg = str(exc)
            if "Open3D not available" in raw_msg:
                self._set_status(self._open3d_missing_msg(), seconds=5.5)
                return
            self._set_status(raw_msg)

    def _scan4_export(self) -> None:
        if not self._scan4_dir:
            self._set_status("No scan to export.")
            return
        metrics_path = self._scan4_dir / "derived" / "metrics.json"
        if not metrics_path.exists():
            if not self._scan4_compute_metrics():
                return
        root = Path(__file__).resolve().parents[3]
        script_path = root / "scripts" / "make_report.py"
        if not script_path.exists():
            self._set_status("Report generator not found.")
            return
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path), "--session_dir", str(self._scan4_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as exc:
            self._set_status(f"Export failed: {exc}")
            return
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            if not detail:
                detail = "unknown error"
            if len(detail) > 150:
                detail = detail[:147] + "..."
            self._set_status(f"Export failed: {detail}", seconds=5.5)
            return
        self._scan4_reload_metrics_snapshot()
        report_full = self._scan4_dir / "reports" / "report_full.pdf"
        if report_full.exists():
            self._set_status(f"Export complete: {report_full}")
        else:
            self._set_status(f"Export complete: {self._scan4_dir / 'reports'}")

    def pop_scan_countdown(self) -> Optional[int]:
        val = self._scan_countdown_pending
        self._scan_countdown_pending = None
        return val

    def _update_countdown(self, remaining: int) -> None:
        if remaining != self._scan_countdown_last:
            self._scan_countdown_last = remaining
            self._scan_countdown_pending = remaining

    def _update_scan(self) -> None:
        if self._scan_prep_active:
            now = time.time()
            if self._scan_prep_started_ts is None:
                self._scan_prep_started_ts = now
            elapsed = now - self._scan_prep_started_ts
            remaining = max(0, int(np.ceil(self._scan_prep_duration_s - elapsed)))
            self._scan_prep_remaining_s = remaining
            self._update_countdown(remaining)
            if remaining <= 0:
                self._scan_prep_active = False
                self._scan_prep_started_ts = None
                self._scan_prep_remaining_s = None
                self._scan_active = True
                self._scan_started_ts = now
                self._scan_last_capture_ts = None
                self._scan_countdown_last = None
            return

        if not self._scan_active:
            return
        if not self._depth_ok():
            self._scan_active = False
            self._set_status("Depth not available for scan.")
            return
        if self._last_mask is None:
            self._scan_active = False
            self._set_status("No mask available for scan.")
            return
        if not self._last_intrinsics:
            self._scan_active = False
            self._set_status("Missing camera intrinsics.")
            return
        now = time.time()
        if self._scan_started_ts is None:
            self._scan_started_ts = now
        if self._scan_last_capture_ts is not None:
            if now - self._scan_last_capture_ts < (1.0 / max(1.0, self._scan_capture_hz)):
                return
        self._scan_last_capture_ts = now
        pts, cols = depth_to_pointcloud(
            self._last_depth,
            self._last_intrinsics,
            mask=self._last_mask,
            rgb_bgr=self._last_frame if self._depth_aligned else None,
            stride=3,
        )
        if pts.size == 0:
            return
        self._scan_points.append(pts)
        if cols is not None:
            self._scan_colors.append(cols)
        self._scan_frames += 1
        elapsed = now - (self._scan_started_ts or now)
        remaining = max(0, int(np.ceil(self._scan_duration_s - elapsed)))
        self._scan_remaining_s = remaining
        self._update_countdown(remaining)

        if elapsed >= self._scan_duration_s and self._scan_frames >= self._scan_min_frames:
            points = np.concatenate(self._scan_points, axis=0)
            colors = np.concatenate(self._scan_colors, axis=0) if self._scan_colors else None
            self._last_scan = (points, colors)
            self._scan_active = False
            self._scan_started_ts = None
            self._scan_last_capture_ts = None
            self._set_status("3D scan captured.")

    def _export_scan(self) -> None:
        if not self._last_scan:
            if not self._depth_ok() or not self._last_intrinsics:
                self._set_status("Depth not available.")
                return
            if self._last_mask is None:
                self._set_status("No mask available for scan.")
                return
            pts, cols = depth_to_pointcloud(
                self._last_depth,
                self._last_intrinsics,
                mask=self._last_mask,
                rgb_bgr=self._last_frame if self._depth_aligned else None,
                stride=2,
            )
            if pts.size == 0:
                self._set_status("No points to export.")
                return
        else:
            pts, cols = self._last_scan

        session_id = self._select_scan_session_id()
        paths = self.store.save_scan(self.profile_name, session_id, pts, cols)
        if paths:
            self._set_status(f"Saved scan: {paths.get('ply') or paths.get('pcd')}")

    def _cycle_pose_filter(self) -> None:
        if not self._pose_filter_options:
            self._pose_filter = None
            return
        if self._pose_filter not in self._pose_filter_options:
            self._pose_filter = None
            return
        idx = self._pose_filter_options.index(self._pose_filter)
        self._pose_filter = self._pose_filter_options[(idx + 1) % len(self._pose_filter_options)]

    def _select_session(self, session_id: str) -> None:
        if self._selected_a is None or (self._selected_a and self._selected_b):
            self._selected_a = session_id
            self._selected_b = None
            return
        if self._selected_a == session_id:
            return
        if self._selected_b is None:
            self._selected_b = session_id

    def _session_by_id(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        for s in self._sessions:
            if str(s.get("session_id")) == str(session_id):
                return s
        return None

    def handle_click(self, pt: Tuple[int, int]) -> None:
        if self._input_mode:
            return
        for btn in self._buttons:
            if _in_rect(pt, btn.rect):
                if btn.key == "session_toggle":
                    if self._capture_active:
                        self._stop_capture()
                    else:
                        self._start_capture()
                elif btn.key == "refresh":
                    self._refresh_sessions()
                elif btn.key == "filter_pose":
                    self._cycle_pose_filter()
                elif btn.key == "clear_selection":
                    self._selected_a = None
                    self._selected_b = None
                elif btn.key == "federation":
                    self._cycle_federation()
                elif btn.key == "classification":
                    self._cycle_category()
                elif btn.key == "suggest_classification":
                    metrics = compute_body_proxy(self._last_mask, self.height_cm or 0.0, self.weight_kg, self._last_props)
                    if not metrics:
                        self._set_status("No proxy metrics available for suggestion.")
                    else:
                        self._suggest_category(metrics)
                elif btn.key == "scan4_guide":
                    self._scan4_guide_enabled = not self._scan4_guide_enabled
                    self._set_status(f"Scan guide {'on' if self._scan4_guide_enabled else 'off'}.")
                elif btn.key == "scan_4v":
                    if not self._depth_ok():
                        self._set_status("Depth not available.")
                    else:
                        self._start_scan4()
                elif btn.key == "scan4_load":
                    self._load_scan4_from_folder()
                elif btn.key == "scan4_metrics":
                    self._scan4_compute_metrics()
                elif btn.key == "scan4_merge":
                    self._scan4_auto_merge()
                elif btn.key == "scan4_align":
                    self._scan4_manual_align()
                elif btn.key == "scan4_export":
                    self._scan4_export()
                elif btn.key == "ui_metrics":
                    self._set_ui_mode("metrics")
                elif btn.key == "browse_rules":
                    self._set_ui_mode("browse_rules")
                elif btn.key == "select_categories":
                    self._set_ui_mode("select_categories")
                elif btn.key == "comparison_table":
                    self._set_ui_mode("comparison_table")
                return
        if self._ui_mode == "browse_rules":
            for rect, ref in self._browse_tree_hits:
                if _in_rect(pt, rect):
                    self._browse_ref = ref
                    return
        if self._ui_mode == "select_categories":
            for rect, ref, is_eligible, reasons in self._selection_hits:
                if not _in_rect(pt, rect):
                    continue
                for idx, row in enumerate(self._selection_rows(self._active_athlete_profile(None))):
                    if row["ref"].key() == ref.key():
                        self._selection_cursor_index = idx
                        break
                if is_eligible:
                    self._toggle_selected_ref(ref)
                    self._set_status("Selected divisions updated.")
                else:
                    if reasons:
                        self._set_status(reasons[0], seconds=5.0)
                    else:
                        self._set_status("Ineligible for selected item.", seconds=4.0)
                return
        for rect, session_id in self._session_hits:
            if _in_rect(pt, rect):
                self._select_session(session_id)
                return

    def handle_key(self, key: int) -> None:
        if key <= 0:
            return
        if self._input_mode:
            self._handle_input_key(key)
            return
        if key in (ord('m'), ord('M')):
            self._set_ui_mode("metrics")
        if key in (ord('b'), ord('B')):
            self._set_ui_mode("browse_rules")
        if key in (ord('e'), ord('E')):
            self._set_ui_mode("select_categories")
        if key in (ord('t'), ord('T')):
            self._set_ui_mode("comparison_table")
        if self._ui_mode == "comparison_table":
            if key == ord('['):
                self._comparison_scroll_chars = max(0, self._comparison_scroll_chars - 8)
            if key == ord(']'):
                self._comparison_scroll_chars += 8
        if self._ui_mode == "select_categories":
            if key in (ord('u'), ord('U')):
                self._selection_cursor_index = max(0, self._selection_cursor_index - 1)
            if key in (ord('d'), ord('D')):
                self._selection_cursor_index += 1
            if key in (10, 13):
                rows = self._selection_rows(self._active_athlete_profile(None))
                if rows:
                    self._selection_cursor_index = int(np.clip(self._selection_cursor_index, 0, len(rows) - 1))
                    row = rows[self._selection_cursor_index]
                    result = row["result"]
                    ref = row["ref"]
                    if result.eligible:
                        self._toggle_selected_ref(ref)
                        self._set_status("Selected divisions updated.")
                    elif result.reasons:
                        self._set_status(result.reasons[0], seconds=5.0)
        if key in (ord('s'), ord('S')):
            if self._capture_active:
                self._stop_capture()
            else:
                self._start_capture()
        if key in (ord('r'), ord('R')):
            self._refresh_sessions()
        if key in (ord('f'), ord('F')):
            self._cycle_federation()
        if key in (ord('c'), ord('C')):
            self._cycle_category()
        if key in (ord('j'), ord('J')):
            metrics = compute_body_proxy(self._last_mask, self.height_cm or 0.0, self.weight_kg, self._last_props)
            if not metrics:
                self._set_status("No proxy metrics available for suggestion.")
            else:
                self._suggest_category(metrics)
        if key in (ord('g'), ord('G')):
            self._scan4_guide_enabled = not self._scan4_guide_enabled
            self._set_status(f"Scan guide {'on' if self._scan4_guide_enabled else 'off'}.")
        if key in (ord('l'), ord('L')):
            self._load_scan4_from_folder()

    def handle_voice_command(self, cmd: str) -> None:
        if cmd in ("scan_3d", "start_scan"):
            if self._depth_ok():
                self._start_scan4()
            else:
                self._start_capture()
        elif cmd == "stop_scan":
            self._stop_capture()
        elif cmd == "save_model":
            if not self._depth_ok():
                self._set_status("Depth not available.")
            else:
                self._scan4_export()

    def _handle_input_key(self, key: int) -> None:
        if key in (8, 127):  # backspace
            self._input_value = self._input_value[:-1]
            return
        if key in (10, 13):  # enter
            if self._input_mode == "height":
                val = self._safe_float(self._input_value)
                if val is None or val <= 0:
                    self._input_error = "Height required."
                    return
                self.height_cm = val
                self._input_mode = "weight"
                self._input_value = ""
                self._input_error = ""
                return
            if self._input_mode == "weight":
                val = self._safe_float(self._input_value)
                self.weight_kg = val if val and val > 0 else None
                self._input_mode = None
                self._input_value = ""
                self._input_error = ""
                self._save_body_profile()
                self._set_status("Body profile saved.")
                return
        if key == 27:  # escape
            if self._input_mode == "weight":
                self.weight_kg = None
                self._input_mode = None
                self._input_value = ""
                self._input_error = ""
                self._save_body_profile()
                return
            self._input_value = ""
            return
        if 48 <= key <= 57 or key == ord('.'):
            self._input_value += chr(key)

    def _safe_float(self, val: Any) -> Optional[float]:
        if val is None or val == "":
            return None
        try:
            return float(val)
        except Exception:
            return None

    def _scan4_pose_checks(self) -> Dict[str, Any]:
        # Heuristics tuned for normalized landmark coordinates [0..1].
        front_min = 0.13
        side_max = 0.145
        front_mask_min = 0.16
        side_mask_max = 0.18
        side_opp_eps = 0.008
        side_abs_min = 0.012
        expected_step = int(np.clip(self._scan4_step, 0, 3))
        expected_front = expected_step in (0, 2)
        expected_side = expected_step in (1, 3)

        shoulder_width: Optional[float] = None
        shoulders_level: Optional[bool] = None
        nose_between_shoulders: Optional[bool] = None
        side_sign: Optional[float] = None
        if self._last_landmarks:
            ls = self._last_landmarks.get("left_shoulder")
            rs = self._last_landmarks.get("right_shoulder")
            nose = self._last_landmarks.get("nose")
            if ls and rs:
                lx = float(ls[0])
                ly = float(ls[1])
                rx = float(rs[0])
                ry = float(rs[1])
                shoulder_width = abs(rx - lx)
                shoulders_level = abs(ly - ry) <= 0.07
                if nose:
                    nx = float(nose[0])
                    mid = 0.5 * (lx + rx)
                    side_sign = nx - mid
                    minx = min(lx, rx)
                    maxx = max(lx, rx)
                    nose_between_shoulders = minx <= nx <= maxx

        center_x_norm = self._mask_center_x_norm(self._last_mask)
        centered_ok: Optional[bool] = None
        if center_x_norm is not None:
            centered_ok = 0.40 <= center_x_norm <= 0.60
        mask_width_norm: Optional[float] = None
        if self._last_mask is not None and self._last_mask.size > 0:
            ys, xs = np.where(self._last_mask > 0)
            if xs.size > 1 and self._last_mask.shape[1] > 0:
                mask_width_norm = float((xs.max() - xs.min()) / float(self._last_mask.shape[1]))

        def _merge_orientation_checks(
            primary: Optional[bool],
            secondary: Optional[bool],
            *,
            allow_either: bool,
        ) -> Optional[bool]:
            vals = [v for v in (primary, secondary) if v is not None]
            if not vals:
                return None
            if len(vals) == 1:
                return bool(vals[0])
            if allow_either:
                return bool(vals[0] or vals[1])
            return bool(vals[0] and vals[1])

        orientation_ok: Optional[bool] = None
        if expected_front:
            orient_lm: Optional[bool] = None
            orient_mask: Optional[bool] = None
            if shoulder_width is not None:
                orient_lm = shoulder_width >= front_min
                if expected_step == 0:
                    if nose_between_shoulders is not None:
                        orient_lm = orient_lm and bool(nose_between_shoulders)
            if mask_width_norm is not None:
                orient_mask = mask_width_norm >= front_mask_min
            orientation_ok = _merge_orientation_checks(
                orient_lm,
                orient_mask,
                allow_either=(expected_step in (0, 2)),
            )
        elif expected_side:
            orient_lm = None
            orient_mask = None
            if shoulder_width is not None:
                orient_lm = shoulder_width <= side_max
                if side_sign is not None:
                    orient_lm = orient_lm and (abs(float(side_sign)) >= side_abs_min)
                if orient_lm and expected_step == 3 and self._scan4_side_ref_sign is not None and side_sign is not None:
                    orient_lm = (float(side_sign) * self._scan4_side_ref_sign) <= -side_opp_eps
            if mask_width_norm is not None:
                orient_mask = mask_width_norm <= side_mask_max
            orientation_ok = _merge_orientation_checks(
                orient_lm,
                orient_mask,
                allow_either=True,
            )

        ready = (
            orientation_ok is True
            and centered_ok is not False
        )
        return {
            "ready": ready,
            "orientation_ok": orientation_ok,
            "centered_ok": centered_ok,
            "shoulders_level": shoulders_level,
            "shoulder_width": shoulder_width,
            "mask_width_norm": mask_width_norm,
            "side_sign": side_sign,
            "expected_front": expected_front,
        }

    def _scan4_step_label(self) -> str:
        labels = ("FRONT", "RIGHT SIDE", "BACK", "LEFT SIDE")
        idx = int(np.clip(self._scan4_step, 0, len(labels) - 1))
        return labels[idx]

    def _mask_center_x_norm(self, mask: Optional[np.ndarray]) -> Optional[float]:
        if mask is None or mask.size == 0:
            return None
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        width = float(mask.shape[1]) if mask.shape[1] > 0 else 0.0
        if width <= 0:
            return None
        return float((xs.min() + xs.max()) * 0.5 / width)

    def _draw_scan4_guide(self, out: np.ndarray) -> None:
        if not self._scan4_active or not self._scan4_guide_enabled:
            return
        h, w = out.shape[:2]
        cx = w // 2
        cy = h // 2
        cv2.line(out, (cx, int(h * 0.14)), (cx, int(h * 0.92)), (80, 180, 220), 1, cv2.LINE_AA)
        cv2.line(out, (int(w * 0.34), cy), (int(w * 0.66), cy), (80, 180, 220), 1, cv2.LINE_AA)

        checks = self._scan4_pose_checks()
        orient_ok: Optional[bool] = checks.get("orientation_ok")
        centered_ok: Optional[bool] = checks.get("centered_ok")
        shoulders_level: Optional[bool] = checks.get("shoulders_level")

        panel_w = min(380, max(280, int(w * 0.28)))
        panel_h = 130
        panel_x = w - panel_w - 12
        panel_y = 86
        overlay = out.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), _TEXT_BG, -1)
        cv2.addWeighted(overlay, 0.62, out, 0.38, 0, out)

        line_y = panel_y + 24
        _put_text(out, "SCAN 3D GUIDE", line_y, scale=0.52, colour=_TEXT_ACCENT, thickness=1, x=panel_x + 10)
        line_y += 22
        _put_text(
            out,
            f"STEP {self._scan4_step + 1}/4  {self._scan4_step_label()}",
            line_y,
            scale=0.47,
            thickness=1,
            x=panel_x + 10,
        )
        line_y += 20
        orient_txt = "OK" if orient_ok is True else ("ADJUST" if orient_ok is False else "LOCK")
        orient_target = self._scan4_step_label().lower()
        _put_text(out, f"ORIENTATION {orient_txt} ({orient_target})", line_y, scale=0.44, thickness=1, x=panel_x + 10)
        line_y += 18
        center_txt = "OK" if centered_ok is True else ("ADJUST" if centered_ok is False else "LOCK")
        _put_text(out, f"CENTERING {center_txt}", line_y, scale=0.44, thickness=1, x=panel_x + 10)
        line_y += 18
        level_txt = "OK" if shoulders_level is True else ("WARN" if shoulders_level is False else "LOCK")
        _put_text(out, f"SHOULDERS {level_txt}", line_y, scale=0.44, thickness=1, x=panel_x + 10)

    def render_scan_overlay(self, frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Lightweight overlay for desktop GUI scan tab (without legacy in-frame menus)."""
        out = frame.copy()
        if mask is not None:
            out = draw_mask_outline(out, mask)
        self._draw_scan4_guide(out)
        if self._scan4_countdown_remaining is not None and (self._scan4_active or self._scan4_prep_active):
            _draw_countdown(out, self._scan4_countdown_remaining)
        if self._scan4_status:
            _put_text(out, self._scan4_status, y=36, scale=0.5, colour=(255, 220, 170), thickness=1, x=12)
        return out

    def render(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray],
        props: Optional[object],
        profile: Any,
        category: str,
        pose_key: str,
        landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict[str, Any]] = None,
        depth_aligned: Optional[bool] = None,
        depth_units: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        self._last_frame = frame
        self._last_mask = mask
        self._last_props = props
        self._last_pose_key = pose_key
        self._last_landmarks = landmarks
        self._last_depth = depth
        self._last_intrinsics = intrinsics
        self._depth_aligned = bool(depth_aligned) if depth_aligned is not None else False
        self._depth_units = depth_units
        self._depth_available = bool(
            depth is not None and depth.size > 0 and np.isfinite(depth).any()
        )
        profile_fed = str(getattr(profile, "federation", "") or "")
        self._set_context_defaults(profile_fed, category)

        self._ensure_body_profile()
        if time.time() - self._last_refresh > 3.0:
            self._refresh_sessions()
        self._update_scan()
        self._update_scan4()
        athlete_profile = self._active_athlete_profile(profile)
        state_selected = list(self._app_state.state.selected_divisions)
        if [ref.key() for ref in state_selected] != [ref.key() for ref in self._selection_refs]:
            self._selection_refs = state_selected

        display = frame.copy()
        if mask is not None:
            display = draw_mask_outline(display, mask)

        if target_size:
            display, _ = _resize_cover(display, target_size[0], target_size[1])

        out = display

        y = 28
        _put_text(out, "BODY METRICS", y, scale=0.7, colour=_TEXT_ACCENT, thickness=2)
        y += 30
        height_txt = _fmt_num(self.height_cm, 1) if self.height_cm else "—"
        weight_txt = _fmt_num(self.weight_kg, 1) if self.weight_kg else "—"
        fed = self._last_federation or profile_fed
        class_label = self._last_category or category
        _put_text(out, f"PROFILE {self.profile_name} | FED {fed} | CLASS {class_label}", y, scale=0.52, thickness=1)
        y += 22
        _put_text(out, f"HEIGHT {height_txt} CM | WEIGHT {weight_txt} KG", y, scale=0.52, thickness=1)
        y += 22
        if self._capture_active:
            _put_text(out, "SESSION: RECORDING (PRESS STOP)", y, scale=0.5, colour=(255, 190, 120), thickness=1)
            y += 22
        if self._status_msg and time.time() < self._status_until:
            _put_text(out, self._status_msg.upper(), y, scale=0.5, colour=(180, 230, 180), thickness=1)
            y += 22

        metrics = compute_body_proxy(mask, self.height_cm or 0.0, self.weight_kg, props)
        if self._scan4_metrics_snapshot and self._scan4_dir and not self._scan4_active:
            widths = self._scan4_metrics_snapshot.get("key_widths_cm") or {}
            ratios = self._scan4_metrics_snapshot.get("ratios") or {}
            _put_text(out, "SCAN4 METRICS (SAVED)", y, scale=0.5, colour=_TEXT_ACCENT, thickness=1)
            y += 22
            for key in ("shoulders", "chest", "waist", "hips", "thigh", "calf"):
                if key not in widths:
                    continue
                _put_text(out, f"{key.upper()}  {_fmt_num(widths.get(key), 1)} CM", y, scale=0.48, thickness=1)
                y += 20
            sw = _fmt_num(ratios.get("shoulder_to_waist"), 3)
            cw = _fmt_num(ratios.get("chest_to_waist"), 3)
            hw = _fmt_num(ratios.get("hip_to_waist"), 3)
            tw = _fmt_num(ratios.get("thigh_to_waist"), 3)
            _put_text(out, f"S/W {sw} | C/W {cw} | H/W {hw} | T/W {tw}", y, scale=0.48, thickness=1)
            y += 20
        else:
            regions = metrics.get("proxy_regions", {}) if metrics else {}
            _put_text(out, "LIVE PROXIES", y, scale=0.5, colour=_TEXT_ACCENT, thickness=1)
            y += 22
            for key in ("upper_torso", "waist", "hips", "thighs", "calves"):
                if key not in regions:
                    continue
                width_cm = _fmt_num(regions[key].get("width_cm"), 1)
                ratio = _fmt_num(regions[key].get("ratio"), 3)
                _put_text(out, f"{key.upper()}  {width_cm} CM | RATIO {ratio}", y, scale=0.48, thickness=1)
                y += 20

            if metrics and metrics.get("proportions"):
                props_out = metrics.get("proportions") or {}
                sw = _fmt_num(props_out.get("shoulder_to_waist"), 2)
                cw = _fmt_num(props_out.get("chest_to_waist"), 2)
                hw = _fmt_num(props_out.get("hip_to_waist"), 2)
                ul = _fmt_num(props_out.get("upper_to_lower_area"), 2)
                _put_text(out, f"S/W {sw} | C/W {cw} | H/W {hw} | U/L {ul}", y, scale=0.48, thickness=1)
                y += 20
        if self._category_suggestion:
            _put_text(
                out,
                f"SUGGESTED CLASS {self._category_suggestion.upper()}",
                y,
                scale=0.47,
                colour=(180, 230, 180),
                thickness=1,
            )
            y += 20
            if self._category_suggestion_reason:
                _put_text(out, self._category_suggestion_reason, y, scale=0.45, thickness=1)
                y += 20

        h, w = out.shape[:2]
        if self._ui_mode == "browse_rules":
            self._session_hits = []
            self._render_browse_rules(out, athlete_profile, y_start=28)
        elif self._ui_mode == "select_categories":
            self._session_hits = []
            self._render_select_categories(out, athlete_profile, y_start=28)
        elif self._ui_mode == "comparison_table":
            self._session_hits = []
            self._render_comparison_table(out, athlete_profile, y_start=28)
        else:
            # Right column: sessions + compare
            right_x = int(w * 0.56)
            right_y = 28
            _put_text(out, "SESSIONS", right_y, scale=0.55, colour=_TEXT_ACCENT, thickness=1, x=right_x)
            right_y += 24

            self._session_hits = []
            sessions = self._filtered_sessions()
            for idx, session in enumerate(sessions[:10]):
                sid = str(session.get("session_id", ""))
                created = str(session.get("created_at", "")).replace("T", " ")[:16]
                pose_id = str(session.get("pose_id") or "-")
                scan_count = 0
                try:
                    scan_count = len(session.get("scans") or [])
                except Exception:
                    scan_count = 0
                mark = " "
                if self._selected_a == sid:
                    mark = "A"
                elif self._selected_b == sid:
                    mark = "B"
                scan_tag = f" | scans {scan_count}" if scan_count else ""
                line = f"[{mark}] {created} | {pose_id}{scan_tag}"
                _put_text(out, line, right_y, scale=0.48, thickness=1, x=right_x)
                (tw, th), _ = cv2.getTextSize(line, _TEXT_FONT, 0.48, 1)
                rect = (right_x, right_y - th - 4, tw + 6, th + 10)
                self._session_hits.append((rect, sid))
                right_y += 20

            # Compare section
            right_y += 6
            _put_text(out, "COMPARE (B - A)", right_y, scale=0.5, colour=_TEXT_ACCENT, thickness=1, x=right_x)
            right_y += 20
            session_a = self._session_by_id(self._selected_a)
            session_b = self._session_by_id(self._selected_b)
            metrics_a = self.store.load_metrics(session_a.get("metrics_path")) if session_a else None
            metrics_b = self.store.load_metrics(session_b.get("metrics_path")) if session_b else None
            if metrics_a and metrics_b:
                comp = compare_metrics(metrics_a, metrics_b)
                deltas = comp.get("region_deltas", {})
                deltas_vals = []
                for key in ("upper_torso", "waist", "hips", "thighs", "calves"):
                    if key not in deltas:
                        continue
                    delta_cm = deltas[key].get("delta_cm")
                    if delta_cm is not None:
                        deltas_vals.append(float(delta_cm))
                    _put_text(
                        out,
                        f"{key.upper()} { _fmt_num(delta_cm, 2) } CM",
                        right_y,
                        scale=0.47,
                        thickness=1,
                        x=right_x,
                    )
                    right_y += 18
                if deltas_vals:
                    global_delta = sum(deltas_vals) / float(len(deltas_vals))
                    _put_text(
                        out,
                        f"GLOBAL { _fmt_num(global_delta, 2) } CM",
                        right_y,
                        scale=0.47,
                        thickness=1,
                        x=right_x,
                    )
                    right_y += 18

                selected_labels = selected_division_labels(self._spec_library, self._selection_refs)
                _put_text(out, "RELAXED PHYSIQUE AUDIT", right_y, scale=0.5, colour=_TEXT_ACCENT, thickness=1, x=right_x)
                right_y += 18
                if selected_labels:
                    _put_text(out, "Selected divisions:", right_y, scale=0.44, thickness=1, x=right_x)
                    right_y += 16
                    for label in selected_labels[:4]:
                        _put_text(out, "- " + label, right_y, scale=0.4, thickness=1, x=right_x)
                        right_y += 15
                else:
                    _put_text(out, "Selected divisions: none", right_y, scale=0.44, thickness=1, x=right_x)
                    right_y += 16

                if athlete_profile.bodyfat_percent is not None:
                    _put_text(
                        out,
                        f"Body fat entered by user: {athlete_profile.bodyfat_percent:.1f}% (user-provided)",
                        right_y,
                        scale=0.41,
                        thickness=1,
                        x=right_x,
                    )
                    right_y += 16

                cfg = load_federation_config(str(self._last_federation or getattr(profile, "federation", "ukbff")))
                if cfg:
                    score = score_metrics(metrics_b, cfg)
                    summary = score.get("summary")
                    missing = score.get("missing_areas") or []
                    if summary:
                        _put_text(out, summary, right_y, scale=0.47, thickness=1, x=right_x)
                        right_y += 18
                    if missing:
                        _put_text(out, "Missing: " + ", ".join(missing), right_y, scale=0.47, thickness=1, x=right_x)
                        right_y += 18

                if self._coaching_targets.target_bodyfat_range or self._coaching_targets.safe_weekly_weight_change_range:
                    _put_text(
                        out,
                        "Coaching targets (not official federation rules):",
                        right_y,
                        scale=0.42,
                        colour=(255, 190, 120),
                        thickness=1,
                        x=right_x,
                    )
                    right_y += 16
                    safe = self._coaching_targets.safe_weekly_weight_change_range
                    if safe:
                        safe_line = (
                            f"Safe weekly weight change: {safe.get('min_percent_bodyweight', 'null')}% to "
                            f"{safe.get('max_percent_bodyweight', 'null')}% bodyweight/week"
                        )
                        _put_text(out, safe_line, right_y, scale=0.39, thickness=1, x=right_x)
                        right_y += 15
            else:
                _put_text(out, "Select two sessions.", right_y, scale=0.47, thickness=1, x=right_x)

        # Buttons
        self._refresh_open3d_status(force=False)
        self._buttons = []
        btn_h = 26
        btn_gap = 8
        btn_y = h - 34
        btn_y2 = btn_y - (btn_h + 8)
        btn_x = 12
        btn_x2 = 12

        def add_btn(key: str, label: str, active: bool = False, disabled: bool = False) -> None:
            nonlocal btn_x
            (tw, _), _meta = cv2.getTextSize(label, _TEXT_FONT, 0.5, 1)
            w_btn = tw + 18
            self._buttons.append(
                UIButton(key=key, label=label, rect=(btn_x, btn_y, w_btn, btn_h), active=active, disabled=disabled)
            )
            btn_x += w_btn + btn_gap

        def add_btn_row2(key: str, label: str, active: bool = False, disabled: bool = False) -> None:
            nonlocal btn_x2
            (tw, _), _meta = cv2.getTextSize(label, _TEXT_FONT, 0.5, 1)
            w_btn = tw + 18
            self._buttons.append(
                UIButton(key=key, label=label, rect=(btn_x2, btn_y2, w_btn, btn_h), active=active, disabled=disabled)
            )
            btn_x2 += w_btn + btn_gap

        def short_label(text: str, max_len: int) -> str:
            if len(text) <= max_len:
                return text
            if max_len <= 3:
                return text[:max_len]
            return text[: max_len - 3] + "..."

        session_label = "STOP SESSION" if self._capture_active else "START SESSION"
        add_btn("session_toggle", session_label, active=self._capture_active)
        add_btn("refresh", "REFRESH")
        pose_label = f"POSE {self._pose_filter or 'ALL'}"
        add_btn("filter_pose", pose_label)
        add_btn("clear_selection", "CLEAR A/B")
        # Legacy scan_3d removed in favor of 4-side scan.
        add_btn("scan_4v", "SCAN 4 SIDES", active=self._scan4 is not None and (self._scan4_dir is None))
        add_btn("scan4_load", "LOAD SCAN DIR")
        if self._scan4_dir:
            add_btn("scan4_metrics", "COMPUTE METRICS")
            add_btn("scan4_merge", "AUTO MERGE", disabled=not self._open3d_available)
            add_btn("scan4_align", "MANUAL ALIGN", disabled=not self._open3d_available)
            add_btn("scan4_export", "EXPORT")

        fed_label = self._last_federation or "WNBF_UK"
        class_label = short_label(self._last_category or "Mens Physique", 14)
        add_btn_row2("federation", f"FED {fed_label}")
        add_btn_row2("classification", f"CLASS {class_label}")
        add_btn_row2("suggest_classification", "SUGGEST CLASS")
        guide_label = "ON" if self._scan4_guide_enabled else "OFF"
        add_btn_row2("scan4_guide", f"SCAN GUIDE {guide_label}", active=self._scan4_guide_enabled)
        add_btn_row2("ui_metrics", "METRICS", active=self._ui_mode == "metrics")
        add_btn_row2("browse_rules", "BROWSE RULES", active=self._ui_mode == "browse_rules")
        add_btn_row2("select_categories", "SELECT CATEGORIES", active=self._ui_mode == "select_categories")
        add_btn_row2("comparison_table", "COMPARISON TABLE", active=self._ui_mode == "comparison_table")

        for btn in self._buttons:
            _draw_button(out, btn)

        if self._input_mode:
            overlay = out.copy()
            cv2.rectangle(overlay, (40, 60), (w - 40, 200), _TEXT_BG, -1)
            cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
            prompt = "Enter height (cm):" if self._input_mode == "height" else "Enter weight (kg, optional):"
            _put_text(out, prompt, 110, scale=0.6, colour=_TEXT_ACCENT, thickness=1, x=60, bg=False)
            _put_text(out, self._input_value or "_", 150, scale=0.7, colour=_TEXT_COLOUR, thickness=2, x=60, bg=False)
            if self._input_error:
                _put_text(out, self._input_error, 185, scale=0.5, colour=(255, 180, 120), thickness=1, x=60, bg=False)
        self._draw_scan4_guide(out)
        if self._scan4_countdown_remaining is not None and self._scan4_active:
            _draw_countdown(out, self._scan4_countdown_remaining)
        if self._scan4_status:
            _put_text(out, self._scan4_status, y=72, scale=0.5, colour=(255, 220, 170), thickness=1, x=12)

        return out
