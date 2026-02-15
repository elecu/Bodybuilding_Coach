from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import os
import re
from typing import Optional
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from .. import __version__
from ..app_state import AppStateStore, ApplicationState, get_global_app_state
from ..athlete_profiles import (
    AthleteProfile,
    AthleteProfileStore,
    SEX_CATEGORIES,
    TURN_AGES,
    build_demo_profile_name,
    build_age_eligibility_snapshot,
    calculate_age_years,
    calculate_competition_countdown,
)
from ..device_manager import DeviceManager, DeviceSnapshot, DeviceState, InputSource
from ..federations.eligibility import evaluate_division_eligibility, matches_age_window
from ..federations.pose_checklist import PoseChecklistItem, build_pose_checklist, selected_division_labels
from ..federations.specs import (
    DivisionSpec,
    FederationSpec,
    SelectedDivisionRef,
    load_federation_specs,
)
from ..metrics.proportions import compute_from_mask
from ..poses.library import POSE_GUIDES, POSES
from ..poses.scoring import score_pose
from ..storage.session import SessionStore
from ..ui.tabs import MetricsTab
from ..vision.overlay import draw_mask_outline, draw_pose_guide, draw_pose_overlay
from ..vision.pose import PoseBackend, PoseResult
from ..voice.commands import VoiceCommandConfig, VoiceCommandListener
from ..voice.tts import TTSSpeaker


def _bool_to_yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _yes_no_to_bool(value: str) -> bool:
    return value.strip().lower() == "yes"


def _parse_positive_float(value: str, field_name: str) -> float:
    text = (value or "").strip()
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return parsed


def _parse_optional_float(value: str, field_name: str) -> Optional[float]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _parse_optional_iso_date(value: str, field_name: str) -> Optional[str]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD.") from exc
    return text


def _parse_required_iso_date(value: str, field_name: str) -> str:
    parsed = _parse_optional_iso_date(value, field_name)
    if parsed is None:
        raise ValueError(f"{field_name} is required.")
    return parsed


class ProfileEditorDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, title: str, profile: Optional[AthleteProfile] = None) -> None:
        self.profile = profile
        self.result: Optional[AthleteProfile] = None
        super().__init__(parent, title)

    def body(self, master: tk.Misc) -> tk.Widget:
        profile = self.profile
        self.profile_name_var = tk.StringVar(value=profile.profile_name if profile else "")
        self.sex_category_var = tk.StringVar(value=profile.sex_category if profile else SEX_CATEGORIES[0])
        self.dob_var = tk.StringVar(value=(profile.date_of_birth or "") if profile else "")
        self.height_var = tk.StringVar(value=str(profile.height_cm) if profile else "175")
        self.weight_var = tk.StringVar(value=str(profile.weight_kg) if profile else "80")
        self.bodyfat_var = tk.StringVar(
            value=str(profile.bodyfat_percent) if profile and profile.bodyfat_percent is not None else ""
        )
        self.has_disability_var = tk.StringVar(
            value=_bool_to_yes_no(profile.has_disability) if profile and profile.has_disability is not None else "No"
        )
        self.drug_tested_var = tk.StringVar(
            value=_bool_to_yes_no(profile.drug_tested_preference) if profile else "Yes"
        )
        self.competition_date_var = tk.StringVar(value=(profile.competition_date or "") if profile else "")

        self.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(master, text="Profile Name").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        name_entry = ttk.Entry(master, textvariable=self.profile_name_var, width=28)
        name_entry.grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(master, text="Sex Category").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        sex_combo = ttk.Combobox(master, state="readonly", values=SEX_CATEGORIES, textvariable=self.sex_category_var)
        sex_combo.grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(master, text="Date of Birth (YYYY-MM-DD)").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        dob_entry = ttk.Entry(master, textvariable=self.dob_var)
        dob_entry.grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(master, text="Height (cm)").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(master, textvariable=self.height_var).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(master, text="Weight (kg)").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(master, textvariable=self.weight_var).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(master, text="Body Fat (%) (Opcional)").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(master, textvariable=self.bodyfat_var).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(
            master,
            text="Do you have a recognised disability? (Opcional)",
            wraplength=340,
        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=(8, 2))
        row += 1

        disability_row = ttk.Frame(master)
        disability_row.grid(row=row, column=1, sticky="w", pady=2)
        ttk.Radiobutton(disability_row, text="Yes", value="Yes", variable=self.has_disability_var).pack(side="left")
        ttk.Radiobutton(disability_row, text="No", value="No", variable=self.has_disability_var).pack(
            side="left", padx=(12, 0)
        )
        row += 1

        ttk.Label(
            master,
            text="Will you compete in drug-tested (natural) federations?",
            wraplength=340,
        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=(8, 2))
        row += 1

        drug_row = ttk.Frame(master)
        drug_row.grid(row=row, column=1, sticky="w", pady=2)
        ttk.Radiobutton(drug_row, text="Yes", value="Yes", variable=self.drug_tested_var).pack(side="left")
        ttk.Radiobutton(drug_row, text="No", value="No", variable=self.drug_tested_var).pack(side="left", padx=(12, 0))
        row += 1

        ttk.Label(master, text="Competition Date (YYYY-MM-DD) (Opcional)").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(master, textvariable=self.competition_date_var).grid(row=row, column=1, sticky="ew", pady=4)

        return name_entry

    def validate(self) -> bool:
        try:
            profile_name = self.profile_name_var.get().strip()
            if not profile_name:
                raise ValueError("Profile Name is required.")

            sex_category = self.sex_category_var.get().strip()
            if sex_category not in SEX_CATEGORIES:
                raise ValueError("Sex Category is required.")

            date_of_birth = _parse_required_iso_date(self.dob_var.get(), "Date of Birth")
            computed_age = calculate_age_years(date_of_birth)
            if computed_age is None:
                raise ValueError("Date of Birth cannot be in the future.")

            height_cm = _parse_positive_float(self.height_var.get(), "Height")
            weight_kg = _parse_positive_float(self.weight_var.get(), "Weight")
            bodyfat_percent = _parse_optional_float(self.bodyfat_var.get(), "Body Fat")
            if bodyfat_percent is not None and (bodyfat_percent < 0 or bodyfat_percent > 100):
                raise ValueError("Body Fat must be between 0 and 100.")
            has_disability = _yes_no_to_bool(self.has_disability_var.get())
            competition_date = _parse_optional_iso_date(self.competition_date_var.get(), "Competition Date")
            drug_tested_preference = _yes_no_to_bool(self.drug_tested_var.get())

            profile_id = self.profile.profile_id if self.profile else uuid.uuid4().hex

            self.result = AthleteProfile(
                profile_id=profile_id,
                profile_name=profile_name,
                sex_category=sex_category,
                date_of_birth=date_of_birth,
                age_years=computed_age,
                height_cm=height_cm,
                weight_kg=weight_kg,
                bodyfat_percent=bodyfat_percent,
                has_disability=has_disability,
                drug_tested_preference=drug_tested_preference,
                competition_date=competition_date,
            )
            self.result.normalize()
            return True
        except ValueError as exc:
            messagebox.showerror("Invalid Profile Data", str(exc), parent=self)
            return False

    def apply(self) -> None:
        return


class ProfilePickerDialog(simpledialog.Dialog):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        profiles: list[AthleteProfile],
        active_profile_id: Optional[str] = None,
    ) -> None:
        self.profiles = profiles
        self.active_profile_id = active_profile_id
        self.result: Optional[str] = None
        super().__init__(parent, title)

    def body(self, master: tk.Misc) -> tk.Widget:
        ttk.Label(master, text="Select profile:").pack(anchor="w", pady=(0, 6))
        frame = ttk.Frame(master)
        frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(frame, height=10, width=46)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        selected_index = 0
        for idx, profile in enumerate(self.profiles):
            marker = " (active)" if profile.profile_id == self.active_profile_id else ""
            self.listbox.insert("end", f"{profile.profile_name}{marker}")
            if profile.profile_id == self.active_profile_id:
                selected_index = idx
        self.listbox.selection_set(selected_index)
        self.listbox.activate(selected_index)
        self.listbox.bind("<Double-Button-1>", lambda _evt: self.ok())
        return self.listbox

    def validate(self) -> bool:
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showerror("Select Profile", "Please select a profile.", parent=self)
            return False
        idx = int(selection[0])
        self.result = self.profiles[idx].profile_id
        return True

    def apply(self) -> None:
        return


@dataclass(frozen=True)
class DivisionChoiceRow:
    federation_name: str
    division_name: str
    class_name: Optional[str]
    ref: SelectedDivisionRef
    eligible: bool
    reasons: tuple[str, ...]

    @property
    def label(self) -> str:
        if self.class_name:
            return f"{self.federation_name} / {self.division_name} / {self.class_name}"
        return f"{self.federation_name} / {self.division_name}"


@dataclass
class PoseCaptureEntry:
    score: float
    created_at: float
    full_path: Path
    cutout_path: Optional[Path] = None


class CategorySelectionDialog(simpledialog.Dialog):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        rows: list[DivisionChoiceRow],
        selected_keys: set[str],
    ) -> None:
        self.rows = rows
        self.selected_keys = selected_keys
        self.result: Optional[list[SelectedDivisionRef]] = None
        self._index_to_row: list[DivisionChoiceRow] = []
        self._tooltip: Optional[tk.Toplevel] = None
        self._tooltip_label: Optional[ttk.Label] = None
        self._tooltip_index: Optional[int] = None
        super().__init__(parent, title)

    def body(self, master: tk.Misc) -> tk.Widget:
        master.columnconfigure(0, weight=1)
        master.rowconfigure(2, weight=1)

        ttk.Label(
            master,
            text=(
                "Categories for the active athlete profile. Eligible rows are selectable. "
                "Ineligible rows appear in gray; hover to see reasons."
            ),
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        controls = ttk.Frame(master)
        controls.grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Button(controls, text="Select All Eligible", command=self._select_all).pack(side="left")
        ttk.Button(controls, text="Clear Selection", command=self._clear_selection).pack(side="left", padx=(8, 0))

        frame = ttk.Frame(master)
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(frame, selectmode="extended", width=96, height=20)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        selected_indexes: list[int] = []
        for idx, row in enumerate(self.rows):
            suffix = ""
            if row.eligible and any("manual confirmation" in r.lower() for r in row.reasons):
                suffix = " [Needs confirmation]"
            self.listbox.insert("end", row.label + suffix)
            self._index_to_row.append(row)
            if row.eligible and row.ref.key() in self.selected_keys:
                selected_indexes.append(idx)
            if not row.eligible:
                self.listbox.itemconfig(idx, fg="#6b7280")
        for idx in selected_indexes:
            self.listbox.selection_set(idx)
        if self.rows:
            self.listbox.activate(0)
        self.listbox.bind("<<ListboxSelect>>", self._on_selection_change)
        self.listbox.bind("<Motion>", self._on_listbox_motion)
        self.listbox.bind("<Leave>", self._on_listbox_leave)
        return self.listbox

    def _select_all(self) -> None:
        self.listbox.selection_clear(0, "end")
        for idx, row in enumerate(self.rows):
            if row.eligible:
                self.listbox.selection_set(idx)

    def _clear_selection(self) -> None:
        self.listbox.selection_clear(0, "end")

    def _on_selection_change(self, _event: Optional[tk.Event] = None) -> None:
        selected = list(self.listbox.curselection())
        for idx in selected:
            try:
                row = self._index_to_row[int(idx)]
            except Exception:
                continue
            if not row.eligible:
                self.listbox.selection_clear(int(idx))

    def _on_listbox_motion(self, event: tk.Event) -> None:
        idx = int(self.listbox.nearest(event.y))
        if idx < 0 or idx >= len(self._index_to_row):
            self._hide_tooltip()
            return
        bbox = self.listbox.bbox(idx)
        if not bbox:
            self._hide_tooltip()
            return
        _x, y, _w, h = bbox
        if event.y < y or event.y > (y + h):
            self._hide_tooltip()
            return
        row = self._index_to_row[idx]
        if row.eligible:
            self._hide_tooltip()
            return
        reasons = list(row.reasons) or ["Not eligible for current athlete profile."]
        text = "\n".join(reasons)
        self._show_tooltip(event.x_root + 14, event.y_root + 16, text=text, row_index=idx)

    def _on_listbox_leave(self, _event: Optional[tk.Event] = None) -> None:
        self._hide_tooltip()

    def _show_tooltip(self, x: int, y: int, text: str, row_index: int) -> None:
        if self._tooltip is None:
            tip = tk.Toplevel(self.listbox)
            tip.withdraw()
            tip.overrideredirect(True)
            tip.attributes("-topmost", True)
            label = tk.Label(
                tip,
                text=text,
                justify="left",
                relief="solid",
                borderwidth=1,
                padx=8,
                pady=6,
                bg="#ffffe0",
                fg="#111827",
                wraplength=520,
            )
            label.pack(fill="both", expand=True)
            self._tooltip = tip
            self._tooltip_label = label
        if self._tooltip_label is None or self._tooltip is None:
            return
        if self._tooltip_index == row_index and self._tooltip_label.cget("text") == text:
            self._tooltip.geometry(f"+{int(x)}+{int(y)}")
            return
        self._tooltip_index = row_index
        self._tooltip_label.configure(text=text)
        self._tooltip.geometry(f"+{int(x)}+{int(y)}")
        self._tooltip.deiconify()

    def _hide_tooltip(self) -> None:
        self._tooltip_index = None
        if self._tooltip is not None:
            self._tooltip.withdraw()

    def validate(self) -> bool:
        self._hide_tooltip()
        refs: list[SelectedDivisionRef] = []
        for idx in self.listbox.curselection():
            try:
                row = self._index_to_row[int(idx)]
            except Exception:
                continue
            if row.eligible:
                refs.append(row.ref)
        self.result = refs
        return True

    def apply(self) -> None:
        return

    def destroy(self) -> None:
        self._hide_tooltip()
        if self._tooltip is not None:
            try:
                self._tooltip.destroy()
            except Exception:
                pass
            self._tooltip = None
            self._tooltip_label = None
        super().destroy()


class DesktopApp:
    _AGE_WINDOW_YEARS_AHEAD = 3
    _POSE_ALL_SELECTED_LABEL = "All selected"
    _TEMP_DISABLED_FEDERATION_IDS = {"nabba"}
    _SOURCE_LABEL_TO_ENUM = {
        "None (Offline)": InputSource.NONE,
        "Webcam": InputSource.WEBCAM,
        "Kinect": InputSource.KINECT,
    }
    _SOURCE_ENUM_TO_LABEL = {enum_val: label for label, enum_val in _SOURCE_LABEL_TO_ENUM.items()}

    def __init__(
        self,
        source_kind: str = "none",
        camera: int | str | None = None,
        width: int = 1280,
        height: int = 720,
        state_store: Optional[AppStateStore] = None,
    ) -> None:
        self.source_kind = (source_kind or "none").strip().lower()
        self.camera = camera
        self.width = int(width)
        self.height = int(height)

        # Desktop must always start safely with device OFF.
        initial_source = InputSource.NONE.value

        self.device_manager = DeviceManager(
            width=self.width,
            height=self.height,
            camera=self.camera,
            initial_source=initial_source,
        )
        self.device_manager.select_source(InputSource.NONE)

        self.profile_store = AthleteProfileStore.default()
        self.app_state = state_store or get_global_app_state()
        self._spec_library = load_federation_specs()
        self._apply_temp_disabled_federations()
        self._prune_disabled_selected_divisions()

        self.root = tk.Tk()
        self.root.title("Bodybuilding Coach")
        self.root.geometry("1500x900")
        self.root.minsize(1150, 720)
        self.root.bind("<F11>", self._on_f11)
        self.root.bind("<Escape>", self._on_escape)

        self._ui_tick_after_id: Optional[str] = None
        self._video_photo: Optional[tk.PhotoImage] = None
        self._frame_count = 0
        self._state_subscription: Optional[int] = None
        self._last_device_key: Optional[tuple[str, str, str]] = None
        self._sample_frame_rgb: Optional[np.ndarray] = None
        self._sample_input_path: Optional[Path] = None
        self._sample_kind: str = ""
        self._error_details_visible = False
        self._video_only_active = False
        self._video_fullscreen_var = tk.BooleanVar(value=False)
        self._video_only_photo: Optional[tk.PhotoImage] = None
        self._device_power_menu_index: Optional[int] = None
        self._device_retry_menu_index: Optional[int] = None
        self._device_menu: Optional[tk.Menu] = None
        self._webcam_devices_menu: Optional[tk.Menu] = None
        self._menubar: Optional[tk.Menu] = None
        self._hidden_menu: Optional[tk.Menu] = None
        self._last_federation_summary_key: Optional[tuple[object, ...]] = None
        self._latest_age_snapshot: Optional[dict[str, object]] = None
        self._session_store = SessionStore.default()
        self._metrics_tab: Optional[MetricsTab] = None
        self._metrics_status_last: str = ""
        self._pose_backend: Optional[PoseBackend] = None
        self._pose_backend_error: str = ""
        self._pose_backend_init_attempted = False
        self._pose_frame_counter = 0
        self._pose_last_result: Optional[PoseResult] = None
        self._pose_last_props: Optional[object] = None
        self._pose_last_score: Optional[float] = None
        self._pose_last_advice: list[str] = []
        self._pose_checklist: list[PoseChecklistItem] = []
        self._pose_index = 0
        self._pose_session_active = False
        self._pose_timer_seconds = 20
        self._pose_timer_started_at = time.time()
        self._pose_last_capture_ts = 0.0
        self._pose_best_captures: dict[str, list[PoseCaptureEntry]] = {}
        self._pose_active_ref_key: Optional[str] = None
        self._pose_selection_options: dict[str, Optional[SelectedDivisionRef]] = {
            self._POSE_ALL_SELECTED_LABEL: None
        }
        self._latest_frame_bgr: Optional[np.ndarray] = None
        self._voice_listener: Optional[VoiceCommandListener] = None
        self._voice_error: str = ""
        self._tts_speaker: Optional[TTSSpeaker] = None
        self._coach_voice_last_line: str = ""
        self._coach_voice_last_ts: float = 0.0
        self._working_divisions_listbox: Optional[tk.Listbox] = None
        self._eligible_divisions_listbox: Optional[tk.Listbox] = None
        self._pose_active_division_combo: Optional[ttk.Combobox] = None
        self._right_notebook: Optional[ttk.Notebook] = None
        self._active_workflow_tab: str = "pose"
        self._last_kinect_probe_ts: float = 0.0
        self._last_kinect_probe_available: bool = False

        self._show_left_var = tk.BooleanVar(value=True)
        self._show_right_var = tk.BooleanVar(value=True)
        self._device_source_var = tk.StringVar(value="None (Offline)")
        self._webcam_device_var = tk.StringVar(value="")
        self._device_message_var = tk.StringVar(value="")
        self._preview_caption_var = tk.StringVar(value="")
        self._offline_hint_var = tk.StringVar(value="Offline: no live input selected")
        self._offline_sample_var = tk.StringVar(value="")
        self._error_message_var = tk.StringVar(value="")
        self._error_steps_var = tk.StringVar(value="")

        self._status_var = tk.StringVar(value="No active profile")
        self._summary_name_var = tk.StringVar(value="—")
        self._summary_id_var = tk.StringVar(value="—")
        self._summary_sex_var = tk.StringVar(value="—")
        self._summary_dob_var = tk.StringVar(value="—")
        self._summary_height_var = tk.StringVar(value="—")
        self._summary_weight_var = tk.StringVar(value="—")
        self._summary_bodyfat_var = tk.StringVar(value="—")
        self._summary_disability_var = tk.StringVar(value="—")
        self._summary_drug_tested_var = tk.StringVar(value="—")
        self._age_today_var = tk.StringVar(value="—")
        self._age_on_competition_var = tk.StringVar(value="—")
        self._competition_turns_vars: dict[int, tk.StringVar] = {
            age: tk.StringVar(value="—") for age in TURN_AGES
        }
        self._countdown_days_var = tk.StringVar(value="—")
        self._countdown_weeks_var = tk.StringVar(value="—")
        self._countdown_months_var = tk.StringVar(value="—")
        self._working_divisions_var = tk.StringVar(value="No categories selected yet.")
        self._eligible_divisions_var = tk.StringVar(value="Load profile to compute eligible categories.")
        self._pose_session_var = tk.StringVar(value="OFF")
        self._pose_current_var = tk.StringVar(value="No pose selected")
        self._pose_score_var = tk.StringVar(value="Score: —")
        self._pose_timer_var = tk.StringVar(value="Timer: —")
        self._pose_advice_var = tk.StringVar(value="")
        self._pose_active_division_var = tk.StringVar(value=self._POSE_ALL_SELECTED_LABEL)
        self._scan_status_var = tk.StringVar(value="No active 3D scan.")
        self._report_status_var = tk.StringVar(value="")
        self._pose_timer_enabled_var = tk.BooleanVar(value=True)
        self._pose_auto_capture_var = tk.BooleanVar(value=True)
        self._power_button_label_var = tk.StringVar(value="Power On")
        self._pose_show_example_var = tk.BooleanVar(value=True)
        self._voice_enabled_var = tk.BooleanVar(value=False)
        self._coach_voice_enabled_var = tk.BooleanVar(value=False)
        self._voice_status_var = tk.StringVar(value="Mic: OFF")
        self._mic_button_label_var = tk.StringVar(value="Mic: OFF")
        self._coach_button_label_var = tk.StringVar(value="Coach Voice: OFF")
        self._kinect_available_var = tk.StringVar(value="Kinect: check from Device > Detected Devices")

        self._build_ui()
        self._build_menu()
        self._sync_voice_button_labels()
        self._state_subscription = self.app_state.subscribe(self._on_state_change)
        self._load_initial_profile()
        self._apply_device_snapshot(self.device_manager.snapshot())
        self._schedule_ui_tick()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_temp_disabled_federations(self) -> None:
        for fed_id in self._TEMP_DISABLED_FEDERATION_IDS:
            self._spec_library.federations.pop(str(fed_id).lower(), None)

    def _prune_disabled_selected_divisions(self) -> None:
        state_refs = list(self.app_state.state.selected_divisions)
        if not state_refs:
            return
        allowed_feds = set(self._spec_library.federations.keys())
        filtered = [ref for ref in state_refs if str(ref.federation_id or "").lower() in allowed_feds]
        if len(filtered) != len(state_refs):
            self.app_state.set_selected_divisions(filtered)

    def _build_ui(self) -> None:
        root = self.root
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        self._build_device_panel(root)

        self.main_frame = ttk.Frame(root, padding=8)
        self.main_frame.grid(row=1, column=0, sticky="nsew")
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        self.left_panel = ttk.Frame(self.main_frame, padding=8)
        self.left_panel.grid(row=0, column=0, sticky="nsw")
        self.center_panel = ttk.Frame(self.main_frame, padding=8)
        self.center_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel = ttk.Frame(self.main_frame, padding=8)
        self.right_panel.grid(row=0, column=2, sticky="nse")
        self.center_panel.rowconfigure(1, weight=1)
        self.center_panel.columnconfigure(0, weight=1)

        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

        self.status_bar = ttk.Label(root, textvariable=self._status_var, anchor="w", relief="sunken", padding=(8, 4))
        self.status_bar.grid(row=2, column=0, sticky="ew")

        self.video_only_frame = ttk.Frame(root)
        self.video_only_frame.rowconfigure(0, weight=1)
        self.video_only_frame.columnconfigure(0, weight=1)
        self.video_only_label = tk.Label(
            self.video_only_frame,
            text="No preview",
            bg="#000000",
            fg="#e9edf3",
            anchor="center",
            justify="center",
        )
        self.video_only_label.grid(row=0, column=0, sticky="nsew")

    def _build_device_panel(self, parent: tk.Misc) -> None:
        panel = ttk.LabelFrame(parent, text="Device", padding=8)
        panel.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        panel.columnconfigure(3, weight=1)
        self.device_panel = panel

        ttk.Label(panel, text="Input Source").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, textvariable=self._device_source_var).grid(row=0, column=1, sticky="w", padx=(8, 10))
        ttk.Label(panel, text="Power").grid(row=0, column=2, sticky="w")
        ttk.Label(panel, textvariable=self._device_message_var, foreground="#475569").grid(
            row=0, column=3, sticky="w", padx=(8, 10)
        )

        self.badge_label = tk.Label(
            panel,
            text=DeviceState.DISCONNECTED.value,
            bg="#4b5563",
            fg="white",
            padx=10,
            pady=3,
            relief="ridge",
        )
        self.badge_label.grid(row=0, column=4, sticky="e", padx=(0, 8))

        ttk.Button(panel, textvariable=self._power_button_label_var, command=self.on_device_power_toggle).grid(
            row=0, column=5, sticky="e", padx=(0, 6)
        )
        ttk.Button(panel, textvariable=self._mic_button_label_var, command=self.on_toggle_voice_commands).grid(
            row=0, column=6, sticky="e", padx=(0, 6)
        )
        ttk.Button(panel, textvariable=self._coach_button_label_var, command=self.on_toggle_coach_voice).grid(
            row=0, column=7, sticky="e", padx=(0, 2)
        )

        ttk.Label(panel, textvariable=self._kinect_available_var, foreground="#475569").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )
        ttk.Label(panel, textvariable=self._voice_status_var, foreground="#475569").grid(
            row=1, column=4, columnspan=4, sticky="e", pady=(8, 0)
        )

    def _build_left_panel(self) -> None:
        summary = ttk.LabelFrame(self.left_panel, text="Profile Summary", padding=8)
        summary.pack(fill="x")
        ttk.Label(summary, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self._summary_name_var).grid(row=0, column=1, sticky="w")
        ttk.Label(summary, text="Sex Category").grid(row=1, column=0, sticky="w")
        ttk.Label(summary, textvariable=self._summary_sex_var).grid(row=1, column=1, sticky="w")
        ttk.Label(summary, text="Height (cm)").grid(row=2, column=0, sticky="w")
        ttk.Label(summary, textvariable=self._summary_height_var).grid(row=2, column=1, sticky="w")
        ttk.Label(summary, text="Weight (kg)").grid(row=3, column=0, sticky="w")
        ttk.Label(summary, textvariable=self._summary_weight_var).grid(row=3, column=1, sticky="w")
        ttk.Label(summary, text="Body Fat (%)").grid(row=4, column=0, sticky="w")
        ttk.Label(summary, textvariable=self._summary_bodyfat_var).grid(row=4, column=1, sticky="w")
        ttk.Label(summary, text="Disability").grid(row=5, column=0, sticky="w")
        ttk.Label(summary, textvariable=self._summary_disability_var).grid(row=5, column=1, sticky="w")
        ttk.Label(summary, text="Drug-tested preference").grid(row=6, column=0, sticky="w")
        ttk.Label(summary, textvariable=self._summary_drug_tested_var).grid(row=6, column=1, sticky="w")

        planning = ttk.LabelFrame(self.left_panel, text="Planning (Competition Countdown)", padding=8)
        planning.pack(fill="x", pady=(12, 0))
        ttk.Label(planning, text="Days Remaining").grid(row=0, column=0, sticky="w")
        ttk.Label(planning, textvariable=self._countdown_days_var).grid(row=0, column=1, sticky="w")
        ttk.Label(planning, text="Weeks Remaining").grid(row=1, column=0, sticky="w")
        ttk.Label(planning, textvariable=self._countdown_weeks_var).grid(row=1, column=1, sticky="w")
        ttk.Label(planning, text="Months Remaining").grid(row=2, column=0, sticky="w")
        ttk.Label(planning, textvariable=self._countdown_months_var).grid(row=2, column=1, sticky="w")

    def _build_center_panel(self) -> None:
        ttk.Label(self.center_panel, text="Live Camera / Kinect View", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.view_stack = ttk.Frame(self.center_panel)
        self.view_stack.grid(row=1, column=0, sticky="nsew")
        self.view_stack.rowconfigure(0, weight=1)
        self.view_stack.columnconfigure(0, weight=1)

        self.preview_frame = ttk.Frame(self.view_stack)
        self.preview_frame.grid(row=0, column=0, sticky="nsew")
        self.preview_frame.rowconfigure(1, weight=1)
        self.preview_frame.columnconfigure(0, weight=1)
        ttk.Label(self.preview_frame, textvariable=self._preview_caption_var).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.video_label = tk.Label(
            self.preview_frame,
            text="No preview",
            bg="#101215",
            fg="#e9edf3",
            anchor="center",
            justify="center",
        )
        self.video_label.grid(row=1, column=0, sticky="nsew")

        self.offline_frame = ttk.Frame(self.view_stack, padding=18)
        self.offline_frame.grid(row=0, column=0, sticky="nsew")
        self.offline_frame.columnconfigure(0, weight=1)
        ttk.Label(
            self.offline_frame,
            textvariable=self._offline_hint_var,
            font=("TkDefaultFont", 12, "bold"),
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.offline_frame,
            textvariable=self._offline_sample_var,
            wraplength=760,
            foreground="#475569",
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.error_frame = ttk.Frame(self.view_stack, padding=18)
        self.error_frame.grid(row=0, column=0, sticky="nsew")
        self.error_frame.columnconfigure(0, weight=1)
        ttk.Label(self.error_frame, text="Connection Error", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            self.error_frame,
            textvariable=self._error_message_var,
            wraplength=760,
            justify="left",
            foreground="#991b1b",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            self.error_frame,
            textvariable=self._error_steps_var,
            wraplength=760,
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.error_toggle_button = ttk.Button(
            self.error_frame,
            text="Show technical details",
            command=self.on_toggle_error_details,
        )
        self.error_toggle_button.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.error_details = ScrolledText(self.error_frame, height=12, width=100, wrap="word")
        self.error_details.configure(state="disabled")

    def _build_right_panel(self) -> None:
        notebook = ttk.Notebook(self.right_panel)
        notebook.pack(fill="both", expand=True)
        self._right_notebook = notebook
        notebook.bind("<<NotebookTabChanged>>", self.on_right_tab_changed)

        pose_tab = ttk.Frame(notebook, padding=10)
        scan_tab = ttk.Frame(notebook, padding=10)
        reports_tab = ttk.Frame(notebook, padding=10)

        notebook.add(pose_tab, text="Pose Coach")
        notebook.add(scan_tab, text="3D Scan / Point Cloud")
        notebook.add(reports_tab, text="Reports / Exports")

        ttk.Label(pose_tab, text="Active Category For Posing", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        active_row = ttk.Frame(pose_tab)
        active_row.pack(fill="x", pady=(4, 8))
        self._pose_active_division_combo = ttk.Combobox(
            active_row,
            state="readonly",
            textvariable=self._pose_active_division_var,
            values=[self._POSE_ALL_SELECTED_LABEL],
        )
        self._pose_active_division_combo.pack(side="left", fill="x", expand=True)
        self._pose_active_division_combo.bind("<<ComboboxSelected>>", self.on_pose_active_division_selected)

        ttk.Separator(pose_tab, orient="horizontal").pack(fill="x", pady=(8, 8))
        ttk.Label(pose_tab, text="Pose Session", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(pose_tab, textvariable=self._pose_session_var).pack(anchor="w", pady=(4, 0))
        ttk.Label(pose_tab, textvariable=self._pose_current_var, wraplength=300, justify="left").pack(anchor="w")
        ttk.Label(pose_tab, textvariable=self._pose_score_var).pack(anchor="w")
        ttk.Label(pose_tab, textvariable=self._pose_timer_var).pack(anchor="w")
        ttk.Label(
            pose_tab,
            textvariable=self._pose_advice_var,
            wraplength=300,
            justify="left",
            foreground="#475569",
        ).pack(anchor="w", pady=(2, 8))

        pose_btn_row = ttk.Frame(pose_tab)
        pose_btn_row.pack(anchor="w", fill="x")
        ttk.Button(pose_btn_row, text="Start/Stop Session", command=self.on_pose_toggle_session).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(pose_btn_row, text="Prev Pose", command=self.on_pose_prev).pack(side="left", padx=(0, 6))
        ttk.Button(pose_btn_row, text="Next Pose", command=self.on_pose_next).pack(side="left", padx=(0, 6))
        ttk.Button(pose_btn_row, text="Capture Now", command=self.on_pose_capture_now).pack(side="left")

        pose_btn_row2 = ttk.Frame(pose_tab)
        pose_btn_row2.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Button(pose_btn_row2, text="Reset Timer", command=self.on_pose_reset_timer).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(pose_btn_row2, text="Pose Timer", variable=self._pose_timer_enabled_var).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(pose_btn_row2, text="Auto Capture Top-3", variable=self._pose_auto_capture_var).pack(
            side="left", padx=(0, 8)
        )
        ttk.Checkbutton(pose_btn_row2, text="Show Pose Example", variable=self._pose_show_example_var).pack(side="left")

        ttk.Label(scan_tab, text="3D Scan / Point Cloud", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(
            scan_tab,
            text="Controls run against the currently active live input/frame context.",
            wraplength=300,
            justify="left",
            foreground="#475569",
        ).pack(anchor="w", pady=(4, 8))
        scan_row1 = ttk.Frame(scan_tab)
        scan_row1.pack(anchor="w", fill="x")
        ttk.Button(scan_row1, text="Start 4-Side Scan", command=self.on_scan4_start).pack(side="left", padx=(0, 6))
        ttk.Button(scan_row1, text="Load Scan Dir", command=self.on_scan4_load).pack(side="left", padx=(0, 6))
        ttk.Button(scan_row1, text="Compute Metrics", command=self.on_scan4_compute_metrics).pack(side="left")
        scan_row2 = ttk.Frame(scan_tab)
        scan_row2.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Button(scan_row2, text="Auto Merge", command=self.on_scan4_auto_merge).pack(side="left", padx=(0, 6))
        ttk.Button(scan_row2, text="Manual Align", command=self.on_scan4_manual_align).pack(side="left", padx=(0, 6))
        ttk.Button(scan_row2, text="Export PDF", command=self.on_scan4_export).pack(side="left")
        scan_row3 = ttk.Frame(scan_tab)
        scan_row3.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Button(scan_row3, text="Open Scan Folder", command=self.on_scan4_open_folder).pack(side="left", padx=(0, 6))
        ttk.Button(scan_row3, text="Open merged.pcd", command=self.on_scan4_open_merged_pcd).pack(side="left")
        ttk.Label(
            scan_tab,
            textvariable=self._scan_status_var,
            wraplength=300,
            justify="left",
            foreground="#475569",
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(reports_tab, text="Reports / Exports", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(
            reports_tab,
            text="Generate or open report artifacts from scan/export workflows.",
            wraplength=300,
            justify="left",
            foreground="#475569",
        ).pack(anchor="w", pady=(4, 8))
        rep_row = ttk.Frame(reports_tab)
        rep_row.pack(anchor="w", fill="x")
        ttk.Button(rep_row, text="Open Sessions Folder", command=self.on_open_sessions_folder).pack(side="left", padx=(0, 6))
        ttk.Button(rep_row, text="Generate PDF (3D Scan)", command=self.on_generate_pdf_report).pack(side="left")
        ttk.Label(
            reports_tab,
            textvariable=self._report_status_var,
            wraplength=300,
            justify="left",
            foreground="#475569",
        ).pack(anchor="w", pady=(8, 0))

    def on_right_tab_changed(self, _event: Optional[tk.Event] = None) -> None:
        if self._right_notebook is None:
            self._active_workflow_tab = "pose"
            return
        try:
            idx = int(self._right_notebook.index(self._right_notebook.select()))
        except Exception:
            idx = 0
        self._active_workflow_tab = "scan" if idx == 1 else "pose"
        if self._active_workflow_tab == "scan":
            self._pose_advice_var.set("")
            self._coach_voice_last_line = ""
            if self._tts_speaker is not None:
                self._tts_speaker.clear_pending()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Profile", command=self.on_new_profile)
        file_menu.add_command(label="Edit Profile", command=self.on_edit_profile)
        file_menu.add_command(label="Switch Profile", command=self.on_switch_profile)
        file_menu.add_command(label="Delete Profile", command=self.on_delete_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Create Demo Profile", command=self.on_create_demo_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        device_menu = tk.Menu(menubar, tearoff=0)
        self._device_menu = device_menu
        source_menu = tk.Menu(device_menu, tearoff=0)
        for label in self._SOURCE_LABEL_TO_ENUM.keys():
            source_menu.add_radiobutton(
                label=label,
                value=label,
                variable=self._device_source_var,
                command=self.on_device_source_selected,
            )
        device_menu.add_cascade(label="Input Source", menu=source_menu)
        device_menu.add_separator()
        device_menu.add_command(label="Power On", command=self.on_device_power_toggle)
        self._device_power_menu_index = int(device_menu.index("end"))
        device_menu.add_command(label="Retry Connection", command=self.on_device_retry)
        self._device_retry_menu_index = int(device_menu.index("end"))
        device_menu.add_separator()
        webcam_devices_menu = tk.Menu(device_menu, tearoff=0, postcommand=self._refresh_webcam_devices_menu)
        self._webcam_devices_menu = webcam_devices_menu
        device_menu.add_cascade(label="Detected Devices", menu=webcam_devices_menu)
        device_menu.add_command(label="Select Webcam Device...", command=self.on_select_webcam_device)
        device_menu.add_command(label="Set Resolution...", command=self.on_set_resolution)
        device_menu.add_separator()
        device_menu.add_command(label="Use Sample Input", command=self.on_use_sample_input)
        menubar.add_cascade(label="Device", menu=device_menu)

        fed_menu = tk.Menu(menubar, tearoff=0)
        fed_menu.add_command(label="Browse Rules", command=self.on_browse_rules)
        fed_menu.add_command(label="Select Categories", command=self.on_select_categories)
        fed_menu.add_command(label="Select All Eligible", command=self.on_select_all_eligible_categories)
        menubar.add_cascade(label="Federations/Categories", menu=fed_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(
            label="Show Profile Summary",
            variable=self._show_left_var,
            command=self.on_toggle_left_panel,
        )
        view_menu.add_checkbutton(
            label="Show Side Tabs",
            variable=self._show_right_var,
            command=self.on_toggle_right_panel,
        )
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label="Video Fullscreen (F11)",
            variable=self._video_fullscreen_var,
            command=self.on_toggle_video_fullscreen,
        )
        view_menu.add_separator()
        view_menu.add_command(label="Reset Layout", command=self.on_reset_layout)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Quick Start", command=self.on_help_quick_start)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.on_help_shortcuts)
        help_menu.add_command(label="Data & Disclaimer", command=self.on_help_disclaimer)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self._menubar = menubar
        self.root.config(menu=menubar)

    def _load_initial_profile(self) -> None:
        profile = self.profile_store.load_active_profile()
        if profile is None:
            self._status_var.set("No active profile. Use File > New Profile or Create Demo Profile.")
            self.app_state.set_active_profile(None)
            return
        self.app_state.set_active_profile(profile)

    def _on_state_change(self, state: ApplicationState) -> None:
        profile = state.active_profile
        device_state = str(state.session_metadata.get("device_state", DeviceState.DISCONNECTED.value))
        source_name = str(state.session_metadata.get("input_source", "Offline"))
        if profile is None:
            self._summary_name_var.set("—")
            self._summary_id_var.set("—")
            self._summary_sex_var.set("—")
            self._summary_dob_var.set("—")
            self._summary_height_var.set("—")
            self._summary_weight_var.set("—")
            self._summary_bodyfat_var.set("—")
            self._summary_disability_var.set("—")
            self._summary_drug_tested_var.set("—")
            self._age_today_var.set("—")
            self._age_on_competition_var.set("—")
            for age in TURN_AGES:
                self._competition_turns_vars[age].set("—")
            self._countdown_days_var.set("—")
            self._countdown_weeks_var.set("—")
            self._countdown_months_var.set("—")
            self._latest_age_snapshot = None
            self._metrics_tab = None
            self._pose_checklist = []
            self._pose_index = 0
            self._pose_session_active = False
            self._pose_session_var.set("OFF")
            self._pose_current_var.set("No pose selected")
            self._pose_score_var.set("Score: —")
            self._pose_timer_var.set("Timer: —")
            self._pose_advice_var.set("")
            self._pose_active_ref_key = None
            self._refresh_pose_selection_options([])
            self._refresh_federation_category_summary(profile=None, selected_refs=state.selected_divisions)
            self._last_federation_summary_key = self._federation_summary_key(
                profile=None,
                selected_refs=state.selected_divisions,
            )
            self._status_var.set(f"Active Profile: none | Device: {device_state} ({source_name})")
            return

        if self._prune_selected_divisions_for_profile(profile, state.selected_divisions):
            return

        if self._metrics_tab is not None and self._metrics_tab.profile_name != profile.profile_name:
            self._metrics_tab = None
            self._metrics_status_last = ""
            self._scan_status_var.set("No active 3D scan.")

        self._summary_name_var.set(profile.profile_name)
        self._summary_id_var.set(profile.profile_id)
        self._summary_sex_var.set(profile.sex_category)
        self._summary_dob_var.set(profile.date_of_birth)
        self._summary_height_var.set(f"{profile.height_cm:.1f}")
        self._summary_weight_var.set(f"{profile.weight_kg:.1f}")
        self._summary_bodyfat_var.set("—" if profile.bodyfat_percent is None else f"{profile.bodyfat_percent:.1f}")
        if profile.has_disability is None:
            self._summary_disability_var.set("—")
        else:
            self._summary_disability_var.set(_bool_to_yes_no(profile.has_disability))
        self._summary_drug_tested_var.set(_bool_to_yes_no(profile.drug_tested_preference))

        age_snapshot = build_age_eligibility_snapshot(profile.date_of_birth, profile.competition_date)
        if age_snapshot is None:
            self._age_today_var.set("—")
            self._age_on_competition_var.set("—")
            for age in TURN_AGES:
                self._competition_turns_vars[age].set("—")
            self._latest_age_snapshot = {
                "valid": False,
                "profile_id": profile.profile_id,
                "date_of_birth": profile.date_of_birth,
                "competition_date": profile.competition_date,
            }
        else:
            self._age_today_var.set(str(age_snapshot.age_today))
            self._age_on_competition_var.set(str(age_snapshot.age_on_competition_date))
            for age in TURN_AGES:
                self._competition_turns_vars[age].set(str(age_snapshot.competition_year_turns[age]))
            self._latest_age_snapshot = {
                "valid": True,
                "profile_id": profile.profile_id,
                "date_of_birth": profile.date_of_birth,
                "competition_date": profile.competition_date,
                "reference_date": age_snapshot.reference_date.isoformat(),
                "uses_competition_date": bool(age_snapshot.uses_competition_date),
                "age_today": int(age_snapshot.age_today),
                "age_on_competition_date": int(age_snapshot.age_on_competition_date),
                "competition_year_turns": {
                    int(age): int(year) for age, year in age_snapshot.competition_year_turns.items()
                },
            }

        countdown = calculate_competition_countdown(profile.competition_date)
        if countdown is None:
            self._countdown_days_var.set("—")
            self._countdown_weeks_var.set("—")
            self._countdown_months_var.set("—")
        else:
            self._countdown_days_var.set(f"{countdown.days_remaining}")
            self._countdown_weeks_var.set(f"{countdown.weeks_remaining:.2f}")
            self._countdown_months_var.set(f"{countdown.months_remaining:.2f}")

        fed_summary_key = self._federation_summary_key(profile=profile, selected_refs=state.selected_divisions)
        if fed_summary_key != self._last_federation_summary_key:
            self._refresh_federation_category_summary(profile=profile, selected_refs=state.selected_divisions)
            self._last_federation_summary_key = fed_summary_key
        self._sync_pose_checklist(state.selected_divisions)

        self._status_var.set(f"Active Profile: {profile.profile_name} | Device: {device_state} ({source_name})")

    def _ensure_metrics_tab(self, profile_name: str) -> None:
        if self._metrics_tab is not None and self._metrics_tab.profile_name == profile_name:
            return
        self._metrics_tab = MetricsTab(profile_name)
        self._metrics_status_last = ""
        self._scan_status_var.set("No active 3D scan.")

    def _metrics_tab_for_active_profile(self, require_profile: bool = True) -> Optional[MetricsTab]:
        profile = self._active_profile()
        if profile is None:
            if require_profile:
                messagebox.showinfo("3D Scan", "Load a profile first.", parent=self.root)
            return None
        self._ensure_metrics_tab(profile.profile_name)
        return self._metrics_tab

    def _label_for_selected_division_ref(self, ref: SelectedDivisionRef) -> str:
        fed = self._spec_library.get_federation(ref.federation_id)
        div = self._spec_library.get_division(ref)
        cls = self._spec_library.get_division_class(ref)
        if fed is None or div is None:
            return ref.key()
        if cls is not None:
            return f"{fed.federation_name} / {div.division_name} / {cls.class_name}"
        return f"{fed.federation_name} / {div.division_name}"

    def _refresh_pose_selection_options(self, selected_refs: list[SelectedDivisionRef]) -> None:
        options: dict[str, Optional[SelectedDivisionRef]] = {self._POSE_ALL_SELECTED_LABEL: None}
        for ref in selected_refs:
            options[self._label_for_selected_division_ref(ref)] = ref
        self._pose_selection_options = options

        valid_keys = {ref.key() for ref in selected_refs}
        if self._pose_active_ref_key and self._pose_active_ref_key not in valid_keys:
            self._pose_active_ref_key = None

        selected_label = self._POSE_ALL_SELECTED_LABEL
        if self._pose_active_ref_key:
            for label, ref in options.items():
                if ref is not None and ref.key() == self._pose_active_ref_key:
                    selected_label = label
                    break
        self._pose_active_division_var.set(selected_label)
        if self._pose_active_division_combo is not None:
            self._pose_active_division_combo.configure(values=list(options.keys()))

    def on_pose_active_division_selected(self, _event: Optional[tk.Event] = None) -> None:
        chosen_label = str(self._pose_active_division_var.get() or "").strip()
        ref = self._pose_selection_options.get(chosen_label)
        self._pose_active_ref_key = ref.key() if ref is not None else None
        self._pose_index = 0
        self._reset_pose_timer()
        self._sync_pose_checklist(list(self.app_state.state.selected_divisions))

    def on_pose_use_all_categories(self) -> None:
        self._pose_active_ref_key = None
        self._pose_active_division_var.set(self._POSE_ALL_SELECTED_LABEL)
        self._pose_index = 0
        self._reset_pose_timer()
        self._sync_pose_checklist(list(self.app_state.state.selected_divisions))

    def _refs_for_active_pose_selection(self, selected_refs: list[SelectedDivisionRef]) -> list[SelectedDivisionRef]:
        if not self._pose_active_ref_key:
            return list(selected_refs)
        filtered = [ref for ref in selected_refs if ref.key() == self._pose_active_ref_key]
        if filtered:
            return filtered
        self._pose_active_ref_key = None
        self._pose_active_division_var.set(self._POSE_ALL_SELECTED_LABEL)
        return list(selected_refs)

    def _sync_pose_checklist(self, selected_refs: list[SelectedDivisionRef]) -> None:
        self._refresh_pose_selection_options(selected_refs)
        active_refs = self._refs_for_active_pose_selection(selected_refs)
        checklist = build_pose_checklist(self._spec_library, active_refs)
        prev_key = tuple(
            (item.federation_id, item.division_id, item.class_id or "", item.pose_name) for item in self._pose_checklist
        )
        next_key = tuple((item.federation_id, item.division_id, item.class_id or "", item.pose_name) for item in checklist)
        self._pose_checklist = checklist
        if self._pose_checklist:
            if self._pose_index >= len(self._pose_checklist):
                self._pose_index = 0
        else:
            self._pose_index = 0
        if prev_key != next_key:
            self._pose_timer_started_at = time.time()
        self._refresh_pose_panel_text()

    def _current_pose_item(self) -> Optional[PoseChecklistItem]:
        if not self._pose_checklist:
            return None
        idx = max(0, min(len(self._pose_checklist) - 1, self._pose_index))
        return self._pose_checklist[idx]

    def _current_pose_key(self) -> str:
        item = self._current_pose_item()
        if item is None:
            return "mp_front"
        return item.pose_key or "mp_front"

    def _current_pose_label(self) -> str:
        item = self._current_pose_item()
        if item is None:
            return "Front Pose (default)"
        return f"{item.pose_name} [{item.division_label}]"

    def _refresh_pose_panel_text(self) -> None:
        self._pose_session_var.set("ON" if self._pose_session_active else "OFF")
        active_label = str(self._pose_active_division_var.get() or self._POSE_ALL_SELECTED_LABEL)
        self._pose_current_var.set(f"Active Category: {active_label}\nCurrent Pose: {self._current_pose_label()}")

    def _reset_pose_timer(self) -> None:
        self._pose_timer_started_at = time.time()

    def on_pose_toggle_session(self) -> None:
        self._pose_session_active = not self._pose_session_active
        self._pose_last_capture_ts = 0.0
        self._reset_pose_timer()
        self._refresh_pose_panel_text()

    def on_pose_prev(self) -> None:
        if self._pose_checklist:
            self._pose_index = (self._pose_index - 1) % len(self._pose_checklist)
        self._reset_pose_timer()
        self._refresh_pose_panel_text()

    def on_pose_next(self) -> None:
        if self._pose_checklist:
            self._pose_index = (self._pose_index + 1) % len(self._pose_checklist)
        self._reset_pose_timer()
        self._refresh_pose_panel_text()

    def on_pose_reset_timer(self) -> None:
        self._reset_pose_timer()

    def on_pose_capture_now(self) -> None:
        if self._pose_last_result is None:
            messagebox.showinfo("Pose Capture", "No live frame available yet.", parent=self.root)
            return
        frame = self.device_manager.snapshot()
        if frame.state != DeviceState.STREAMING:
            messagebox.showinfo("Pose Capture", "Device must be streaming to capture.", parent=self.root)
            return
        if self._latest_frame_bgr is None:
            messagebox.showinfo("Pose Capture", "No frame buffered yet.", parent=self.root)
            return
        self._capture_pose_snapshot(
            frame_bgr=self._latest_frame_bgr,
            pose_result=self._pose_last_result,
            score=self._pose_last_score,
            manual=True,
        )

    def on_toggle_voice_commands(self) -> None:
        enabled = not bool(self._voice_enabled_var.get())
        self._voice_enabled_var.set(enabled)
        if enabled:
            if not self._init_voice_listener():
                self._voice_enabled_var.set(False)
                self._sync_voice_button_labels()
                return
            self._voice_status_var.set("Mic: ON")
            self._sync_voice_button_labels()
            return
        self._voice_status_var.set("Mic: OFF")
        self._sync_voice_button_labels()

    def on_toggle_coach_voice(self) -> None:
        enabled = not bool(self._coach_voice_enabled_var.get())
        self._coach_voice_enabled_var.set(enabled)
        if enabled:
            if self._tts_speaker is None:
                try:
                    self._tts_speaker = TTSSpeaker(backend="piper_bin")
                    self._tts_speaker.start()
                except Exception as exc:
                    self._coach_voice_enabled_var.set(False)
                    self._tts_speaker = None
                    self._voice_status_var.set(f"Coach voice error: {exc}")
                    self._sync_voice_button_labels()
                    return
            self._voice_status_var.set("Coach voice: ON")
            self._sync_voice_button_labels()
            return
        if self._tts_speaker is not None:
            self._tts_speaker.stop()
            self._tts_speaker = None
        if self._voice_enabled_var.get():
            self._voice_status_var.set("Mic: ON")
        else:
            self._voice_status_var.set("Mic: OFF")
        self._sync_voice_button_labels()

    def _sync_voice_button_labels(self) -> None:
        self._mic_button_label_var.set("Mic: ON" if self._voice_enabled_var.get() else "Mic: OFF")
        self._coach_button_label_var.set(
            "Coach Voice: ON" if self._coach_voice_enabled_var.get() else "Coach Voice: OFF"
        )

    def _init_voice_listener(self) -> bool:
        if self._voice_listener is not None:
            return True
        repo_root = Path(__file__).resolve().parent.parent.parent
        model_path = repo_root / "models" / "vosk"
        if not model_path.exists():
            self._voice_error = "Vosk model not found in models/vosk."
            self._voice_status_var.set("Mic: model missing")
            return False
        try:
            mic_prefer: Optional[str] = None
            if self.device_manager.snapshot().selected_source == InputSource.KINECT:
                mic_prefer = "Xbox NUI Sensor"
            self._voice_listener = VoiceCommandListener(
                VoiceCommandConfig(
                    model_path=str(model_path),
                    mic_prefer=mic_prefer,
                    source_kind="kinect2" if mic_prefer else "v4l2",
                )
            )
            self._voice_listener.start()
            self._voice_error = ""
            return True
        except Exception as exc:
            self._voice_error = str(exc)
            self._voice_status_var.set(f"Mic error: {exc}")
            self._voice_listener = None
            return False

    def _poll_voice_commands(self) -> None:
        if not self._voice_enabled_var.get() or self._voice_listener is None:
            return
        err = self._voice_listener.error()
        if err:
            self._voice_status_var.set(f"Mic error: {err}")
            self._voice_error = err
            self._voice_enabled_var.set(False)
            self._sync_voice_button_labels()
            return
        cmd = self._voice_listener.pop_command()
        if not cmd:
            return
        self._apply_voice_command(cmd)

    def _apply_voice_command(self, cmd: str) -> None:
        normalized = str(cmd or "").strip().lower()
        if normalized == "next_pose":
            if self._active_workflow_tab == "pose":
                self.on_pose_next()
        elif normalized == "prev_pose":
            if self._active_workflow_tab == "pose":
                self.on_pose_prev()
        elif normalized in ("start_session", "open_posing"):
            if self._active_workflow_tab == "pose" and not self._pose_session_active:
                self.on_pose_toggle_session()
        elif normalized in ("stop_session",):
            if self._active_workflow_tab == "pose" and self._pose_session_active:
                self.on_pose_toggle_session()
        elif normalized in ("start_scan", "scan_3d"):
            self.on_scan4_start()
        elif normalized in ("stop_scan",):
            if self._metrics_tab is not None:
                self._metrics_tab.stop_scan4_session()
                self._scan_status_var.set(self._metrics_tab.latest_status_message() or "Scan stopped.")

    def _capture_pose_snapshot(
        self,
        frame_bgr: Optional[np.ndarray],
        pose_result: PoseResult,
        score: Optional[float],
        manual: bool = False,
    ) -> None:
        profile = self._active_profile()
        if profile is None or frame_bgr is None:
            return
        pose_key = self._current_pose_key()
        pose_name = self._current_pose_label()
        item = self._current_pose_item()
        payload: dict[str, object] = {
            "capture_id": uuid.uuid4().hex,
            "pose": pose_key,
            "pose_id": pose_key,
            "pose_name": pose_name,
            "pose_score": float(score or 0.0),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "profile_id": profile.profile_id,
            "manual_capture": bool(manual),
        }
        if item is not None:
            payload["federation_id"] = item.federation_id
            payload["division_id"] = item.division_id
            if item.class_id:
                payload["class_id"] = item.class_id

        full_path = self._session_store.save_capture(
            profile_name=profile.profile_name,
            payload=payload,
            frame=frame_bgr,
            variant="full",
        )
        cutout_path: Optional[Path] = None
        if pose_result.mask is not None:
            cutout = self._apply_cutout(frame_bgr, pose_result.mask)
            cutout_path = self._session_store.save_capture(
                profile_name=profile.profile_name,
                payload=payload,
                frame=cutout,
                variant="cutout",
            )
        entry = PoseCaptureEntry(
            score=float(score or 0.0),
            created_at=time.time(),
            full_path=full_path,
            cutout_path=cutout_path,
        )
        bucket = self._pose_best_captures.setdefault(pose_key, [])
        bucket.append(entry)
        bucket.sort(key=lambda item_: (item_.score, item_.created_at), reverse=True)
        dropped = bucket[3:]
        del bucket[3:]
        for drop in dropped:
            self._cleanup_pose_capture(drop)
        self._report_status_var.set(
            f"Capture saved for {pose_name}. Keeping top {len(bucket)} for pose '{pose_key}'."
        )

    def _cleanup_pose_capture(self, entry: PoseCaptureEntry) -> None:
        for path in [entry.full_path, entry.cutout_path]:
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except Exception:
                continue

    def _apply_cutout(self, frame: np.ndarray, mask: Optional[np.ndarray], bg_colour: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        if mask is None:
            return frame
        blur = cv2.GaussianBlur(mask, (9, 9), 0)
        alpha = blur.astype(np.float32) / 255.0
        alpha = alpha[..., None]
        bg = np.zeros_like(frame, dtype=np.float32)
        bg[:] = bg_colour
        out = frame.astype(np.float32) * alpha + bg * (1.0 - alpha)
        return out.astype(np.uint8)

    def _draw_pose_example_inset(self, frame_bgr: np.ndarray, pose_key: str) -> np.ndarray:
        guide = POSE_GUIDES.get(str(pose_key or "").strip())
        if not guide:
            return frame_bgr

        out = frame_bgr.copy()
        h, w = out.shape[:2]
        inset_h = max(140, min(int(h * 0.34), h - 24))
        inset_w = max(110, min(int(inset_h * 0.62), w - 24))
        x0 = max(8, w - inset_w - 12)
        y0 = 14
        if y0 + inset_h > h - 8:
            y0 = max(8, h - inset_h - 8)

        panel = np.zeros((inset_h, inset_w, 3), dtype=np.uint8)
        panel[:, :] = (18, 18, 22)
        panel = draw_pose_guide(panel, guide, colour=(0, 220, 255), alpha=0.9)
        cv2.rectangle(panel, (0, 0), (inset_w - 1, inset_h - 1), (76, 76, 88), 1)
        cv2.putText(panel, "POSE EXAMPLE", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 220), 1, cv2.LINE_AA)

        roi = out[y0 : y0 + inset_h, x0 : x0 + inset_w]
        blended = cv2.addWeighted(roi, 0.25, panel, 0.75, 0)
        out[y0 : y0 + inset_h, x0 : x0 + inset_w] = blended
        return out

    def _legacy_profile_for_metrics(self, profile: AthleteProfile) -> SimpleNamespace:
        return SimpleNamespace(
            name=profile.profile_name,
            federation="WNBF_UK",
            bodyfat_percent=profile.bodyfat_percent,
            plan=SimpleNamespace(competition_date=profile.competition_date),
        )

    def _ensure_pose_backend(self) -> None:
        if self._pose_backend is not None:
            return
        if self._pose_backend_init_attempted:
            return
        self._pose_backend_init_attempted = True
        try:
            self._pose_backend = PoseBackend()
            self._pose_backend_error = ""
        except Exception as exc:
            self._pose_backend = None
            self._pose_backend_error = str(exc)

    def _apply_pose_coach(self, frame_bgr: np.ndarray, packet: Optional[dict]) -> np.ndarray:
        self._latest_frame_bgr = frame_bgr
        self._ensure_pose_backend()
        res: Optional[PoseResult] = None
        if self._pose_backend is not None:
            self._pose_frame_counter += 1
            if self._pose_last_result is None or (self._pose_frame_counter % 2) == 0:
                self._pose_last_result = self._pose_backend.process_bgr(frame_bgr)
            res = self._pose_last_result
        if res is None:
            self._pose_score_var.set("Score: —")
            self._pose_advice_var.set(self._pose_backend_error or "Pose backend unavailable.")
            return frame_bgr

        self._pose_last_result = res
        props = compute_from_mask(res.mask) if res.mask is not None else None
        self._pose_last_props = props

        profile = self._active_profile()
        if profile is not None and self._metrics_tab is not None:
            self._metrics_tab.height_cm = float(profile.height_cm)
            self._metrics_tab.weight_kg = float(profile.weight_kg)
            self._metrics_tab.update_live_context(
                frame=frame_bgr,
                mask=res.mask,
                props=props,
                profile=self._legacy_profile_for_metrics(profile),
                category="Bodybuilding",
                pose_key=self._current_pose_key(),
                landmarks=res.landmarks,
                depth=(packet or {}).get("depth"),
                intrinsics=((packet or {}).get("meta") or {}).get("intrinsics"),
                depth_aligned=((packet or {}).get("meta") or {}).get("depth_aligned"),
                depth_units=((packet or {}).get("meta") or {}).get("depth_units"),
            )

        pose_key = self._current_pose_key()
        pose_def = POSES.get(pose_key)
        score_value: Optional[float] = None
        advice: list[str] = []
        if pose_def is not None and res.landmarks:
            ps = score_pose(
                res.landmarks,
                pose_def.target,
                pose_def.tolerance,
                weights=pose_def.weights,
                props=props,
            )
            score_value = float(ps.score_0_100)
            advice = list(ps.advice)
        self._pose_last_score = score_value
        self._pose_last_advice = advice

        scan4_active = self._metrics_tab.is_scan4_active() if self._metrics_tab is not None else False
        if self._coach_voice_enabled_var.get() and self._tts_speaker is not None and not scan4_active:
            now_ts = time.time()
            if advice and score_value is not None and score_value < 95.0:
                line = advice[0]
                if line != self._coach_voice_last_line or (now_ts - self._coach_voice_last_ts) > 6.0:
                    self._tts_speaker.say(line)
                    self._coach_voice_last_line = line
                    self._coach_voice_last_ts = now_ts
            elif (now_ts - self._coach_voice_last_ts) > 7.5 and self._coach_voice_last_line:
                self._tts_speaker.say("Good. Hold it.")
                self._coach_voice_last_line = ""
                self._coach_voice_last_ts = now_ts

        if self._pose_timer_enabled_var.get():
            remaining = max(0, int(np.ceil(self._pose_timer_seconds - (time.time() - self._pose_timer_started_at))))
            self._pose_timer_var.set(f"Timer: {remaining}s")
            if self._pose_session_active and remaining <= 0:
                self.on_pose_next()
        else:
            self._pose_timer_var.set("Timer: OFF")

        if score_value is None:
            self._pose_score_var.set("Score: —")
        else:
            self._pose_score_var.set(f"Score: {score_value:.1f}/100")
        self._pose_advice_var.set(advice[0] if advice else "")

        if (
            self._pose_session_active
            and self._pose_auto_capture_var.get()
            and score_value is not None
            and score_value >= 70.0
            and (time.time() - self._pose_last_capture_ts) >= 1.4
        ):
            self._capture_pose_snapshot(
                frame_bgr=frame_bgr,
                pose_result=res,
                score=score_value,
                manual=False,
            )
            self._pose_last_capture_ts = time.time()

        out = frame_bgr.copy()
        if res.mask is not None:
            out = draw_mask_outline(out, res.mask)
        if res.landmarks:
            joint_ok = {name: True for name in res.landmarks.keys()}
            line_ok: dict[tuple[str, str], bool] = {}
            out = draw_pose_overlay(out, res.landmarks, joint_ok=joint_ok, line_ok=line_ok)
        if self._pose_show_example_var.get():
            out = self._draw_pose_example_inset(out, pose_key)
        pose_line = f"POSE: {self._current_pose_label()}"
        score_line = "SCORE: —" if score_value is None else f"SCORE: {score_value:.1f}/100"
        cv2.putText(out, pose_line, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(out, score_line, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 220, 255), 2, cv2.LINE_AA)
        return out

    def _apply_scan_workflow(self, frame_bgr: np.ndarray, packet: Optional[dict]) -> np.ndarray:
        self._latest_frame_bgr = frame_bgr
        self._ensure_pose_backend()
        res: Optional[PoseResult] = None
        if self._pose_backend is not None:
            self._pose_frame_counter += 1
            if self._pose_last_result is None or (self._pose_frame_counter % 2) == 0:
                self._pose_last_result = self._pose_backend.process_bgr(frame_bgr)
            res = self._pose_last_result
        if res is None:
            return frame_bgr

        self._pose_last_result = res
        props = compute_from_mask(res.mask) if res.mask is not None else None
        self._pose_last_props = props

        profile = self._active_profile()
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=False)
        if profile is not None and metrics_tab is not None:
            metrics_tab.height_cm = float(profile.height_cm)
            metrics_tab.weight_kg = float(profile.weight_kg)
            metrics_tab.update_live_context(
                frame=frame_bgr,
                mask=res.mask,
                props=props,
                profile=self._legacy_profile_for_metrics(profile),
                category="Bodybuilding",
                pose_key=self._current_pose_key(),
                landmarks=res.landmarks,
                depth=(packet or {}).get("depth"),
                intrinsics=((packet or {}).get("meta") or {}).get("intrinsics"),
                depth_aligned=((packet or {}).get("meta") or {}).get("depth_aligned"),
                depth_units=((packet or {}).get("meta") or {}).get("depth_units"),
            )
            return metrics_tab.render_scan_overlay(frame_bgr, mask=res.mask)
        return frame_bgr

    def _poll_scan_announcements(self) -> None:
        metrics_tab = self._metrics_tab
        if metrics_tab is None:
            return
        scan4_tts_reset = metrics_tab.pop_scan4_tts_reset()
        if scan4_tts_reset and self._tts_speaker is not None:
            self._tts_speaker.clear_pending()
        scan4_announce = metrics_tab.pop_scan4_announce()
        if (
            scan4_announce
            and self._coach_voice_enabled_var.get()
            and self._tts_speaker is not None
        ):
            self._tts_speaker.say_priority(scan4_announce, clear_pending=True)

    def on_scan4_start(self) -> None:
        snapshot = self.device_manager.snapshot()
        if snapshot.selected_source == InputSource.KINECT:
            depth_enabled = os.environ.get("BBCOACH_KINECT_ENABLE_DEPTH", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if not depth_enabled:
                os.environ["BBCOACH_KINECT_ENABLE_DEPTH"] = "1"
                if snapshot.state == DeviceState.STREAMING:
                    self._status_var.set("Enabling Kinect depth stream for 3D scan...")
                    self.root.update_idletasks()
                    self._apply_device_snapshot(self.device_manager.disconnect())
                    self._apply_device_snapshot(self.device_manager.connect())
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None:
            return
        metrics_tab.start_scan4_session()
        self._scan_status_var.set(metrics_tab.latest_status_message() or "4-side scan started.")

    def on_scan4_load(self) -> None:
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None:
            return
        metrics_tab.load_scan4_session()
        self._scan_status_var.set(metrics_tab.latest_status_message() or "Scan folder loaded.")

    def on_scan4_compute_metrics(self) -> None:
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None:
            return
        ok = metrics_tab.compute_scan4_metrics()
        msg = metrics_tab.latest_status_message()
        if not msg:
            msg = "Metrics computed." if ok else "Metrics could not be computed."
        self._scan_status_var.set(msg)

    def on_scan4_auto_merge(self) -> None:
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None:
            return
        metrics_tab.auto_merge_scan4()
        self._scan_status_var.set(metrics_tab.latest_status_message() or "Auto merge executed.")

    def on_scan4_manual_align(self) -> None:
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None:
            return
        metrics_tab.manual_align_scan4()
        self._scan_status_var.set(metrics_tab.latest_status_message() or "Manual align executed.")

    def on_scan4_export(self) -> None:
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None:
            return
        if not self._ensure_scan_dir_for_export():
            return
        metrics_tab.export_scan4_report()
        self._scan_status_var.set(metrics_tab.latest_status_message() or "Export finished.")
        self._report_status_var.set(self._scan_status_var.get())

    def on_generate_pdf_report(self) -> None:
        self.on_scan4_export()

    def on_scan4_open_folder(self) -> None:
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None or metrics_tab.current_scan_dir() is None:
            messagebox.showinfo("3D Scan", "No scan directory selected yet.", parent=self.root)
            return
        self._open_path_in_system(metrics_tab.current_scan_dir())

    def on_scan4_open_merged_pcd(self) -> None:
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None or metrics_tab.current_scan_dir() is None:
            messagebox.showinfo("3D Scan", "No scan directory selected yet.", parent=self.root)
            return
        merged = metrics_tab.current_scan_dir() / "exports" / "merged.pcd"
        if not merged.exists():
            messagebox.showinfo("Open merged.pcd", "merged.pcd not found yet. Run Auto Merge first.", parent=self.root)
            return
        self._open_path_in_system(merged)

    def on_open_sessions_folder(self) -> None:
        profile = self._active_profile()
        if profile is None:
            self._open_path_in_system(self._session_store.root)
            return
        user_dir = self._session_store.root / profile.profile_name
        if not user_dir.exists():
            user_dir.mkdir(parents=True, exist_ok=True)
        self._open_path_in_system(user_dir)

    def _ensure_scan_dir_for_export(self) -> bool:
        metrics_tab = self._metrics_tab_for_active_profile(require_profile=True)
        if metrics_tab is None:
            return False
        current = metrics_tab.current_scan_dir()
        if current is not None and current.exists():
            return True
        latest = self._find_latest_scan_dir_for_active_profile()
        if latest is None:
            messagebox.showinfo(
                "Generate PDF",
                "No completed 3D scan found for the active profile. Run a 3D scan first.",
                parent=self.root,
            )
            return False
        if not metrics_tab.set_scan4_dir(latest):
            messagebox.showerror(
                "Generate PDF",
                f"Could not load scan folder:\n{latest}",
                parent=self.root,
            )
            return False
        self._scan_status_var.set(metrics_tab.latest_status_message() or f"Loaded scan: {latest.name}")
        return True

    def _find_latest_scan_dir_for_active_profile(self) -> Optional[Path]:
        profile = self._active_profile()
        if profile is None:
            return None
        user_root = self._session_store.root / profile.profile_name
        if not user_root.exists():
            return None
        candidates: list[Path] = []
        for path in user_root.glob("*/metrics/*"):
            if not path.is_dir():
                continue
            raw = path / "raw"
            if raw.exists() and any(raw.glob("view_*")):
                candidates.append(path)
                continue
            # Backward-compatible fallback layouts.
            if any(path.glob("view_*")):
                candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _open_path_in_system(self, path: Optional[Path]) -> None:
        if path is None:
            return
        target = Path(path)
        if not target.exists():
            messagebox.showerror("Open Path", f"Path not found:\n{target}", parent=self.root)
            return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(target)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            elif sys.platform.startswith("win"):
                import os

                os.startfile(str(target))  # type: ignore[attr-defined]
            else:
                raise RuntimeError(f"Unsupported platform: {sys.platform}")
            self._report_status_var.set(f"Opened: {target}")
        except Exception as exc:
            messagebox.showerror("Open Path", f"Could not open path:\n{target}\n\n{exc}", parent=self.root)

    def _prune_selected_divisions_for_profile(
        self,
        profile: AthleteProfile,
        selected_refs: list[SelectedDivisionRef],
    ) -> bool:
        if not selected_refs:
            return False
        allowed = {row.ref.key() for row in self._all_division_choice_rows(profile)}
        filtered = [ref for ref in selected_refs if ref.key() in allowed]
        if len(filtered) == len(selected_refs):
            return False
        self.app_state.set_selected_divisions(filtered)
        return True

    def _active_profile(self) -> Optional[AthleteProfile]:
        return self.app_state.state.active_profile

    def _candidate_class_specs(self, division: DivisionSpec) -> list[object]:
        classes = list(getattr(division, "classes", []) or [])
        if classes:
            return classes
        classes.extend(getattr(getattr(division, "eligibility", None), "height_classes", []) or [])
        classes.extend(getattr(getattr(division, "eligibility", None), "weight_classes", []) or [])
        dedup: dict[str, object] = {}
        for cls in classes:
            cls_id = str(getattr(cls, "class_id", "") or "")
            if not cls_id:
                continue
            dedup[cls_id] = cls
        return list(dedup.values())

    def _matches_profile_sex(self, profile: AthleteProfile, division: DivisionSpec) -> bool:
        allowed = [str(item).strip().lower() for item in (division.eligibility.sex_categories or []) if str(item).strip()]
        if not allowed:
            return True
        profile_sex = str(profile.sex_category or "").strip().lower()
        if not profile_sex:
            return False
        return profile_sex in allowed

    def _matches_profile_sex_for_class(self, profile: AthleteProfile, class_spec: object) -> bool:
        profile_sex = str(profile.sex_category or "").strip().lower()
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

    def _division_rows_for_federation(self, fed: FederationSpec) -> list[DivisionChoiceRow]:
        profile = self._active_profile()
        rows: list[DivisionChoiceRow] = []
        for division in fed.divisions:
            if profile is not None and not self._matches_profile_sex(profile, division):
                continue
            classes = self._candidate_class_specs(division)
            if classes:
                for cls in classes:
                    if profile is not None and not self._matches_profile_sex_for_class(profile, cls):
                        continue
                    ref = SelectedDivisionRef(
                        federation_id=fed.federation_id,
                        division_id=division.division_id,
                        class_id=str(getattr(cls, "class_id", "") or None),
                    )
                    eligible = True
                    reasons: tuple[str, ...] = ()
                    if profile is not None:
                        result = evaluate_division_eligibility(profile, division, class_spec=cls)
                        eligible = bool(result.eligible)
                        reasons = tuple(result.reasons)
                    rows.append(
                        DivisionChoiceRow(
                            federation_name=fed.federation_name,
                            division_name=division.division_name,
                            class_name=str(getattr(cls, "class_name", "") or None),
                            ref=ref,
                            eligible=eligible,
                            reasons=reasons,
                        )
                    )
            else:
                ref = SelectedDivisionRef(federation_id=fed.federation_id, division_id=division.division_id)
                eligible = True
                reasons = ()
                if profile is not None:
                    result = evaluate_division_eligibility(profile, division, class_spec=None)
                    eligible = bool(result.eligible)
                    reasons = tuple(result.reasons)
                rows.append(
                    DivisionChoiceRow(
                        federation_name=fed.federation_name,
                        division_name=division.division_name,
                        class_name=None,
                        ref=ref,
                        eligible=eligible,
                        reasons=tuple(reasons),
                    )
                )
        return rows

    def _all_division_choice_rows(self, profile: AthleteProfile) -> list[DivisionChoiceRow]:
        rows: list[DivisionChoiceRow] = []
        for fed in self._spec_library.list_federations():
            for division in fed.divisions:
                if not self._matches_profile_sex(profile, division):
                    continue
                classes = self._candidate_class_specs(division)
                if classes:
                    for cls in classes:
                        if not self._matches_profile_sex_for_class(profile, cls):
                            continue
                        if not matches_age_window(
                            profile,
                            division,
                            class_spec=cls,
                            years_ahead=self._AGE_WINDOW_YEARS_AHEAD,
                        ):
                            continue
                        ref = SelectedDivisionRef(
                            federation_id=fed.federation_id,
                            division_id=division.division_id,
                            class_id=str(getattr(cls, "class_id", "") or None),
                        )
                        result = evaluate_division_eligibility(profile, division, class_spec=cls)
                        rows.append(
                            DivisionChoiceRow(
                                federation_name=fed.federation_name,
                                division_name=division.division_name,
                                class_name=str(getattr(cls, "class_name", "") or None),
                                ref=ref,
                                eligible=bool(result.eligible),
                                reasons=tuple(result.reasons),
                            )
                        )
                else:
                    if not matches_age_window(
                        profile,
                        division,
                        class_spec=None,
                        years_ahead=self._AGE_WINDOW_YEARS_AHEAD,
                    ):
                        continue
                    ref = SelectedDivisionRef(federation_id=fed.federation_id, division_id=division.division_id)
                    result = evaluate_division_eligibility(profile, division, class_spec=None)
                    rows.append(
                        DivisionChoiceRow(
                            federation_name=fed.federation_name,
                            division_name=division.division_name,
                            class_name=None,
                            ref=ref,
                            eligible=bool(result.eligible),
                            reasons=tuple(result.reasons),
                        )
                    )
        return rows

    def _federation_summary_key(
        self,
        profile: Optional[AthleteProfile],
        selected_refs: list[SelectedDivisionRef],
    ) -> tuple[object, ...]:
        selected_key = tuple(sorted(ref.key() for ref in selected_refs))
        if profile is None:
            return ("none", selected_key)
        return (
            profile.profile_id,
            profile.sex_category,
            profile.date_of_birth,
            profile.competition_date,
            profile.has_disability,
            profile.height_cm,
            profile.weight_kg,
            selected_key,
        )

    def _refresh_federation_category_summary(
        self,
        profile: Optional[AthleteProfile],
        selected_refs: Optional[list[SelectedDivisionRef]] = None,
    ) -> None:
        refs = list(selected_refs) if selected_refs is not None else list(self.app_state.state.selected_divisions)
        self._refresh_pose_selection_options(refs)
        selected = selected_division_labels(self._spec_library, refs)
        if selected:
            self._working_divisions_var.set("\n".join(selected))
            self._set_listbox_items(self._working_divisions_listbox, selected)
        else:
            self._working_divisions_var.set("None selected yet.")
            self._set_listbox_items(self._working_divisions_listbox, ["None selected yet."])
        if profile is None:
            self._eligible_divisions_var.set("Load profile to compute eligible categories.")
            self._set_listbox_items(self._eligible_divisions_listbox, ["Load profile to compute eligible categories."])
            return
        eligible_rows = [row.label for row in self._all_division_choice_rows(profile) if row.eligible]
        if not eligible_rows:
            self._eligible_divisions_var.set("No eligible categories found for this profile.")
            self._set_listbox_items(self._eligible_divisions_listbox, ["No eligible categories found for this profile."])
            return
        self._eligible_divisions_var.set("\n".join(eligible_rows))
        self._set_listbox_items(self._eligible_divisions_listbox, eligible_rows)

    def _set_listbox_items(self, listbox: Optional[tk.Listbox], rows: list[str]) -> None:
        if listbox is None:
            return
        listbox.delete(0, "end")
        for row in rows:
            listbox.insert("end", row)

    def _set_active_profile(self, profile: AthleteProfile) -> None:
        self.profile_store.save_active_profile(profile)
        self.app_state.set_active_profile(profile)

    def on_new_profile(self) -> None:
        dialog = ProfileEditorDialog(self.root, "New Profile", profile=None)
        profile = dialog.result
        if profile is None:
            return
        self._set_active_profile(profile)

    def on_edit_profile(self) -> None:
        active = self._active_profile()
        if active is None:
            messagebox.showinfo("Edit Profile", "No active profile. Create or switch to one first.", parent=self.root)
            return
        dialog = ProfileEditorDialog(self.root, "Edit Profile", profile=active)
        updated = dialog.result
        if updated is None:
            return
        updated.profile_id = active.profile_id
        self._set_active_profile(updated)

    def on_switch_profile(self) -> None:
        profiles = self.profile_store.list_profiles()
        if not profiles:
            messagebox.showinfo("Switch Profile", "No profiles found.", parent=self.root)
            return
        active_id = self.app_state.state.active_profile_id
        dialog = ProfilePickerDialog(self.root, "Switch Profile", profiles=profiles, active_profile_id=active_id)
        selected_id = dialog.result
        if selected_id is None:
            return
        selected_profile = self.profile_store.load(selected_id)
        self._set_active_profile(selected_profile)

    def on_delete_profile(self) -> None:
        profiles = self.profile_store.list_profiles()
        if not profiles:
            messagebox.showinfo("Delete Profile", "No profiles found.", parent=self.root)
            return
        active_id = self.app_state.state.active_profile_id
        dialog = ProfilePickerDialog(self.root, "Delete Profile", profiles=profiles, active_profile_id=active_id)
        selected_id = dialog.result
        if selected_id is None:
            return
        selected = next((p for p in profiles if p.profile_id == selected_id), None)
        if selected is None:
            return
        if not messagebox.askyesno(
            "Delete Profile - Step 1/3",
            f"You are about to delete profile '{selected.profile_name}'. Continue?",
            parent=self.root,
            icon="warning",
        ):
            return
        if not messagebox.askyesno(
            "Delete Profile - Step 2/3",
            "This action removes athlete profile data from the profiles store. Continue?",
            parent=self.root,
            icon="warning",
        ):
            return
        if not messagebox.askyesno(
            "Delete Profile - Final Step 3/3",
            "Final confirmation: permanently delete this profile?",
            parent=self.root,
            icon="warning",
        ):
            return
        self.profile_store.delete(selected.profile_id)
        next_active = self.profile_store.load_active_profile()
        self.app_state.set_active_profile(next_active)

    def on_create_demo_profile(self) -> None:
        existing = self.profile_store.list_profiles()
        name = build_demo_profile_name(existing)
        demo = self.profile_store.create(
            profile_name=name,
            sex_category="Male",
            date_of_birth="1998-06-15",
            height_cm=178.0,
            weight_kg=84.5,
            bodyfat_percent=10.8,
            has_disability=False,
            drug_tested_preference=True,
            competition_date=(date.today() + timedelta(days=126)).isoformat(),
        )
        self._set_active_profile(demo)

    def on_browse_rules(self) -> None:
        if not self._spec_library.federations:
            messagebox.showerror(
                "Federations/Categories",
                "No federation YAML specs loaded.",
                parent=self.root,
            )
            return
        win = tk.Toplevel(self.root)
        win.title("Browse Rules")
        win.geometry("1180x760")
        win.minsize(940, 620)

        main = ttk.Frame(win, padding=8)
        main.pack(fill="both", expand=True)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        tree = ttk.Treeview(main, columns=("kind",), show="tree")
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(main, orient="vertical", command=tree.yview)
        yscroll.grid(row=0, column=0, sticky="nse")
        tree.configure(yscrollcommand=yscroll.set)

        details = ScrolledText(main, wrap="word")
        details.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        details.configure(state="disabled")
        details_image_refs: list[tk.PhotoImage] = []

        item_to_ref: dict[str, SelectedDivisionRef] = {}
        for fed in self._spec_library.list_federations():
            fed_item = tree.insert("", "end", text=fed.federation_name, open=False)
            for row in self._division_rows_for_federation(fed):
                div_label = row.division_name
                if row.class_name:
                    div_label += f" / {row.class_name}"
                item = tree.insert(fed_item, "end", text=div_label)
                item_to_ref[item] = row.ref

        def _set_details_text(text: str) -> None:
            details_image_refs.clear()
            details.configure(state="normal")
            details.delete("1.0", "end")
            details.insert("1.0", text)
            details.configure(state="disabled")

        def _render_ref(ref: SelectedDivisionRef) -> None:
            fed = self._spec_library.get_federation(ref.federation_id)
            div = self._spec_library.get_division(ref)
            cls = self._spec_library.get_division_class(ref)
            if fed is None or div is None:
                _set_details_text("No details available.")
                return
            lines: list[str] = []
            lines.append(f"Federation: {fed.federation_name}")
            lines.append(f"Division: {div.division_name}")
            if cls is not None:
                lines.append(f"Class: {cls.class_name}")
            lines.append("")
            lines.append("Eligibility Requirements")
            if div.eligibility.sex_categories:
                lines.append(f"- Sex categories: {', '.join(div.eligibility.sex_categories)}")
            if getattr(div.eligibility, "disability_required", False):
                lines.append("- Disability requirement: category is open to disabled athletes only.")
            if div.eligibility.age_rule_text:
                lines.append(f"- Age rule: {div.eligibility.age_rule_text}")
            elif div.eligibility.age_min_years is not None or div.eligibility.age_max_years is not None:
                lines.append(
                    f"- Age bounds: {div.eligibility.age_min_years if div.eligibility.age_min_years is not None else '-'}"
                    f" to {div.eligibility.age_max_years if div.eligibility.age_max_years is not None else '-'}"
                )
            if div.eligibility.novice_restriction_text:
                lines.append(f"- Novice restriction: {div.eligibility.novice_restriction_text}")
            if cls is not None:
                if cls.min_height_cm is not None or cls.max_height_cm is not None:
                    lines.append(
                        f"- Class height: {cls.min_height_cm if cls.min_height_cm is not None else '-'}"
                        f" to {cls.max_height_cm if cls.max_height_cm is not None else '-'} cm"
                    )
                if cls.min_weight_kg is not None or cls.max_weight_kg is not None:
                    lines.append(
                        f"- Class weight: {cls.min_weight_kg if cls.min_weight_kg is not None else '-'}"
                        f" to {cls.max_weight_kg if cls.max_weight_kg is not None else '-'} kg"
                    )
                if cls.notes:
                    lines.append(f"- Class notes: {cls.notes}")
            if len(lines) == 4:
                lines.append("- Not specified")
            lines.append("")
            lines.append("Rounds")
            if div.rounds:
                for rnd in div.rounds:
                    row_txt = f"- {rnd.round_name}"
                    if rnd.judged_on:
                        row_txt += f": {rnd.judged_on}"
                    if rnd.routine_time_limit_seconds is not None:
                        row_txt += f" ({rnd.routine_time_limit_seconds}s)"
                    lines.append(row_txt)
            else:
                lines.append("- Not specified")
            lines.append("")
            lines.append("Mandatory Poses")
            if div.mandatory_poses:
                for pose in div.mandatory_poses:
                    lines.append(f"- {pose}")
            else:
                lines.append("- Not specified")
            lines.append("")
            lines.append("Attire")
            lines.append(div.attire or "Not specified")
            lines.append("")
            lines.append("Sources")
            for src in (div.sources.source_urls + div.sources.source_files):
                lines.append(f"- {src}")
            if div.needs_verification:
                lines.append("")
                lines.append("needs_verification=true")

            details.configure(state="normal")
            details.delete("1.0", "end")
            details.insert("1.0", "\n".join(lines))

            repo_root = Path(__file__).resolve().parents[2]
            if div.rule_images:
                details.insert("end", "\n\nRule Photos\n")
                details_image_refs.clear()
                for idx, image_ref in enumerate(div.rule_images, start=1):
                    image_path = Path(image_ref.image_path)
                    if not image_path.is_absolute():
                        image_path = repo_root / image_path
                    caption = image_ref.caption or f"Image {idx}"
                    if image_ref.page_number is not None:
                        caption = f"{caption} (page {image_ref.page_number})"
                    details.insert("end", f"- {caption}\n")
                    if image_ref.source_file:
                        details.insert("end", f"  source: {image_ref.source_file}\n")
                    if not image_path.exists():
                        details.insert("end", f"  missing file: {image_ref.image_path}\n\n")
                        continue
                    try:
                        photo = tk.PhotoImage(file=str(image_path))
                        width = max(1, int(photo.width()))
                        height = max(1, int(photo.height()))
                        scale = max(1, int(max(width / 460.0, height / 340.0)))
                        if scale > 1:
                            photo = photo.subsample(scale, scale)
                        details_image_refs.append(photo)
                        details.image_create("end", image=photo)
                        details.insert("end", "\n\n")
                    except Exception:
                        details.insert("end", f"  could not render image: {image_ref.image_path}\n\n")

            details.configure(state="disabled")

        def _on_tree_select(_evt: Optional[tk.Event] = None) -> None:
            sel = tree.selection()
            if not sel:
                return
            ref = item_to_ref.get(sel[0])
            if ref is None:
                _set_details_text("Select a division/class to see details.")
                return
            _render_ref(ref)

        tree.bind("<<TreeviewSelect>>", _on_tree_select)
        _set_details_text("Select a federation division/class from the tree.")

    def on_select_categories(self) -> None:
        profile = self._active_profile()
        if profile is None:
            messagebox.showinfo("Select Categories", "Load or create a profile first.", parent=self.root)
            return
        rows = self._all_division_choice_rows(profile)
        if not rows:
            messagebox.showinfo(
                "Select Categories",
                (
                    "No categories match the athlete age window "
                    f"(competition year + next {self._AGE_WINDOW_YEARS_AHEAD} years)."
                ),
                parent=self.root,
            )
            return
        selected_keys = {ref.key() for ref in self.app_state.state.selected_divisions}
        dialog = CategorySelectionDialog(self.root, "Select Categories", rows=rows, selected_keys=selected_keys)
        if dialog.result is None:
            return
        self.app_state.set_selected_divisions(dialog.result)

    def on_select_all_eligible_categories(self) -> None:
        profile = self._active_profile()
        if profile is None:
            messagebox.showinfo("Select All Eligible", "Load or create a profile first.", parent=self.root)
            return
        rows = [row for row in self._all_division_choice_rows(profile) if row.eligible]
        self.app_state.set_selected_divisions([row.ref for row in rows])

    def on_clear_selected_divisions(self) -> None:
        self.app_state.set_selected_divisions([])

    def on_reset_session_metadata(self) -> None:
        snapshot = self.device_manager.snapshot()
        self.app_state.clear_session_metadata()
        self.app_state.set_session_metadata(
            running=(snapshot.state == DeviceState.STREAMING),
            device_state=snapshot.state.value,
            input_source=self._source_label(snapshot.selected_source),
            device_message=snapshot.status_message,
        )

    def on_toggle_left_panel(self) -> None:
        if self._show_left_var.get():
            self.left_panel.grid()
        else:
            self.left_panel.grid_remove()

    def on_toggle_right_panel(self) -> None:
        if self._show_right_var.get():
            self.right_panel.grid()
        else:
            self.right_panel.grid_remove()

    def on_reset_layout(self) -> None:
        self._show_left_var.set(True)
        self._show_right_var.set(True)
        self.left_panel.grid()
        self.right_panel.grid()

    def on_about(self) -> None:
        year = date.today().year
        messagebox.showinfo(
            "About",
            (
                "Bodybuilding Coach\n"
                f"Version: {__version__}\n"
                f"Author/Creator: Edwin Herrera\n"
                f"Copyright (c) {year} Edwin Herrera"
            ),
            parent=self.root,
        )

    def on_help_quick_start(self) -> None:
        messagebox.showinfo(
            "Quick Start",
            (
                "1. File > New Profile (or Switch Profile).\n"
                "2. Device > Input Source (Webcam/Kinect).\n"
                "3. Device > Power On.\n"
                "4. Federations/Categories > Select Categories.\n"
                "5. Use Pose Coach to run pose session and scoring.\n"
                "6. Use 3D Scan / Point Cloud to capture and process scan data.\n"
                "7. Reports / Exports to open session folder or generate PDF from 3D scan."
            ),
            parent=self.root,
        )

    def on_help_shortcuts(self) -> None:
        messagebox.showinfo(
            "Keyboard Shortcuts",
            (
                "F11: Video fullscreen\n"
                "Esc: Exit video fullscreen\n\n"
                "Additional controls may be available in the video modules\n"
                "(Pose Coach / 3D Scan) depending on active mode."
            ),
            parent=self.root,
        )

    def on_help_disclaimer(self) -> None:
        messagebox.showinfo(
            "Data & Disclaimer",
            (
                "Federation/category rules in this app come from local summary sources\n"
                "(docs/federations_summary.pdf and docs/federations_summary.tex).\n\n"
                "If a rule detail is unclear or missing, it is marked for verification.\n"
                "This software is a training aid and does not replace official federation\n"
                "rulebooks, judges, medical advice, or legal guidance."
            ),
            parent=self.root,
        )

    def on_toggle_video_fullscreen(self) -> None:
        enabled = bool(self._video_fullscreen_var.get())
        if enabled:
            self._enter_video_only_fullscreen()
        else:
            self._exit_video_only_fullscreen()

    def _enter_video_only_fullscreen(self) -> None:
        if self._video_only_active:
            return
        self._video_only_active = True
        self.device_panel.grid_remove()
        self.main_frame.grid_remove()
        self.status_bar.grid_remove()
        self.video_only_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.attributes("-fullscreen", True)
        if self._hidden_menu is None:
            self._hidden_menu = tk.Menu(self.root, tearoff=0)
        self.root.configure(menu=self._hidden_menu)
        self._render_center_view(self.device_manager.snapshot())

    def _exit_video_only_fullscreen(self) -> None:
        if not self._video_only_active:
            self.root.attributes("-fullscreen", False)
            self.root.configure(menu=self._menubar)
            return
        self._video_only_active = False
        self.root.attributes("-fullscreen", False)
        self.root.configure(menu=self._menubar)
        self.video_only_frame.grid_remove()
        self.device_panel.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        self.main_frame.grid(row=1, column=0, sticky="nsew")
        self.status_bar.grid(row=2, column=0, sticky="ew")
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        if not self._show_left_var.get():
            self.left_panel.grid_remove()
        if not self._show_right_var.get():
            self.right_panel.grid_remove()
        self._render_center_view(self.device_manager.snapshot())

    def _on_f11(self, _event: Optional[tk.Event] = None) -> str:
        self._video_fullscreen_var.set(not self._video_fullscreen_var.get())
        self.on_toggle_video_fullscreen()
        return "break"

    def _on_escape(self, _event: Optional[tk.Event] = None) -> Optional[str]:
        if not self._video_fullscreen_var.get():
            return None
        self._video_fullscreen_var.set(False)
        self.on_toggle_video_fullscreen()
        return "break"

    def on_device_source_selected(self, _event: Optional[tk.Event] = None) -> None:
        label = self._device_source_var.get().strip()
        source = self._SOURCE_LABEL_TO_ENUM.get(label, InputSource.NONE)
        snapshot = self.device_manager.select_source(source)
        if snapshot.state != DeviceState.ERROR:
            self._clear_error_details()
        self._apply_device_snapshot(snapshot)

    def on_device_power_toggle(self) -> None:
        snapshot = self.device_manager.snapshot()
        if snapshot.state in (DeviceState.STREAMING, DeviceState.CONNECTING):
            next_snapshot = self.device_manager.disconnect()
            self._clear_error_details()
            self._apply_device_snapshot(next_snapshot)
            return
        if snapshot.selected_source == InputSource.NONE:
            # If a webcam device path/index is already selected, auto-switch source to Webcam.
            cam = self.device_manager.camera
            if cam not in (None, ""):
                snapshot = self.device_manager.select_source(InputSource.WEBCAM)
            else:
                messagebox.showinfo(
                    "Device Power",
                    "Select an input source first from Device > Input Source.",
                    parent=self.root,
                )
                return
        self._sample_frame_rgb = None
        self._sample_input_path = None
        self._sample_kind = ""
        self._set_badge(DeviceState.CONNECTING)
        self._device_message_var.set(f"Connecting to {self._source_label(snapshot.selected_source)}...")
        self.root.update_idletasks()
        next_snapshot = self.device_manager.connect()
        if next_snapshot.state != DeviceState.ERROR:
            self._clear_error_details()
        self._apply_device_snapshot(next_snapshot)

    def _camera_menu_value(self) -> str:
        camera = self.device_manager.camera
        if camera in (None, ""):
            return ""
        if isinstance(camera, int):
            return f"/dev/video{camera}"
        text = str(camera).strip()
        if text.isdigit():
            return f"/dev/video{text}"
        return text

    def _probe_kinect_available(self, force: bool = False) -> bool:
        now = time.time()
        if (not force) and (now - self._last_kinect_probe_ts) < 2.0:
            return self._last_kinect_probe_available
        self._last_kinect_probe_ts = now
        try:
            from ..vision import capture_kinect

            pfn = capture_kinect._load_backend()  # type: ignore[attr-defined]
            fn = pfn.Freenect2()
            count = int(fn.enumerateDevices())
            available = count > 0
            self._last_kinect_probe_available = available
            if available:
                self._kinect_available_var.set(f"Kinect: detected ({count})")
            else:
                self._kinect_available_var.set("Kinect: not detected")
            return available
        except Exception:
            self._last_kinect_probe_available = False
            self._kinect_available_var.set("Kinect: backend unavailable")
            return False

    def _refresh_webcam_devices_menu(self) -> None:
        if self._webcam_devices_menu is None:
            return
        menu = self._webcam_devices_menu
        menu.delete(0, "end")

        current_value = self._camera_menu_value()
        self._webcam_device_var.set(current_value)
        kinect_available = self._probe_kinect_available(force=True)

        if kinect_available:
            menu.add_command(label="Kinect v2 (detected) — select source", command=self.on_select_kinect_source)
        else:
            menu.add_command(label="Kinect v2 (not detected)", state="disabled")
        menu.add_separator()
        menu.add_command(label="Webcams (/dev/video*)", state="disabled")

        try:
            nodes = sorted(self.device_manager.webcam_nodes_provider(), key=lambda p: str(p))
        except Exception:
            nodes = []

        if not nodes:
            menu.add_command(label="No /dev/video* detected", state="disabled")
        else:
            for node in nodes:
                node_text = str(node)
                menu.add_radiobutton(
                    label=node_text,
                    value=node_text,
                    variable=self._webcam_device_var,
                    command=lambda dev=node_text: self.on_select_webcam_device_from_menu(dev),
                )
        menu.add_separator()
        menu.add_command(label="Refresh Device List", command=self._refresh_webcam_devices_menu)

    def on_select_kinect_source(self) -> None:
        if not self._probe_kinect_available(force=True):
            messagebox.showinfo("Kinect", "No Kinect detected right now.", parent=self.root)
            return
        snapshot = self.device_manager.select_source(InputSource.KINECT)
        if snapshot.state != DeviceState.ERROR:
            self._clear_error_details()
        self._apply_device_snapshot(snapshot)
        self._status_var.set("Kinect source selected. Use Device > Power On.")
        if os.environ.get("BBCOACH_KINECT_ENABLE_DEPTH", "").strip().lower() not in ("1", "true", "yes", "on"):
            self._kinect_available_var.set("Kinect: detected (safe RGB mode, depth off)")

    def on_select_webcam_device_from_menu(self, device_path: str) -> None:
        text = str(device_path or "").strip()
        if not text:
            return
        self.device_manager.camera = text
        self._webcam_device_var.set(text)
        snapshot = self.device_manager.select_source(InputSource.WEBCAM)
        if snapshot.state != DeviceState.ERROR:
            self._clear_error_details()
        self._apply_device_snapshot(snapshot)
        self._status_var.set(f"Webcam device set to {text}. Source switched to Webcam. Use Device > Power On.")

    def on_select_webcam_device(self) -> None:
        current = self.device_manager.camera
        initial = "" if current is None else str(current)
        value = simpledialog.askstring(
            "Webcam Device",
            "Enter webcam device index or path (e.g. 0 or /dev/video2):",
            initialvalue=initial,
            parent=self.root,
        )
        if value is None:
            return
        text = value.strip()
        if not text:
            messagebox.showerror("Webcam Device", "Value is required.", parent=self.root)
            return
        self.device_manager.camera = int(text) if text.isdigit() else text
        self._webcam_device_var.set(self._camera_menu_value())
        snapshot = self.device_manager.select_source(InputSource.WEBCAM)
        if snapshot.state != DeviceState.ERROR:
            self._clear_error_details()
        self._apply_device_snapshot(snapshot)
        self._status_var.set(
            f"Webcam device set to {self.device_manager.camera}. Source switched to Webcam. Use Device > Power On."
        )

    def on_set_resolution(self) -> None:
        current_w = int(getattr(self.device_manager, "width", self.width))
        current_h = int(getattr(self.device_manager, "height", self.height))
        width = simpledialog.askinteger(
            "Resolution",
            "Width:",
            initialvalue=current_w,
            minvalue=320,
            maxvalue=7680,
            parent=self.root,
        )
        if width is None:
            return
        height = simpledialog.askinteger(
            "Resolution",
            "Height:",
            initialvalue=current_h,
            minvalue=240,
            maxvalue=4320,
            parent=self.root,
        )
        if height is None:
            return
        self.width = int(width)
        self.height = int(height)
        self.device_manager.width = int(width)
        self.device_manager.height = int(height)
        self._status_var.set(f"Resolution set to {width}x{height}. Reconnect to apply.")

    def on_device_connect(self) -> None:
        self._sample_frame_rgb = None
        self._sample_input_path = None
        self._sample_kind = ""
        current = self.device_manager.snapshot()
        if current.selected_source != InputSource.NONE:
            self._set_badge(DeviceState.CONNECTING)
            self._device_message_var.set(f"Connecting to {self._source_label(current.selected_source)}...")
            self.root.update_idletasks()
        snapshot = self.device_manager.connect()
        self._apply_device_snapshot(snapshot)

    def on_device_disconnect(self) -> None:
        snapshot = self.device_manager.disconnect()
        self._apply_device_snapshot(snapshot)

    def on_device_retry(self) -> None:
        current = self.device_manager.snapshot()
        if current.selected_source != InputSource.NONE:
            self._set_badge(DeviceState.CONNECTING)
            self._device_message_var.set(f"Connecting to {self._source_label(current.selected_source)}...")
            self.root.update_idletasks()
        snapshot = self.device_manager.retry()
        self._apply_device_snapshot(snapshot)

    def on_use_sample_input(self) -> None:
        snapshot = self.device_manager.snapshot()
        if snapshot.state == DeviceState.STREAMING:
            return
        loaded = self._load_sample_input_frame()
        if loaded is None:
            messagebox.showinfo(
                "Sample Input",
                "No sample input found in data/, bbcoach/data/, or sessions/.",
                parent=self.root,
            )
            return
        path, kind, frame_bgr = loaded
        self._sample_input_path = path
        self._sample_kind = kind
        self._sample_frame_rgb = self._to_rgb(frame_bgr)
        self._offline_sample_var.set(f"Sample input loaded: {path}")
        if snapshot.state == DeviceState.ERROR:
            snapshot = self.device_manager.select_source(InputSource.NONE)
        self._apply_device_snapshot(snapshot)

    def on_toggle_error_details(self) -> None:
        self._error_details_visible = not self._error_details_visible
        if self._error_details_visible:
            self.error_toggle_button.configure(text="Hide technical details")
            self.error_details.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        else:
            self.error_toggle_button.configure(text="Show technical details")
            self.error_details.grid_remove()

    def _clear_error_details(self) -> None:
        self._error_details_visible = False
        self.error_toggle_button.configure(text="Show technical details")
        self.error_details.grid_remove()

    def _schedule_ui_tick(self) -> None:
        self._ui_tick_after_id = self.root.after(33, self._on_ui_tick)

    def _on_ui_tick(self) -> None:
        self._poll_device_stream()
        self._schedule_ui_tick()

    def _poll_device_stream(self) -> None:
        self._poll_voice_commands()
        snapshot_before = self.device_manager.snapshot()
        if self._metrics_tab is not None:
            msg = self._metrics_tab.latest_status_message()
            if msg and msg != self._metrics_status_last:
                self._metrics_status_last = msg
                self._scan_status_var.set(msg)
        if snapshot_before.state == DeviceState.STREAMING:
            frame, packet = self.device_manager.read_stream_frame()
            if frame is not None:
                if self._active_workflow_tab == "scan":
                    annotated = self._apply_scan_workflow(frame, packet)
                else:
                    annotated = self._apply_pose_coach(frame, packet)
                annotated = self._maybe_mirror_for_display(annotated, snapshot_before.selected_source)
                frame_rgb = self._to_rgb(annotated)
                view_w = max(640, self.video_label.winfo_width())
                view_h = max(360, self.video_label.winfo_height())
                fitted = self._fit_frame(frame_rgb, view_w, view_h)
                self._set_video_image(fitted)
                if self._video_only_active:
                    full_w = max(640, self.video_only_label.winfo_width())
                    full_h = max(360, self.video_only_label.winfo_height())
                    full_fitted = self._fit_frame(frame_rgb, full_w, full_h)
                    self._set_video_only_image(full_fitted)
                self._frame_count += 1
                self.app_state.set_session_metadata(
                    running=True,
                    device_state=snapshot_before.state.value,
                    input_source=self._source_label(snapshot_before.selected_source),
                    device_message=snapshot_before.status_message,
                    frame_count=self._frame_count,
                    depth_available=bool(packet and packet.get("depth") is not None),
                    timestamp=(packet or {}).get("timestamp"),
                    pose_score=self._pose_last_score,
                    current_pose=self._current_pose_label(),
                )
        self._poll_scan_announcements()

        snapshot_after = self.device_manager.snapshot()
        key = (
            snapshot_after.state.value,
            snapshot_after.selected_source.value,
            snapshot_after.status_message,
        )
        if key != self._last_device_key:
            self._apply_device_snapshot(snapshot_after)

    def _apply_device_snapshot(self, snapshot: DeviceSnapshot) -> None:
        self._last_device_key = (
            snapshot.state.value,
            snapshot.selected_source.value,
            snapshot.status_message,
        )
        self._sync_source_combo(snapshot.selected_source)
        self._device_message_var.set(snapshot.status_message)
        self._set_badge(snapshot.state)
        self._sync_device_buttons(snapshot)
        self._render_center_view(snapshot)
        self.app_state.set_session_metadata(
            running=(snapshot.state == DeviceState.STREAMING),
            device_state=snapshot.state.value,
            input_source=self._source_label(snapshot.selected_source),
            device_message=snapshot.status_message,
        )

    def _sync_source_combo(self, source: InputSource) -> None:
        wanted = self._SOURCE_ENUM_TO_LABEL[source]
        if self._device_source_var.get() != wanted:
            self._device_source_var.set(wanted)

    def _set_badge(self, state: DeviceState) -> None:
        colours = {
            DeviceState.DISCONNECTED: ("#4b5563", "white"),
            DeviceState.CONNECTING: ("#b45309", "white"),
            DeviceState.STREAMING: ("#166534", "white"),
            DeviceState.ERROR: ("#991b1b", "white"),
        }
        bg, fg = colours.get(state, ("#4b5563", "white"))
        self.badge_label.configure(text=state.value, bg=bg, fg=fg)

    def _sync_device_buttons(self, snapshot: DeviceSnapshot) -> None:
        power_label = (
            "Power Off"
            if snapshot.state in (DeviceState.STREAMING, DeviceState.CONNECTING)
            else "Power On"
        )
        self._power_button_label_var.set(power_label)
        if self._device_menu is None or self._device_power_menu_index is None:
            return
        self._device_menu.entryconfigure(self._device_power_menu_index, label=power_label)
        if self._device_retry_menu_index is not None:
            retry_state = "normal" if snapshot.selected_source != InputSource.NONE else "disabled"
            self._device_menu.entryconfigure(self._device_retry_menu_index, state=retry_state)

    def _render_center_view(self, snapshot: DeviceSnapshot) -> None:
        if snapshot.state == DeviceState.ERROR:
            self._preview_caption_var.set("")
            self._error_message_var.set(snapshot.status_message)
            self._error_steps_var.set("Please connect the device and press Retry.")
            if self._video_only_active:
                self._set_video_only_text(snapshot.status_message)
            details = (snapshot.technical_details or "").strip()
            logs = self.device_manager.read_recent_logs(max_lines=60).strip()
            full_details = details
            if logs:
                full_details = f"{full_details}\n\n--- sessions/logs/device.log (tail) ---\n{logs}".strip()
            if not full_details:
                full_details = "No technical details available."
            self._set_error_details_text(full_details)
            self.error_frame.tkraise()
            return

        if snapshot.state == DeviceState.STREAMING:
            self._preview_caption_var.set(f"{self._source_label(snapshot.selected_source)} live preview")
            self.preview_frame.tkraise()
            if self._video_only_active:
                self._set_video_only_text("Waiting for frames...")
            return

        if self._sample_frame_rgb is not None:
            self._preview_caption_var.set(f"Sample input preview ({self._sample_kind})")
            view_w = max(640, self.video_label.winfo_width())
            view_h = max(360, self.video_label.winfo_height())
            fitted = self._fit_frame(self._sample_frame_rgb, view_w, view_h)
            self._set_video_image(fitted)
            if self._video_only_active:
                full_w = max(640, self.video_only_label.winfo_width())
                full_h = max(360, self.video_only_label.winfo_height())
                full_fitted = self._fit_frame(self._sample_frame_rgb, full_w, full_h)
                self._set_video_only_image(full_fitted)
            self.preview_frame.tkraise()
            return

        if snapshot.state == DeviceState.CONNECTING:
            self._offline_hint_var.set("Connecting...")
        elif snapshot.selected_source == InputSource.NONE:
            self._offline_hint_var.set("Offline: no live input selected")
        else:
            self._offline_hint_var.set(
                f"Offline: {self._source_label(snapshot.selected_source)} selected. Use Device > Power On."
            )
        if self._sample_input_path:
            self._offline_sample_var.set(f"Sample ready: {self._sample_input_path}")
        else:
            self._offline_sample_var.set("Use 'Use sample input' to test offline without hardware.")
        if self._video_only_active:
            self._set_video_only_text(self._offline_hint_var.get())
        self.offline_frame.tkraise()

    def _set_error_details_text(self, text: str) -> None:
        self.error_details.configure(state="normal")
        self.error_details.delete("1.0", "end")
        self.error_details.insert("1.0", text)
        self.error_details.configure(state="disabled")

    def _to_rgb(self, frame: np.ndarray) -> np.ndarray:
        arr = frame
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def _maybe_mirror_for_display(self, frame_bgr: np.ndarray, source: InputSource) -> np.ndarray:
        # Keep scoring/metrics on the original frame; mirror only the rendered preview.
        if source == InputSource.WEBCAM:
            return cv2.flip(frame_bgr, 1)
        return frame_bgr

    def _fit_frame(self, frame_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        if h <= 0 or w <= 0:
            return frame_rgb
        scale = min(target_w / float(w), target_h / float(h))
        scaled_w = max(1, int(w * scale))
        scaled_h = max(1, int(h * scale))
        resized = cv2.resize(frame_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        off_x = max(0, (target_w - scaled_w) // 2)
        off_y = max(0, (target_h - scaled_h) // 2)
        canvas[off_y : off_y + scaled_h, off_x : off_x + scaled_w] = resized
        return canvas

    def _set_video_image(self, frame_rgb: np.ndarray) -> None:
        h, w = frame_rgb.shape[:2]
        ppm_header = f"P6 {w} {h} 255\n".encode("ascii")
        data = ppm_header + frame_rgb.tobytes()
        photo = tk.PhotoImage(data=data, format="PPM")
        self._video_photo = photo
        self.video_label.configure(image=photo, text="")

    def _set_video_only_image(self, frame_rgb: np.ndarray) -> None:
        h, w = frame_rgb.shape[:2]
        ppm_header = f"P6 {w} {h} 255\n".encode("ascii")
        data = ppm_header + frame_rgb.tobytes()
        photo = tk.PhotoImage(data=data, format="PPM")
        self._video_only_photo = photo
        self.video_only_label.configure(image=photo, text="")

    def _set_video_only_text(self, text: str) -> None:
        self._video_only_photo = None
        self.video_only_label.configure(image="", text=text)

    def _source_label(self, source: InputSource) -> str:
        return self._SOURCE_ENUM_TO_LABEL.get(source, "None (Offline)")

    def _load_sample_input_frame(self) -> Optional[tuple[Path, str, np.ndarray]]:
        sample = self._find_sample_input_file()
        if sample is None:
            return None
        path, kind = sample

        if kind == "video":
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                return path, kind, frame
            return None

        if kind == "image":
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is not None:
                return path, kind, frame
            return None

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Sample point-cloud/mesh loaded (preview placeholder)",
            (24, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(path),
            (24, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        return path, kind, frame

    def _find_sample_input_file(self) -> Optional[tuple[Path, str]]:
        repo_root = Path(__file__).resolve().parent.parent.parent
        roots = [
            repo_root / "data",
            repo_root / "bbcoach" / "data",
            repo_root / "sessions",
        ]
        unique_roots = [r for i, r in enumerate(roots) if r.exists() and r not in roots[:i]]

        for root in unique_roots:
            path = self._first_match(root, (".mp4", ".avi", ".mov", ".mkv"))
            if path is not None:
                return path, "video"
        for root in unique_roots:
            path = self._first_match(root, (".png", ".jpg", ".jpeg", ".bmp"))
            if path is not None:
                return path, "image"
        for root in unique_roots:
            path = self._first_match(root, (".pcd", ".ply", ".obj", ".stl", ".mesh"))
            if path is not None:
                return path, "pointcloud/mesh"
        return None

    def _first_match(self, root: Path, extensions: tuple[str, ...]) -> Optional[Path]:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                return path
        return None

    def _on_close(self) -> None:
        if self._ui_tick_after_id is not None:
            self.root.after_cancel(self._ui_tick_after_id)
            self._ui_tick_after_id = None
        if self._voice_listener is not None:
            try:
                self._voice_listener.stop()
            except Exception:
                pass
            self._voice_listener = None
        if self._tts_speaker is not None:
            try:
                self._tts_speaker.stop()
            except Exception:
                pass
            self._tts_speaker = None
        self.device_manager.shutdown()
        if self._state_subscription is not None:
            self.app_state.unsubscribe(self._state_subscription)
            self._state_subscription = None
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_desktop(
    source_kind: str = "none",
    camera: int | str | None = None,
    width: int = 1280,
    height: int = 720,
    state_store: Optional[AppStateStore] = None,
) -> None:
    app = DesktopApp(
        source_kind=source_kind,
        camera=camera,
        width=width,
        height=height,
        state_store=state_store,
    )
    app.run()
