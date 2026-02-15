from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import copy
import json
from pathlib import Path

from .athlete_profiles import AthleteProfile
from .federations.specs import SelectedDivisionRef, parse_selected_division_ref


@dataclass
class ApplicationState:
    active_profile_id: Optional[str] = None
    active_profile: Optional[AthleteProfile] = None
    selected_divisions: list[SelectedDivisionRef] = field(default_factory=list)
    session_metadata: dict[str, Any] = field(default_factory=dict)


StateSubscriber = Callable[[ApplicationState], None]


class AppStateStore:
    def __init__(
        self,
        initial_state: Optional[ApplicationState] = None,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._state = initial_state or ApplicationState()
        self._subscribers: dict[int, StateSubscriber] = {}
        self._next_subscriber_id = 1
        repo_root = Path(__file__).resolve().parent.parent
        self._persist_path = persist_path or (repo_root / "config" / "app_state.json")
        self._selected_divisions_default: list[SelectedDivisionRef] = []
        self._selected_divisions_by_profile: dict[str, list[SelectedDivisionRef]] = {}
        self._load_persisted_state()
        # Keep a predictable baseline before any profile is loaded.
        if not self._state.selected_divisions:
            self._state.selected_divisions = list(self._selected_divisions_default)

    @property
    def state(self) -> ApplicationState:
        return self._state

    def snapshot(self) -> ApplicationState:
        return copy.deepcopy(self._state)

    def subscribe(self, callback: StateSubscriber, emit_initial: bool = True) -> int:
        token = self._next_subscriber_id
        self._next_subscriber_id += 1
        self._subscribers[token] = callback
        if emit_initial:
            callback(self.snapshot())
        return token

    def unsubscribe(self, token: int) -> None:
        self._subscribers.pop(token, None)

    def _emit(self) -> None:
        snapshot = self.snapshot()
        for callback in list(self._subscribers.values()):
            try:
                callback(snapshot)
            except Exception:
                continue

    def set_active_profile(self, profile: Optional[AthleteProfile]) -> None:
        self._state.active_profile = copy.deepcopy(profile) if profile is not None else None
        profile_id = profile.profile_id if profile is not None else None
        self._state.active_profile_id = profile_id
        if profile_id is None:
            self._state.selected_divisions = list(self._selected_divisions_default)
        else:
            self._state.selected_divisions = list(
                self._selected_divisions_by_profile.get(str(profile_id), self._selected_divisions_default)
            )
        self._emit()

    def set_selected_divisions(self, divisions: list[Any]) -> None:
        normalized = self._normalize_selected_divisions(divisions)
        self._state.selected_divisions = normalized
        active_profile_id = str(self._state.active_profile_id or "").strip()
        if active_profile_id:
            self._selected_divisions_by_profile[active_profile_id] = list(normalized)
        else:
            self._selected_divisions_default = list(normalized)
        self._persist_selected_divisions()
        self._emit()

    def set_session_metadata(self, **metadata: Any) -> None:
        self._state.session_metadata.update(metadata)
        self._emit()

    def replace_session_metadata(self, metadata: dict[str, Any]) -> None:
        self._state.session_metadata = dict(metadata)
        self._emit()

    def clear_session_metadata(self) -> None:
        self._state.session_metadata = {}
        self._emit()

    def _persist_selected_divisions(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 2,
                "selected_divisions_default": [ref.to_dict() for ref in self._selected_divisions_default],
                "selected_divisions_by_profile": {
                    profile_id: [ref.to_dict() for ref in refs]
                    for profile_id, refs in self._selected_divisions_by_profile.items()
                },
            }
            self._persist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            return

    def _load_persisted_state(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        # Preferred schema (v2): profile-scoped selections.
        default_refs = self._normalize_selected_divisions(payload.get("selected_divisions_default", []))
        profile_map: dict[str, list[SelectedDivisionRef]] = {}
        raw_profile_map = payload.get("selected_divisions_by_profile")
        if isinstance(raw_profile_map, dict):
            for raw_profile_id, raw_refs in raw_profile_map.items():
                profile_id = str(raw_profile_id or "").strip()
                if not profile_id:
                    continue
                refs = self._normalize_selected_divisions(raw_refs if isinstance(raw_refs, list) else [])
                profile_map[profile_id] = refs
        # Backward compatibility migration from old global schema.
        if not default_refs and not profile_map:
            default_refs = self._normalize_selected_divisions(payload.get("selected_divisions", []))
        self._selected_divisions_default = default_refs
        self._selected_divisions_by_profile = profile_map
        self._state.selected_divisions = list(default_refs)

    def _normalize_selected_divisions(self, values: Any) -> list[SelectedDivisionRef]:
        normalized: list[SelectedDivisionRef] = []
        if not isinstance(values, list):
            return normalized
        for item in values:
            ref = parse_selected_division_ref(item)
            if ref is None:
                continue
            normalized.append(ref)
        return normalized


_GLOBAL_APP_STATE = AppStateStore()


def get_global_app_state() -> AppStateStore:
    return _GLOBAL_APP_STATE
