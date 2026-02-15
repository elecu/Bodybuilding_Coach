from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .specs import FederationSpecsLibrary, SelectedDivisionRef


@dataclass(frozen=True)
class PoseChecklistItem:
    federation_id: str
    federation_name: str
    division_id: str
    division_name: str
    class_id: Optional[str]
    class_name: Optional[str]
    pose_name: str
    pose_key: Optional[str]

    @property
    def division_label(self) -> str:
        if self.class_name:
            return f"{self.federation_name} / {self.division_name} / {self.class_name}"
        return f"{self.federation_name} / {self.division_name}"


_POSE_ALIASES: dict[str, str] = {
    "front double biceps": "bb_front_double_biceps",
    "front double bicep": "bb_front_double_biceps",
    "front lat spread": "bb_front_lat_spread",
    "side chest": "bb_side_chest",
    "side triceps": "bb_side_triceps",
    "side tricep": "bb_side_triceps",
    "back double biceps": "bb_back_double_biceps",
    "rear double biceps": "bb_back_double_biceps",
    "rear bicep": "bb_back_double_biceps",
    "back lat spread": "bb_back_lat_spread",
    "rear lat spread": "bb_back_lat_spread",
    "abdominals and thighs": "bb_abs_and_thighs",
    "abdominals and thigh": "bb_abs_and_thighs",
    "abs and thigh": "bb_abs_and_thighs",
    "abs and thighs": "bb_abs_and_thighs",
    "front pose": "mp_front",
    "front stance": "mp_front",
    "quarter turn right": "mp_left",
    "quarter turn to the right": "mp_left",
    "right side pose": "mp_left",
    "left side pose": "mp_right",
    "back pose": "mp_back",
    "quarter turn face right": "mp_left",
    "quarter turn face back": "mp_back",
    "quarter turn face front": "mp_front",
}


def map_pose_name_to_pose_key(pose_name: str, sequence_index: int = 0) -> Optional[str]:
    normalized = " ".join(str(pose_name or "").strip().lower().replace("-", " ").split())

    if normalized in _POSE_ALIASES:
        key = _POSE_ALIASES[normalized]
        if normalized in ("quarter turn right", "quarter turn to the right", "quarter turn face right"):
            # Alternate left/right side when repeated in the mandatory list.
            return "mp_left" if (sequence_index % 2 == 1) else "mp_right"
        return key

    if normalized.startswith("front double biceps"):
        return "bb_front_double_biceps"
    if normalized.startswith("front lat spread"):
        return "bb_front_lat_spread"
    if normalized.startswith("side chest"):
        return "bb_side_chest"
    if normalized.startswith("side triceps") or normalized.startswith("side tricep"):
        return "bb_side_triceps"
    if normalized.startswith("back double biceps") or normalized.startswith("rear double biceps"):
        return "bb_back_double_biceps"
    if normalized.startswith("back lat spread") or normalized.startswith("rear lat spread"):
        return "bb_back_lat_spread"
    if normalized.startswith("abs and thigh") or normalized.startswith("abdominals and thigh"):
        return "bb_abs_and_thighs"

    return None


def build_pose_checklist(
    library: FederationSpecsLibrary,
    selected_divisions: list[SelectedDivisionRef],
) -> list[PoseChecklistItem]:
    checklist: list[PoseChecklistItem] = []
    for ref in selected_divisions:
        fed = library.get_federation(ref.federation_id)
        div = library.get_division(ref)
        if fed is None or div is None:
            continue
        cls = library.get_division_class(ref)
        mandatory = div.mandatory_poses or []
        for idx, pose_name in enumerate(mandatory):
            checklist.append(
                PoseChecklistItem(
                    federation_id=fed.federation_id,
                    federation_name=fed.federation_name,
                    division_id=div.division_id,
                    division_name=div.division_name,
                    class_id=cls.class_id if cls is not None else None,
                    class_name=cls.class_name if cls is not None else None,
                    pose_name=pose_name,
                    pose_key=map_pose_name_to_pose_key(pose_name, sequence_index=idx),
                )
            )
    return checklist


def selected_division_labels(
    library: FederationSpecsLibrary,
    selected_divisions: list[SelectedDivisionRef],
) -> list[str]:
    labels: list[str] = []
    for ref in selected_divisions:
        fed = library.get_federation(ref.federation_id)
        div = library.get_division(ref)
        if fed is None or div is None:
            continue
        cls = library.get_division_class(ref)
        if cls is not None:
            labels.append(f"{fed.federation_name} / {div.division_name} / {cls.class_name}")
        else:
            labels.append(f"{fed.federation_name} / {div.division_name}")
    return labels
