from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


ConfidenceLevel = str


@dataclass(frozen=True)
class SelectedDivisionRef:
    federation_id: str
    division_id: str
    class_id: Optional[str] = None

    def key(self) -> str:
        if self.class_id:
            return f"{self.federation_id}:{self.division_id}:{self.class_id}"
        return f"{self.federation_id}:{self.division_id}"

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "federation_id": self.federation_id,
            "division_id": self.division_id,
            "class_id": self.class_id,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SelectedDivisionRef":
        return SelectedDivisionRef(
            federation_id=str(data.get("federation_id", "")).strip(),
            division_id=str(data.get("division_id", "")).strip(),
            class_id=(str(data.get("class_id")).strip() if data.get("class_id") not in (None, "") else None),
        )


@dataclass
class HeightToWeightCapSpec:
    cap_id: str
    class_id: Optional[str] = None
    min_height_cm: Optional[float] = None
    max_height_cm: Optional[float] = None
    formula_type: Optional[str] = None
    formula_plus_kg: Optional[float] = None
    max_weight_kg: Optional[float] = None
    text: Optional[str] = None
    needs_verification: bool = False

    def max_allowed_weight_kg(self, height_cm: Optional[float]) -> Optional[float]:
        if height_cm is None:
            return None
        h = float(height_cm)
        if self.min_height_cm is not None and h < float(self.min_height_cm):
            return None
        if self.max_height_cm is not None and h > float(self.max_height_cm):
            return None
        if self.formula_type == "height_minus_100_plus":
            return h - 100.0 + float(self.formula_plus_kg or 0.0)
        if self.max_weight_kg is not None:
            return float(self.max_weight_kg)
        return None


@dataclass
class DivisionClassSpec:
    class_id: str
    class_name: str
    min_height_cm: Optional[float] = None
    max_height_cm: Optional[float] = None
    min_weight_kg: Optional[float] = None
    max_weight_kg: Optional[float] = None
    notes: Optional[str] = None
    needs_verification: bool = False


@dataclass
class DivisionEligibilityRules:
    @dataclass
    class AgeRuleSemantics:
        rule_type: Optional[str] = None
        turn_age_years: Optional[int] = None
        reference_date_field: str = "competition_date"
        notes: Optional[str] = None
        needs_verification: bool = False

    sex_categories: list[str] = field(default_factory=list)
    age_min_years: Optional[int] = None
    age_max_years: Optional[int] = None
    age_rule_text: Optional[str] = None
    age_rule_mode: Optional[str] = None
    age_rule_semantics: Optional["DivisionEligibilityRules.AgeRuleSemantics"] = None
    novice_restriction_text: Optional[str] = None
    first_timer_only: bool = False
    disability_required: bool = False
    height_classes: list[DivisionClassSpec] = field(default_factory=list)
    weight_classes: list[DivisionClassSpec] = field(default_factory=list)
    height_to_weight_caps: list[HeightToWeightCapSpec] = field(default_factory=list)


@dataclass
class DivisionSourceRef:
    source_urls: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)


@dataclass
class DivisionRoundSpec:
    round_id: str
    round_name: str
    judged_on: Optional[str] = None
    routine_time_limit_seconds: Optional[int] = None


@dataclass
class DivisionRuleImageSpec:
    image_path: str
    caption: Optional[str] = None
    source_file: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class DivisionSpec:
    division_id: str
    division_name: str
    classes: list[DivisionClassSpec] = field(default_factory=list)
    eligibility: DivisionEligibilityRules = field(default_factory=DivisionEligibilityRules)
    rounds: list[DivisionRoundSpec] = field(default_factory=list)
    mandatory_poses: Optional[list[str]] = None
    rule_images: list[DivisionRuleImageSpec] = field(default_factory=list)
    attire: Optional[str] = None
    notes_keywords: Optional[str] = None
    needs_verification: bool = False
    sources: DivisionSourceRef = field(default_factory=DivisionSourceRef)


@dataclass
class FederationSourceRef:
    source_urls: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)


@dataclass
class FederationSpec:
    federation_id: str
    federation_name: str
    season_year: Optional[int]
    sources: FederationSourceRef
    divisions: list[DivisionSpec]


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: list[str]
    confidence: ConfidenceLevel
    needs_user_confirmation: bool


@dataclass
class FederationSpecsLibrary:
    federations: dict[str, FederationSpec]

    def list_federations(self) -> list[FederationSpec]:
        return sorted(self.federations.values(), key=lambda fed: fed.federation_name.lower())

    def get_federation(self, federation_id: str) -> Optional[FederationSpec]:
        return self.federations.get(str(federation_id or "").lower())

    def get_division(self, ref: SelectedDivisionRef) -> Optional[DivisionSpec]:
        fed = self.get_federation(ref.federation_id)
        if fed is None:
            return None
        for div in fed.divisions:
            if div.division_id == ref.division_id:
                return div
        return None

    def get_division_class(self, ref: SelectedDivisionRef) -> Optional[DivisionClassSpec]:
        div = self.get_division(ref)
        if div is None:
            return None
        if not ref.class_id:
            return None
        for cls in div.classes:
            if cls.class_id == ref.class_id:
                return cls
        for cls in div.eligibility.height_classes + div.eligibility.weight_classes:
            if cls.class_id == ref.class_id:
                return cls
        return None

    def iter_division_refs(self) -> list[SelectedDivisionRef]:
        refs: list[SelectedDivisionRef] = []
        for fed in self.list_federations():
            for div in fed.divisions:
                class_specs = list(div.classes)
                if not class_specs:
                    class_specs.extend(div.eligibility.height_classes)
                    class_specs.extend(div.eligibility.weight_classes)
                if class_specs:
                    seen: set[str] = set()
                    for cls in class_specs:
                        if cls.class_id in seen:
                            continue
                        seen.add(cls.class_id)
                        refs.append(
                            SelectedDivisionRef(
                                federation_id=fed.federation_id,
                                division_id=div.division_id,
                                class_id=cls.class_id,
                            )
                        )
                else:
                    refs.append(SelectedDivisionRef(federation_id=fed.federation_id, division_id=div.division_id))
        return refs


_DEFAULT_SPECS_DIR = Path(__file__).resolve().parents[2] / "federations"


def default_specs_dir() -> Path:
    return _DEFAULT_SPECS_DIR


def load_federation_specs(specs_dir: Optional[Path] = None) -> FederationSpecsLibrary:
    root = Path(specs_dir or _DEFAULT_SPECS_DIR)
    federations: dict[str, FederationSpec] = {}
    if not root.exists():
        return FederationSpecsLibrary(federations={})
    for path in sorted(root.glob("*.yaml")):
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                continue
            spec = _parse_federation_spec(loaded)
            if spec.federation_id:
                federations[spec.federation_id.lower()] = spec
        except Exception:
            continue
    return FederationSpecsLibrary(federations=federations)


def parse_selected_division_ref(value: Any) -> Optional[SelectedDivisionRef]:
    if isinstance(value, SelectedDivisionRef):
        return value
    if isinstance(value, dict):
        ref = SelectedDivisionRef.from_dict(value)
        if ref.federation_id and ref.division_id:
            return ref
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(":") if p.strip()]
        if len(parts) == 2:
            return SelectedDivisionRef(parts[0], parts[1], None)
        if len(parts) >= 3:
            return SelectedDivisionRef(parts[0], parts[1], parts[2])
    return None


def _parse_federation_spec(raw: dict[str, Any]) -> FederationSpec:
    sources = raw.get("sources") if isinstance(raw.get("sources"), dict) else {}
    fed_sources = FederationSourceRef(
        source_urls=_as_str_list(sources.get("source_urls")),
        source_files=_as_str_list(sources.get("source_files")),
    )

    divisions_raw = raw.get("divisions") if isinstance(raw.get("divisions"), list) else []
    divisions: list[DivisionSpec] = [_parse_division_spec(item) for item in divisions_raw if isinstance(item, dict)]

    return FederationSpec(
        federation_id=str(raw.get("federation_id", "")).strip().lower(),
        federation_name=str(raw.get("federation_name", "")).strip(),
        season_year=_as_int(raw.get("season_year")),
        sources=fed_sources,
        divisions=divisions,
    )


def _parse_division_spec(raw: dict[str, Any]) -> DivisionSpec:
    classes_raw = raw.get("classes") if isinstance(raw.get("classes"), list) else []
    classes = [_parse_division_class(item) for item in classes_raw if isinstance(item, dict)]

    elig_raw = raw.get("eligibility") if isinstance(raw.get("eligibility"), dict) else {}
    age_min_years = _as_int(elig_raw.get("age_min_years"))
    age_max_years = _as_int(elig_raw.get("age_max_years"))
    age_rule_text = _as_optional_str(elig_raw.get("age_rule_text"))
    age_rule_mode = _as_optional_str(elig_raw.get("age_rule_mode"))
    age_rule_semantics = _parse_age_rule_semantics(
        elig_raw.get("age_rule_semantics"),
        age_min_years=age_min_years,
        age_max_years=age_max_years,
        age_rule_mode=age_rule_mode,
        age_rule_text=age_rule_text,
    )
    eligibility = DivisionEligibilityRules(
        sex_categories=_as_str_list(elig_raw.get("sex_categories")),
        age_min_years=age_min_years,
        age_max_years=age_max_years,
        age_rule_text=age_rule_text,
        age_rule_mode=age_rule_mode,
        age_rule_semantics=age_rule_semantics,
        novice_restriction_text=_as_optional_str(elig_raw.get("novice_restriction_text")),
        first_timer_only=bool(elig_raw.get("first_timer_only", False)),
        disability_required=bool(elig_raw.get("disability_required", False)),
        height_classes=[
            _parse_division_class(item)
            for item in (elig_raw.get("height_classes") if isinstance(elig_raw.get("height_classes"), list) else [])
            if isinstance(item, dict)
        ],
        weight_classes=[
            _parse_division_class(item)
            for item in (elig_raw.get("weight_classes") if isinstance(elig_raw.get("weight_classes"), list) else [])
            if isinstance(item, dict)
        ],
        height_to_weight_caps=[
            _parse_height_to_weight_cap(item)
            for item in (
                elig_raw.get("height_to_weight_caps") if isinstance(elig_raw.get("height_to_weight_caps"), list) else []
            )
            if isinstance(item, dict)
        ],
    )

    rounds_raw = raw.get("rounds") if isinstance(raw.get("rounds"), list) else []
    rounds: list[DivisionRoundSpec] = []
    for item in rounds_raw:
        if not isinstance(item, dict):
            continue
        rounds.append(
            DivisionRoundSpec(
                round_id=str(item.get("round_id", "")).strip() or f"round_{len(rounds) + 1}",
                round_name=str(item.get("round_name", "")).strip(),
                judged_on=_as_optional_str(item.get("judged_on")),
                routine_time_limit_seconds=_as_int(item.get("routine_time_limit_seconds")),
            )
        )

    src_raw = raw.get("sources") if isinstance(raw.get("sources"), dict) else {}
    division_sources = DivisionSourceRef(
        source_urls=_as_str_list(src_raw.get("source_urls")),
        source_files=_as_str_list(src_raw.get("source_files")),
    )

    mandatory_poses: Optional[list[str]] = None
    if raw.get("mandatory_poses") is None:
        mandatory_poses = None
    elif isinstance(raw.get("mandatory_poses"), list):
        mandatory_poses = [str(p).strip() for p in raw.get("mandatory_poses") if str(p).strip()]
    else:
        mandatory_poses = []

    division_needs_verification = bool(raw.get("needs_verification", False))
    if age_rule_semantics is not None and age_rule_semantics.needs_verification:
        division_needs_verification = True

    return DivisionSpec(
        division_id=str(raw.get("division_id", "")).strip(),
        division_name=str(raw.get("division_name", "")).strip(),
        classes=classes,
        eligibility=eligibility,
        rounds=rounds,
        mandatory_poses=mandatory_poses,
        rule_images=[
            _parse_division_rule_image(item)
            for item in (raw.get("rule_images") if isinstance(raw.get("rule_images"), list) else [])
            if isinstance(item, dict)
        ],
        attire=_as_optional_str(raw.get("attire")),
        notes_keywords=_as_optional_str(raw.get("notes_keywords")),
        needs_verification=division_needs_verification,
        sources=division_sources,
    )


def _parse_division_class(raw: dict[str, Any]) -> DivisionClassSpec:
    return DivisionClassSpec(
        class_id=str(raw.get("class_id", "")).strip(),
        class_name=str(raw.get("class_name", "")).strip(),
        min_height_cm=_as_float(raw.get("min_height_cm")),
        max_height_cm=_as_float(raw.get("max_height_cm")),
        min_weight_kg=_as_float(raw.get("min_weight_kg")),
        max_weight_kg=_as_float(raw.get("max_weight_kg")),
        notes=_as_optional_str(raw.get("notes")),
        needs_verification=bool(raw.get("needs_verification", False)),
    )


def _parse_height_to_weight_cap(raw: dict[str, Any]) -> HeightToWeightCapSpec:
    return HeightToWeightCapSpec(
        cap_id=str(raw.get("cap_id", "")).strip(),
        class_id=_as_optional_str(raw.get("class_id")),
        min_height_cm=_as_float(raw.get("min_height_cm")),
        max_height_cm=_as_float(raw.get("max_height_cm")),
        formula_type=_as_optional_str(raw.get("formula_type")),
        formula_plus_kg=_as_float(raw.get("formula_plus_kg")),
        max_weight_kg=_as_float(raw.get("max_weight_kg")),
        text=_as_optional_str(raw.get("text")),
        needs_verification=bool(raw.get("needs_verification", False)),
    )


def _parse_division_rule_image(raw: dict[str, Any]) -> DivisionRuleImageSpec:
    return DivisionRuleImageSpec(
        image_path=str(raw.get("image_path", "")).strip(),
        caption=_as_optional_str(raw.get("caption")),
        source_file=_as_optional_str(raw.get("source_file")),
        page_number=_as_int(raw.get("page_number")),
    )


def _parse_age_rule_semantics(
    raw: Any,
    *,
    age_min_years: Optional[int],
    age_max_years: Optional[int],
    age_rule_mode: Optional[str],
    age_rule_text: Optional[str],
) -> Optional[DivisionEligibilityRules.AgeRuleSemantics]:
    if isinstance(raw, dict):
        semantics = DivisionEligibilityRules.AgeRuleSemantics(
            rule_type=_as_optional_str(raw.get("rule_type")),
            turn_age_years=_as_int(raw.get("turn_age_years")),
            reference_date_field=_as_optional_str(raw.get("reference_date_field")) or "competition_date",
            notes=_as_optional_str(raw.get("notes")),
            needs_verification=bool(raw.get("needs_verification", False)),
        )
        if (
            semantics.rule_type is None
            and semantics.turn_age_years is None
            and age_rule_text
            and not semantics.needs_verification
        ):
            semantics.needs_verification = True
        return semantics

    if age_rule_mode == "year_end_max_age" and age_max_years is not None:
        return DivisionEligibilityRules.AgeRuleSemantics(
            rule_type="junior_until_dec31_year_turn",
            turn_age_years=int(age_max_years),
            reference_date_field="competition_date",
            notes="year_turns_N = dob.year + N; eligible until 31 Dec of that year.",
            needs_verification=False,
        )
    if age_rule_mode == "year_start_min_age" and age_min_years is not None:
        return DivisionEligibilityRules.AgeRuleSemantics(
            rule_type="masters_from_jan1_year_turn",
            turn_age_years=int(age_min_years),
            reference_date_field="competition_date",
            notes="year_turns_N = dob.year + N; eligible from 1 Jan of that year.",
            needs_verification=False,
        )
    if age_rule_text:
        return DivisionEligibilityRules.AgeRuleSemantics(
            rule_type=None,
            turn_age_years=None,
            reference_date_field="competition_date",
            notes="Age rule text present but machine semantics are unclear.",
            needs_verification=True,
        )
    return None


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None
