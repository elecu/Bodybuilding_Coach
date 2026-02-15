from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import calendar
import re
from typing import Any, Optional

from .specs import (
    DivisionClassSpec,
    DivisionSpec,
    EligibilityResult,
    FederationSpecsLibrary,
    SelectedDivisionRef,
)

_NON_AGE_NUMERIC_UNITS = r"(?:kg|kgs|kilograms?|lb|lbs|pounds?|cm|mm|m)\b"
_AGE_TEXT_SUFFIX = r"(?:\s*(?:years?|yrs?|y\/?o|s))?"
_WOMEN_LABEL_REGEX = r"\b(women|women'?s|female|females|ladies|woman)\b"
_MEN_LABEL_REGEX = r"\b(men|men'?s|male|males|man)\b"


@dataclass(frozen=True)
class DivisionEligibilityEvaluation:
    ref: SelectedDivisionRef
    division_name: str
    class_name: Optional[str]
    result: EligibilityResult


def evaluate_division_eligibility(
    profile: Any,
    division: DivisionSpec,
    class_spec: Optional[DivisionClassSpec] = None,
    event_date: Optional[date] = None,
) -> EligibilityResult:
    reasons: list[str] = []
    needs_confirmation = False
    missing_required = False
    eligible = True

    event_date = event_date or _resolve_event_date(profile)
    strict_weight_checks = _weight_checks_are_strict(profile)

    sex_categories = [s.strip().lower() for s in division.eligibility.sex_categories if s]
    profile_sex = _as_text(_profile_get(profile, "sex_category"))
    if sex_categories:
        if not profile_sex:
            needs_confirmation = True
            missing_required = True
            reasons.append(
                "Sex category is required for this division. Allowed categories: "
                + ", ".join(division.eligibility.sex_categories)
            )
        else:
            if profile_sex.lower() not in sex_categories:
                eligible = False
                reasons.append(
                    f"Sex category '{profile_sex}' is not eligible for this division (allowed: "
                    + ", ".join(division.eligibility.sex_categories)
                    + ")."
                )

    class_sex_eval = _evaluate_class_sex_hints(profile, class_spec)
    eligible = eligible and class_sex_eval["eligible"]
    if class_sex_eval["reason"]:
        reasons.append(class_sex_eval["reason"])
    needs_confirmation = needs_confirmation or class_sex_eval["needs_confirmation"]
    missing_required = missing_required or class_sex_eval["missing_required"]

    class_height_eval = _evaluate_class_height_hints(profile, class_spec)
    eligible = eligible and class_height_eval["eligible"]
    if class_height_eval["reason"]:
        reasons.append(class_height_eval["reason"])
    needs_confirmation = needs_confirmation or class_height_eval["needs_confirmation"]
    missing_required = missing_required or class_height_eval["missing_required"]

    disability_eval = _evaluate_disability_requirement(profile, division, class_spec)
    eligible = eligible and disability_eval["eligible"]
    if disability_eval["reason"]:
        reasons.append(disability_eval["reason"])
    needs_confirmation = needs_confirmation or disability_eval["needs_confirmation"]
    missing_required = missing_required or disability_eval["missing_required"]

    age_result = _evaluate_age(profile, division, event_date)
    eligible = eligible and age_result["eligible"]
    if age_result["reason"]:
        reasons.append(age_result["reason"])
    needs_confirmation = needs_confirmation or age_result["needs_confirmation"]
    missing_required = missing_required or age_result["missing_required"]

    class_age_eval = _evaluate_class_age_hints(profile, class_spec, event_date)
    eligible = eligible and class_age_eval["eligible"]
    if class_age_eval["reason"]:
        reasons.append(class_age_eval["reason"])
    needs_confirmation = needs_confirmation or class_age_eval["needs_confirmation"]
    missing_required = missing_required or class_age_eval["missing_required"]

    division_age_eval = _evaluate_division_age_hints(profile, division, event_date)
    eligible = eligible and division_age_eval["eligible"]
    if division_age_eval["reason"]:
        reasons.append(division_age_eval["reason"])
    needs_confirmation = needs_confirmation or division_age_eval["needs_confirmation"]
    missing_required = missing_required or division_age_eval["missing_required"]

    class_eval = _evaluate_class_ranges(profile, division, class_spec, strict_weight_checks=strict_weight_checks)
    eligible = eligible and class_eval["eligible"]
    reasons.extend(class_eval["reasons"])
    needs_confirmation = needs_confirmation or class_eval["needs_confirmation"]
    missing_required = missing_required or class_eval["missing_required"]

    cap_eval = _evaluate_height_weight_caps(
        profile,
        division,
        class_spec,
        strict_weight_checks=strict_weight_checks,
    )
    eligible = eligible and cap_eval["eligible"]
    reasons.extend(cap_eval["reasons"])
    needs_confirmation = needs_confirmation or cap_eval["needs_confirmation"]
    missing_required = missing_required or cap_eval["missing_required"]

    novice_text = _as_text(division.eligibility.novice_restriction_text)
    if novice_text:
        history = _competition_history(profile)
        if history is None:
            needs_confirmation = True
            reasons.append(
                "Novice/first-timer restriction applies and competition history is not present in profile. "
                + novice_text
            )
        elif division.eligibility.first_timer_only and len(history) > 0:
            eligible = False
            reasons.append("Division is restricted to first-timer/novice athletes. Profile has competition history.")

    if division.needs_verification or (class_spec is not None and bool(getattr(class_spec, "needs_verification", False))):
        needs_confirmation = True
        reasons.append("Selected spec has fields marked needs_verification=true. Manual confirmation recommended.")

    if not reasons:
        reasons.append("No blocking eligibility rules were found in the selected spec.")

    if missing_required:
        confidence = "low"
    elif needs_confirmation:
        confidence = "medium"
    else:
        confidence = "high"

    return EligibilityResult(
        eligible=bool(eligible),
        reasons=reasons,
        confidence=confidence,
        needs_user_confirmation=bool(needs_confirmation),
    )


def evaluate_age_gate(
    profile: Any,
    division: DivisionSpec,
    class_spec: Optional[DivisionClassSpec] = None,
    event_date: Optional[date] = None,
) -> dict[str, Any]:
    event_date = event_date or _resolve_event_date(profile)
    age_result = _evaluate_age(profile, division, event_date)
    class_age_result = _evaluate_class_age_hints(profile, class_spec, event_date)
    return {
        "eligible": bool(age_result["eligible"] and class_age_result["eligible"]),
        "reason": " ".join(
            item
            for item in [str(age_result.get("reason", "")).strip(), str(class_age_result.get("reason", "")).strip()]
            if item
        ).strip(),
        "needs_confirmation": bool(age_result["needs_confirmation"] or class_age_result["needs_confirmation"]),
        "missing_required": bool(age_result["missing_required"] or class_age_result["missing_required"]),
    }


def projected_event_dates(profile: Any, years_ahead: int = 5) -> list[date]:
    span = max(0, int(years_ahead))
    base = _competition_date_from_profile(profile) or date.today()
    out: list[date] = []
    for offset in range(span + 1):
        year = base.year + offset
        month = base.month
        day = min(base.day, calendar.monthrange(year, month)[1])
        out.append(date(year, month, day))
    return out


def matches_age_window(
    profile: Any,
    division: DivisionSpec,
    class_spec: Optional[DivisionClassSpec] = None,
    years_ahead: int = 5,
) -> bool:
    has_unknown = False
    for event_dt in projected_event_dates(profile, years_ahead=years_ahead):
        gate = evaluate_age_gate(profile=profile, division=division, class_spec=class_spec, event_date=event_dt)
        if gate["eligible"]:
            return True
        if gate["needs_confirmation"]:
            has_unknown = True
    return has_unknown


def _evaluate_class_age_hints(
    profile: Any,
    class_spec: Optional[DivisionClassSpec],
    event_date: date,
) -> dict[str, Any]:
    if class_spec is None:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }

    class_name_raw = str(class_spec.class_name or "").strip()
    notes_raw = str(class_spec.notes or "").strip()
    age = _resolved_age_from_dob(profile, event_date)
    if age is None:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }
    return _evaluate_label_age_hints(
        label_raw=class_name_raw,
        notes_raw=notes_raw,
        age=age,
        scope_name="Class",
    )


def _evaluate_class_height_hints(
    profile: Any,
    class_spec: Optional[DivisionClassSpec],
) -> dict[str, Any]:
    if class_spec is None:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }

    class_name_raw = str(class_spec.class_name or "").strip()
    notes_raw = str(class_spec.notes or "").strip()
    source_text = f"{class_name_raw} {notes_raw}".lower()
    height_cm = _as_float(_profile_get(profile, "height_cm"))

    def _float(text: str) -> Optional[float]:
        try:
            return float(text)
        except Exception:
            return None

    def _missing_height() -> dict[str, Any]:
        return {
            "eligible": True,
            "reason": f"Height is required to validate class '{class_name_raw}'.",
            "needs_confirmation": True,
            "missing_required": True,
        }

    # 1) Range with explicit unit: 170-175 cm / 170 to 175 cm
    range_match = re.search(r"\b(\d{2,3}(?:\.\d+)?)\s*(?:-|–|to)\s*(\d{2,3}(?:\.\d+)?)\s*cm\b", source_text)
    if range_match is not None:
        low = _float(range_match.group(1))
        high = _float(range_match.group(2))
        if low is not None and high is not None and low <= high:
            if height_cm is None:
                return _missing_height()
            if height_cm < low or height_cm > high:
                return {
                    "eligible": False,
                    "reason": (
                        f"Class '{class_name_raw}' requires height between {low:.1f} and {high:.1f} cm. "
                        f"Athlete height is {height_cm:.1f} cm."
                    ),
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            return {
                "eligible": True,
                "reason": "",
                "needs_confirmation": False,
                "missing_required": False,
            }

    # 2) Over X cm up to (and including) Y cm
    over_up_to_match = re.search(
        r"\bover\s*(\d{2,3}(?:\.\d+)?)\s*cm\b.*?\bup to(?: and including)?\s*(\d{2,3}(?:\.\d+)?)\s*cm\b",
        source_text,
    )
    if over_up_to_match is not None:
        minimum_exclusive = _float(over_up_to_match.group(1))
        maximum_inclusive = _float(over_up_to_match.group(2))
        if (
            minimum_exclusive is not None
            and maximum_inclusive is not None
            and minimum_exclusive < maximum_inclusive
        ):
            if height_cm is None:
                return _missing_height()
            if height_cm <= minimum_exclusive or height_cm > maximum_inclusive:
                return {
                    "eligible": False,
                    "reason": (
                        f"Class '{class_name_raw}' requires height over {minimum_exclusive:.1f} cm and up to "
                        f"{maximum_inclusive:.1f} cm. Athlete height is {height_cm:.1f} cm."
                    ),
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            return {
                "eligible": True,
                "reason": "",
                "needs_confirmation": False,
                "missing_required": False,
            }

    # 3) From X cm up to (and including) Y cm
    from_up_to_match = re.search(
        r"\bfrom\s*(\d{2,3}(?:\.\d+)?)\s*cm\b.*?\bup to(?: and including)?\s*(\d{2,3}(?:\.\d+)?)\s*cm\b",
        source_text,
    )
    if from_up_to_match is not None:
        minimum_inclusive = _float(from_up_to_match.group(1))
        maximum_inclusive = _float(from_up_to_match.group(2))
        if (
            minimum_inclusive is not None
            and maximum_inclusive is not None
            and minimum_inclusive <= maximum_inclusive
        ):
            if height_cm is None:
                return _missing_height()
            if height_cm < minimum_inclusive or height_cm > maximum_inclusive:
                return {
                    "eligible": False,
                    "reason": (
                        f"Class '{class_name_raw}' requires height between {minimum_inclusive:.1f} and "
                        f"{maximum_inclusive:.1f} cm. Athlete height is {height_cm:.1f} cm."
                    ),
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            return {
                "eligible": True,
                "reason": "",
                "needs_confirmation": False,
                "missing_required": False,
            }

    # 4) Up to / up to and including N cm
    up_to_match = re.search(r"\bup to(?: and including)?\s*(\d{2,3}(?:\.\d+)?)\s*cm\b", source_text)
    if up_to_match is not None:
        maximum = _float(up_to_match.group(1))
        if maximum is not None:
            if height_cm is None:
                return _missing_height()
            if height_cm > maximum:
                return {
                    "eligible": False,
                    "reason": (
                        f"Class '{class_name_raw}' allows up to {maximum:.1f} cm. "
                        f"Athlete height is {height_cm:.1f} cm."
                    ),
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            return {
                "eligible": True,
                "reason": "",
                "needs_confirmation": False,
                "missing_required": False,
            }

    # 5) Over N cm (strict > N)
    over_match = re.search(r"\bover\s*(\d{2,3}(?:\.\d+)?)\s*cm\b", source_text)
    if over_match is not None:
        minimum_exclusive = _float(over_match.group(1))
        if minimum_exclusive is not None:
            if height_cm is None:
                return _missing_height()
            if height_cm <= minimum_exclusive:
                return {
                    "eligible": False,
                    "reason": (
                        f"Class '{class_name_raw}' requires height over {minimum_exclusive:.1f} cm. "
                        f"Athlete height is {height_cm:.1f} cm."
                    ),
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            return {
                "eligible": True,
                "reason": "",
                "needs_confirmation": False,
                "missing_required": False,
            }

    # 6) From N cm / N cm and over / N+ cm
    min_match = re.search(
        r"\b(?:from\s*|)(\d{2,3}(?:\.\d+)?)\s*cm\s*(?:and\s*)?(?:over|plus|\+)\b|\bfrom\s*(\d{2,3}(?:\.\d+)?)\s*cm\b",
        source_text,
    )
    if min_match is not None:
        raw = min_match.group(1) or min_match.group(2)
        minimum_inclusive = _float(str(raw))
        if minimum_inclusive is not None:
            if height_cm is None:
                return _missing_height()
            if height_cm < minimum_inclusive:
                return {
                    "eligible": False,
                    "reason": (
                        f"Class '{class_name_raw}' requires minimum height {minimum_inclusive:.1f} cm. "
                        f"Athlete height is {height_cm:.1f} cm."
                    ),
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            return {
                "eligible": True,
                "reason": "",
                "needs_confirmation": False,
                "missing_required": False,
            }

    # 7) Under N cm (strict < N)
    under_match = re.search(r"\bunder\s*(\d{2,3}(?:\.\d+)?)\s*cm\b", source_text)
    if under_match is not None:
        maximum_exclusive = _float(under_match.group(1))
        if maximum_exclusive is not None:
            if height_cm is None:
                return _missing_height()
            if height_cm >= maximum_exclusive:
                return {
                    "eligible": False,
                    "reason": (
                        f"Class '{class_name_raw}' requires height under {maximum_exclusive:.1f} cm. "
                        f"Athlete height is {height_cm:.1f} cm."
                    ),
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            return {
                "eligible": True,
                "reason": "",
                "needs_confirmation": False,
                "missing_required": False,
            }

    return {
        "eligible": True,
        "reason": "",
        "needs_confirmation": False,
        "missing_required": False,
    }


def _evaluate_class_sex_hints(
    profile: Any,
    class_spec: Optional[DivisionClassSpec],
) -> dict[str, Any]:
    if class_spec is None:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }
    class_name_raw = str(class_spec.class_name or "").strip()
    notes_raw = str(class_spec.notes or "").strip()
    class_text = class_name_raw.lower()
    notes_text = notes_raw.lower()
    women_name_hint = bool(re.search(_WOMEN_LABEL_REGEX, class_text))
    men_name_hint = bool(re.search(_MEN_LABEL_REGEX, class_text))
    women_notes_hint = bool(re.search(_WOMEN_LABEL_REGEX, notes_text))
    men_notes_hint = bool(re.search(_MEN_LABEL_REGEX, notes_text))
    if women_name_hint != men_name_hint:
        women_hint = women_name_hint
        men_hint = men_name_hint
    else:
        women_hint = women_name_hint or women_notes_hint
        men_hint = men_name_hint or men_notes_hint
    if not women_hint and not men_hint:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }
    if women_hint and men_hint:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }
    profile_sex = _as_text(_profile_get(profile, "sex_category"))
    if not profile_sex:
        return {
            "eligible": True,
            "reason": "Sex category is required to validate this sex-specific class.",
            "needs_confirmation": True,
            "missing_required": True,
        }
    profile_sex_norm = profile_sex.lower()
    if women_hint and profile_sex_norm != "female":
        return {
            "eligible": False,
            "reason": f"Class '{class_name_raw}' is women-specific. Athlete sex category is '{profile_sex}'.",
            "needs_confirmation": False,
            "missing_required": False,
        }
    if men_hint and profile_sex_norm != "male":
        return {
            "eligible": False,
            "reason": f"Class '{class_name_raw}' is men-specific. Athlete sex category is '{profile_sex}'.",
            "needs_confirmation": False,
            "missing_required": False,
        }
    return {
        "eligible": True,
        "reason": "",
        "needs_confirmation": False,
        "missing_required": False,
    }


def _evaluate_disability_requirement(
    profile: Any,
    division: DivisionSpec,
    class_spec: Optional[DivisionClassSpec],
) -> dict[str, Any]:
    required = bool(getattr(division.eligibility, "disability_required", False))
    if not required:
        div_text = str(getattr(division, "division_name", "") or "").lower()
        cls_text = ""
        if class_spec is not None:
            cls_text = f"{str(class_spec.class_name or '').lower()} {str(class_spec.notes or '').lower()}"
        required = any(token in div_text or token in cls_text for token in ("disability", "disabled"))

    if not required:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }

    has_disability = _as_optional_bool(_profile_get(profile, "has_disability"))
    if has_disability is True:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }
    if has_disability is False:
        return {
            "eligible": False,
            "reason": "This category is open to disabled athletes only.",
            "needs_confirmation": False,
            "missing_required": False,
        }
    return {
        "eligible": True,
        "reason": "Disability status is required in the athlete profile to validate this category.",
        "needs_confirmation": True,
        "missing_required": True,
    }


def _evaluate_division_age_hints(
    profile: Any,
    division: DivisionSpec,
    event_date: date,
) -> dict[str, Any]:
    if getattr(division, "classes", None):
        # Division title may list multiple class families (e.g., teens/masters) and is
        # not a reliable single age gate when explicit classes are present.
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }

    age = _resolved_age_from_dob(profile, event_date)
    if age is None:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }
    notes_text = _as_text(getattr(getattr(division, "eligibility", None), "age_rule_text", None)) or ""
    return _evaluate_label_age_hints(
        label_raw=str(getattr(division, "division_name", "") or ""),
        notes_raw=notes_text,
        age=age,
        scope_name="Division",
    )


def _evaluate_label_age_hints(
    label_raw: str,
    notes_raw: str,
    age: int,
    scope_name: str,
) -> dict[str, Any]:
    label_clean = str(label_raw or "").strip()
    notes_clean = str(notes_raw or "").strip()
    source_text = f"{label_clean} {notes_clean}".lower()

    # Explicit numeric patterns in labels/notes are applied directly.
    range_match = re.search(
        rf"\b(\d{{1,2}})\s*(?:-|–|to)\s*(\d{{1,2}})(?!\s*{_NON_AGE_NUMERIC_UNITS})",
        source_text,
    )
    if range_match is not None:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        if low <= high and (age < low or age > high):
            return {
                "eligible": False,
                "reason": f"{scope_name} '{label_clean}' has an age range {low}-{high}. Athlete age is {age}.",
                "needs_confirmation": False,
                "missing_required": False,
            }

    over_match = re.search(
        rf"\bover\s*(\d{{1,2}}){_AGE_TEXT_SUFFIX}(?!\s*{_NON_AGE_NUMERIC_UNITS})",
        source_text,
    )
    if over_match is not None:
        minimum = int(over_match.group(1))
        if age < minimum:
            return {
                "eligible": False,
                "reason": f"{scope_name} '{label_clean}' has minimum age {minimum}. Athlete age is {age}.",
                "needs_confirmation": False,
                "missing_required": False,
            }

    plus_match = re.search(r"\b(\d{1,2})\s*\+", source_text)
    if plus_match is not None:
        minimum = int(plus_match.group(1))
        if age < minimum:
            return {
                "eligible": False,
                "reason": f"{scope_name} '{label_clean}' has minimum age {minimum}. Athlete age is {age}.",
                "needs_confirmation": False,
                "missing_required": False,
            }

    from_match = re.search(
        rf"\bfrom\s*(\d{{1,2}}){_AGE_TEXT_SUFFIX}(?!\s*{_NON_AGE_NUMERIC_UNITS})",
        source_text,
    )
    if from_match is not None:
        minimum = int(from_match.group(1))
        if age < minimum:
            return {
                "eligible": False,
                "reason": f"{scope_name} '{label_clean}' has minimum age {minimum}. Athlete age is {age}.",
                "needs_confirmation": False,
                "missing_required": False,
            }

    under_match = re.search(
        rf"\bunder\s*(\d{{1,2}}){_AGE_TEXT_SUFFIX}(?!\s*{_NON_AGE_NUMERIC_UNITS})",
        source_text,
    )
    if under_match is not None:
        maximum = int(under_match.group(1))
        if age >= maximum:
            return {
                "eligible": False,
                "reason": f"{scope_name} '{label_clean}' is marked under {maximum}. Athlete age is {age}.",
                "needs_confirmation": False,
                "missing_required": False,
            }

    # Plain-language age bucket to prevent clearly invalid assignments.
    if "teen" in source_text and age > 19:
        return {
            "eligible": False,
            "reason": f"{scope_name} '{label_clean}' is a teen category. Athlete age is {age}.",
            "needs_confirmation": False,
            "missing_required": False,
        }

    return {
        "eligible": True,
        "reason": "",
        "needs_confirmation": False,
        "missing_required": False,
    }


def evaluate_ref_eligibility(
    library: FederationSpecsLibrary,
    profile: Any,
    ref: SelectedDivisionRef,
    event_date: Optional[date] = None,
) -> Optional[DivisionEligibilityEvaluation]:
    fed = library.get_federation(ref.federation_id)
    if fed is None:
        return None
    div = library.get_division(ref)
    if div is None:
        return None
    cls = library.get_division_class(ref)
    result = evaluate_division_eligibility(profile=profile, division=div, class_spec=cls, event_date=event_date)
    return DivisionEligibilityEvaluation(
        ref=ref,
        division_name=div.division_name,
        class_name=cls.class_name if cls is not None else None,
        result=result,
    )


def evaluate_all_divisions(
    library: FederationSpecsLibrary,
    profile: Any,
    event_date: Optional[date] = None,
) -> list[DivisionEligibilityEvaluation]:
    evaluations: list[DivisionEligibilityEvaluation] = []
    for ref in library.iter_division_refs():
        eval_item = evaluate_ref_eligibility(library, profile, ref, event_date=event_date)
        if eval_item is not None:
            evaluations.append(eval_item)
    return evaluations


def _evaluate_age(profile: Any, division: DivisionSpec, event_date: date) -> dict[str, Any]:
    age_min = division.eligibility.age_min_years
    age_max = division.eligibility.age_max_years
    age_rule_text = _as_text(division.eligibility.age_rule_text)
    age_rule_semantics = division.eligibility.age_rule_semantics

    if age_min is None and age_max is None and age_rule_semantics is None and not age_rule_text:
        return {
            "eligible": True,
            "reason": "",
            "needs_confirmation": False,
            "missing_required": False,
        }

    semantic_eval = _evaluate_age_rule_semantics(profile, event_date, age_rule_semantics, age_rule_text)
    if semantic_eval is not None:
        return semantic_eval
    if age_rule_text:
        text_eval = _evaluate_age_rule_text_hints(profile, event_date, age_rule_text)
        if text_eval is not None:
            return text_eval
        return {
            "eligible": True,
            "reason": "Age rule text is present but semantics are not machine-verifiable. Manual confirmation required.",
            "needs_confirmation": True,
            "missing_required": True,
        }

    age = _resolved_age_from_dob(profile, event_date)
    if age is None:
        return {
            "eligible": True,
            "reason": "Age could not be verified from profile for this division.",
            "needs_confirmation": True,
            "missing_required": True,
        }

    if age_min is not None and age < age_min:
        return {
            "eligible": False,
            "reason": f"Minimum age is {age_min}. Athlete age is {age}.",
            "needs_confirmation": False,
            "missing_required": False,
        }

    if age_max is not None and age > age_max:
        return {
            "eligible": False,
            "reason": f"Maximum age is {age_max}. Athlete age is {age}.",
            "needs_confirmation": False,
            "missing_required": False,
        }

    return {
        "eligible": True,
        "reason": age_rule_text or "Age requirement satisfied.",
        "needs_confirmation": False,
        "missing_required": False,
    }


def _evaluate_age_rule_text_hints(
    profile: Any,
    event_date: date,
    age_rule_text: str,
) -> Optional[dict[str, Any]]:
    text = str(age_rule_text or "").strip()
    if not text:
        return None
    age = _resolved_age_from_dob(profile, event_date)
    if age is None:
        return {
            "eligible": True,
            "reason": "Date of birth is required to verify age rule text.",
            "needs_confirmation": True,
            "missing_required": True,
        }
    source = text.lower()

    range_match = re.search(
        rf"\b(\d{{1,2}})\s*(?:-|–|to)\s*(\d{{1,2}})(?!\s*{_NON_AGE_NUMERIC_UNITS})",
        source,
    )
    if range_match is not None:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        if low <= high:
            if age < low:
                return {
                    "eligible": False,
                    "reason": f"Minimum age from rule text is {low}. Athlete age is {age}.",
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            if age > high:
                return {
                    "eligible": False,
                    "reason": f"Maximum age from rule text is {high}. Athlete age is {age}.",
                    "needs_confirmation": False,
                    "missing_required": False,
                }
            return {
                "eligible": True,
                "reason": age_rule_text,
                "needs_confirmation": False,
                "missing_required": False,
            }

    min_patterns: list[str] = [
        rf"\b(?:age|aged|ages)\s*(\d{{1,2}}){_AGE_TEXT_SUFFIX}\s*(?:and\s*)?(?:over|plus|\+)\b",
        rf"\b(\d{{1,2}}){_AGE_TEXT_SUFFIX}\s*(?:and\s*)?(?:over|plus|\+)\b",
        rf"\bover\s*(\d{{1,2}}){_AGE_TEXT_SUFFIX}(?!\s*{_NON_AGE_NUMERIC_UNITS})",
        rf"\bfrom\s*(\d{{1,2}}){_AGE_TEXT_SUFFIX}(?!\s*{_NON_AGE_NUMERIC_UNITS})",
        r"\b(\d{1,2})\s*\+",
    ]
    for pattern in min_patterns:
        match = re.search(pattern, source)
        if match is None:
            continue
        min_age = int(match.group(1))
        if age < min_age:
            return {
                "eligible": False,
                "reason": f"Minimum age from rule text is {min_age}. Athlete age is {age}.",
                "needs_confirmation": False,
                "missing_required": False,
            }
        return {
            "eligible": True,
            "reason": age_rule_text,
            "needs_confirmation": False,
            "missing_required": False,
        }

    max_patterns = [
        rf"\bunder\s*(\d{{1,2}}){_AGE_TEXT_SUFFIX}(?!\s*{_NON_AGE_NUMERIC_UNITS})",
        r"\buntil\s*(?:age\s*)?(\d{1,2})\b",
    ]
    for pattern in max_patterns:
        match = re.search(pattern, source)
        if match is None:
            continue
        max_age = int(match.group(1))
        if age >= max_age and "under" in pattern:
            return {
                "eligible": False,
                "reason": f"Maximum age from rule text is under {max_age}. Athlete age is {age}.",
                "needs_confirmation": False,
                "missing_required": False,
            }
        if age > max_age and "until" in pattern:
            return {
                "eligible": False,
                "reason": f"Maximum age from rule text is {max_age}. Athlete age is {age}.",
                "needs_confirmation": False,
                "missing_required": False,
            }
        return {
            "eligible": True,
            "reason": age_rule_text,
            "needs_confirmation": False,
            "missing_required": False,
        }

    return None


def _evaluate_age_rule_semantics(
    profile: Any,
    event_date: date,
    semantics: Any,
    age_rule_text: Optional[str],
) -> Optional[dict[str, Any]]:
    if semantics is None:
        return None

    rule_type = _as_text(getattr(semantics, "rule_type", None))
    turn_age_years = getattr(semantics, "turn_age_years", None)
    needs_verification = bool(getattr(semantics, "needs_verification", False))
    notes = _as_text(getattr(semantics, "notes", None))

    if rule_type in ("masters_from_jan1_year_turn", "junior_until_dec31_year_turn"):
        if turn_age_years is None:
            return {
                "eligible": True,
                "reason": (
                    age_rule_text
                    or notes
                    or "Age-rule semantics require a turn-age value N, but it is missing. Manual confirmation required."
                ),
                "needs_confirmation": True,
                "missing_required": True,
            }
        competition_date = _competition_date_from_profile(profile)
        if competition_date is None:
            return {
                "eligible": True,
                "reason": "Competition date is required to evaluate this year-turn age rule.",
                "needs_confirmation": True,
                "missing_required": True,
            }
        dob = _as_text(_profile_get(profile, "date_of_birth"))
        dob_date = _parse_iso_date(dob)
        if dob_date is None:
            return {
                "eligible": True,
                "reason": "Date of birth is required to evaluate this year-turn age rule.",
                "needs_confirmation": True,
                "missing_required": True,
            }
        year_turns_n = int(dob_date.year + int(turn_age_years))
        if rule_type == "masters_from_jan1_year_turn":
            boundary = date(year_turns_n, 1, 1)
            eligible = event_date >= boundary
            return {
                "eligible": eligible,
                "reason": (
                    age_rule_text
                    or f"year_turns_{int(turn_age_years)}={year_turns_n}; eligible from {boundary.isoformat()}."
                ),
                "needs_confirmation": False,
                "missing_required": False,
            }
        boundary = date(year_turns_n, 12, 31)
        eligible = event_date <= boundary
        return {
            "eligible": eligible,
            "reason": (
                age_rule_text
                or f"year_turns_{int(turn_age_years)}={year_turns_n}; eligible until {boundary.isoformat()}."
            ),
            "needs_confirmation": False,
            "missing_required": False,
        }

    if needs_verification:
        return {
            "eligible": True,
            "reason": (
                age_rule_text
                or notes
                or "Age-rule semantics are marked needs_verification and cannot be auto-applied."
            ),
            "needs_confirmation": True,
            "missing_required": True,
        }
    if age_rule_text or notes:
        return {
            "eligible": True,
            "reason": "Age rule is present but semantics are unclear. Manual confirmation required.",
            "needs_confirmation": True,
            "missing_required": True,
        }
    return None


def _evaluate_class_ranges(
    profile: Any,
    division: DivisionSpec,
    class_spec: Optional[DivisionClassSpec],
    strict_weight_checks: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    missing_required = False
    needs_confirmation = False

    classes = [class_spec] if class_spec is not None else list(division.classes)
    if not classes:
        classes.extend(division.eligibility.height_classes)
        classes.extend(division.eligibility.weight_classes)

    if not classes:
        return {
            "eligible": True,
            "reasons": reasons,
            "needs_confirmation": False,
            "missing_required": False,
        }

    h = _as_float(_profile_get(profile, "height_cm"))
    w = _as_float(_profile_get(profile, "weight_kg"))

    def _passes(cls: DivisionClassSpec) -> tuple[bool, list[str], bool]:
        cls_reasons: list[str] = []
        missing = False
        ok = True
        if cls.min_height_cm is not None or cls.max_height_cm is not None:
            if h is None:
                ok = False
                missing = True
                cls_reasons.append(f"Height is required for class '{cls.class_name}'.")
            else:
                if cls.min_height_cm is not None and h < cls.min_height_cm:
                    ok = False
                    cls_reasons.append(f"Height {h:.1f} cm is below class minimum {cls.min_height_cm:.1f} cm.")
                if cls.max_height_cm is not None and h > cls.max_height_cm:
                    ok = False
                    cls_reasons.append(f"Height {h:.1f} cm is above class maximum {cls.max_height_cm:.1f} cm.")
        if cls.min_weight_kg is not None or cls.max_weight_kg is not None:
            if w is None:
                if strict_weight_checks:
                    ok = False
                    missing = True
                    cls_reasons.append(f"Weight is required for class '{cls.class_name}'.")
                else:
                    missing = True
                    cls_reasons.append(
                        f"Weight class check for '{cls.class_name}' is advisory until competition day."
                    )
            else:
                if cls.min_weight_kg is not None and w < cls.min_weight_kg:
                    if strict_weight_checks:
                        ok = False
                        cls_reasons.append(f"Weight {w:.1f} kg is below class minimum {cls.min_weight_kg:.1f} kg.")
                    else:
                        missing = True
                        cls_reasons.append(
                            f"Current weight {w:.1f} kg is below class minimum {cls.min_weight_kg:.1f} kg "
                            "(advisory until competition day)."
                        )
                if cls.max_weight_kg is not None and w > cls.max_weight_kg:
                    if strict_weight_checks:
                        ok = False
                        cls_reasons.append(f"Weight {w:.1f} kg is above class maximum {cls.max_weight_kg:.1f} kg.")
                    else:
                        missing = True
                        cls_reasons.append(
                            f"Current weight {w:.1f} kg is above class maximum {cls.max_weight_kg:.1f} kg "
                            "(advisory until competition day)."
                        )
        return ok, cls_reasons, missing

    if class_spec is not None:
        ok, cls_reasons, missing = _passes(class_spec)
        reasons.extend(cls_reasons)
        if missing:
            missing_required = True
            needs_confirmation = True
        return {
            "eligible": ok,
            "reasons": reasons,
            "needs_confirmation": needs_confirmation,
            "missing_required": missing_required,
        }

    passing_classes: list[str] = []
    class_failure_messages: list[str] = []
    for cls in classes:
        ok, cls_reasons, missing = _passes(cls)
        if ok:
            passing_classes.append(cls.class_name)
        else:
            class_failure_messages.extend(cls_reasons)
        if missing:
            missing_required = True
            needs_confirmation = True

    if passing_classes:
        reasons.append("Eligible class options: " + ", ".join(passing_classes))
        return {
            "eligible": True,
            "reasons": reasons,
            "needs_confirmation": needs_confirmation,
            "missing_required": missing_required,
        }

    if class_failure_messages:
        reasons.extend(class_failure_messages)
    return {
        "eligible": False,
        "reasons": reasons,
        "needs_confirmation": needs_confirmation,
        "missing_required": missing_required,
    }


def _evaluate_height_weight_caps(
    profile: Any,
    division: DivisionSpec,
    class_spec: Optional[DivisionClassSpec],
    strict_weight_checks: bool,
) -> dict[str, Any]:
    caps = division.eligibility.height_to_weight_caps
    if not caps:
        return {
            "eligible": True,
            "reasons": [],
            "needs_confirmation": False,
            "missing_required": False,
        }

    h = _as_float(_profile_get(profile, "height_cm"))
    w = _as_float(_profile_get(profile, "weight_kg"))
    if h is None or w is None:
        missing_text = "Height and weight are required for classic height-to-weight cap checks."
        return {
            "eligible": True,
            "reasons": [missing_text],
            "needs_confirmation": True,
            "missing_required": True,
        }

    candidate_caps = []
    for cap in caps:
        if class_spec is not None and cap.class_id and cap.class_id != class_spec.class_id:
            continue
        candidate_caps.append(cap)

    if not candidate_caps:
        candidate_caps = list(caps)

    matching_cap = None
    for cap in candidate_caps:
        allowed = cap.max_allowed_weight_kg(h)
        if allowed is not None:
            matching_cap = cap
            break

    if matching_cap is None:
        return {
            "eligible": True,
            "reasons": ["Height-to-weight cap table exists but no matching row was found for current height."],
            "needs_confirmation": True,
            "missing_required": False,
        }

    max_allowed = matching_cap.max_allowed_weight_kg(h)
    if max_allowed is None:
        return {
            "eligible": True,
            "reasons": ["Could not compute maximum allowed weight from the configured cap table."],
            "needs_confirmation": True,
            "missing_required": True,
        }

    if w > max_allowed:
        if not strict_weight_checks:
            return {
                "eligible": True,
                "reasons": [
                    f"Max allowed weight is {max_allowed:.1f} kg for height {h:.1f} cm. "
                    f"Current weight is {w:.1f} kg (advisory until competition day)."
                ],
                "needs_confirmation": True,
                "missing_required": False,
            }
        return {
            "eligible": False,
            "reasons": [f"Max allowed weight is {max_allowed:.1f} kg for height {h:.1f} cm. Current weight is {w:.1f} kg."],
            "needs_confirmation": False,
            "missing_required": False,
        }

    return {
        "eligible": True,
        "reasons": [f"Max allowed weight is {max_allowed:.1f} kg for height {h:.1f} cm."],
        "needs_confirmation": False,
        "missing_required": False,
    }


def _weight_checks_are_strict(profile: Any) -> bool:
    competition_date = _competition_date_from_profile(profile)
    if competition_date is None:
        return False
    return date.today() >= competition_date


def _resolve_event_date(profile: Any) -> date:
    competition_date = _as_text(_profile_get(profile, "competition_date"))
    if not competition_date:
        plan = _profile_get(profile, "plan")
        if plan is not None:
            competition_date = _as_text(getattr(plan, "competition_date", None))
    parsed = _parse_iso_date(competition_date)
    return parsed or date.today()


def _competition_date_from_profile(profile: Any) -> Optional[date]:
    competition_date = _as_text(_profile_get(profile, "competition_date"))
    if not competition_date:
        plan = _profile_get(profile, "plan")
        if plan is not None:
            competition_date = _as_text(getattr(plan, "competition_date", None))
    return _parse_iso_date(competition_date)


def _competition_history(profile: Any) -> Optional[list[Any]]:
    for key in ("competition_history", "contest_history"):
        val = _profile_get(profile, key)
        if isinstance(val, list):
            return val
    return None


def _resolved_age(profile: Any, today: date) -> Optional[int]:
    resolver = getattr(profile, "resolved_age_years", None)
    if callable(resolver):
        try:
            resolved = resolver(today=today)
            if resolved is not None:
                return int(resolved)
        except Exception:
            pass
    return _resolved_age_from_dob(profile, today)


def _resolved_age_from_dob(profile: Any, today: date) -> Optional[int]:
    dob = _as_text(_profile_get(profile, "date_of_birth"))
    dob_date = _parse_iso_date(dob)
    if dob_date is None:
        return None
    years = today.year - dob_date.year
    if (today.month, today.day) < (dob_date.month, dob_date.day):
        years -= 1
    return max(0, years)


def _profile_get(profile: Any, key: str) -> Any:
    if profile is None:
        return None
    if isinstance(profile, dict):
        return profile.get(key)
    return getattr(profile, key, None)


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_optional_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "si", "sí"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    return None


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except Exception:
        return None
