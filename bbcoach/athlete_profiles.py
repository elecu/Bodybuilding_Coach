from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Optional
import uuid


SEX_CATEGORIES = ("Male", "Female", "Other")
TURN_AGES = (23, 35, 40, 45, 50, 60)


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _parse_optional_bool(value: object) -> Optional[bool]:
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
    raise ValueError("has_disability must be true/false or yes/no.")


def calculate_age_years(date_of_birth: Optional[str], today: Optional[date] = None) -> Optional[int]:
    dob = _parse_iso_date(date_of_birth)
    if dob is None:
        return None
    today = today or date.today()
    if dob > today:
        return None
    years = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        years -= 1
    return max(0, years)


@dataclass(frozen=True)
class CompetitionCountdown:
    days_remaining: int
    weeks_remaining: float
    months_remaining: float


def calculate_competition_countdown(
    competition_date: Optional[str], today: Optional[date] = None
) -> Optional[CompetitionCountdown]:
    comp_date = _parse_iso_date(competition_date)
    if comp_date is None:
        return None
    today = today or date.today()
    days = (comp_date - today).days
    return CompetitionCountdown(
        days_remaining=days,
        weeks_remaining=days / 7.0,
        months_remaining=days / 30.44,
    )


@dataclass(frozen=True)
class AgeEligibilitySnapshot:
    age_today: int
    age_on_competition_date: int
    reference_date: date
    uses_competition_date: bool
    competition_year_turns: dict[int, int]


def build_age_eligibility_snapshot(
    date_of_birth: Optional[str],
    competition_date: Optional[str],
    today: Optional[date] = None,
) -> Optional[AgeEligibilitySnapshot]:
    dob = _parse_iso_date(date_of_birth)
    if dob is None:
        return None
    today = today or date.today()
    comp = _parse_iso_date(competition_date)
    reference = comp or today
    age_today = calculate_age_years(dob.isoformat(), today=today)
    age_reference = calculate_age_years(dob.isoformat(), today=reference)
    if age_today is None or age_reference is None:
        return None
    turns = {age: (dob.year + age) for age in TURN_AGES}
    return AgeEligibilitySnapshot(
        age_today=age_today,
        age_on_competition_date=age_reference,
        reference_date=reference,
        uses_competition_date=(comp is not None),
        competition_year_turns=turns,
    )


@dataclass
class AthleteProfile:
    profile_id: str
    profile_name: str
    sex_category: str = "Male"
    date_of_birth: str = ""
    age_years: Optional[int] = None
    height_cm: float = 175.0
    weight_kg: float = 80.0
    bodyfat_percent: Optional[float] = None
    has_disability: Optional[bool] = None
    drug_tested_preference: bool = True
    competition_date: Optional[str] = None

    def normalize(self) -> None:
        self.profile_name = (self.profile_name or "").strip()
        self.sex_category = self.sex_category if self.sex_category in SEX_CATEGORIES else "Other"
        self.date_of_birth = (self.date_of_birth or "").strip()
        if not self.date_of_birth:
            raise ValueError("date_of_birth is required.")
        dob = _parse_iso_date(self.date_of_birth)
        if dob is None:
            raise ValueError("date_of_birth must use YYYY-MM-DD.")
        if dob > date.today():
            raise ValueError("date_of_birth cannot be in the future.")
        self.date_of_birth = dob.isoformat()
        self.competition_date = (self.competition_date or "").strip() or None
        if self.competition_date:
            parsed_comp = _parse_iso_date(self.competition_date)
            if parsed_comp is None:
                raise ValueError("competition_date must use YYYY-MM-DD.")
            self.competition_date = parsed_comp.isoformat()
        self.height_cm = float(self.height_cm)
        self.weight_kg = float(self.weight_kg)
        self.bodyfat_percent = None if self.bodyfat_percent in ("", None) else float(self.bodyfat_percent)
        self.has_disability = _parse_optional_bool(self.has_disability)
        self.age_years = calculate_age_years(self.date_of_birth)

    def resolved_age_years(self, today: Optional[date] = None) -> Optional[int]:
        return calculate_age_years(self.date_of_birth, today=today)

    def to_dict(self) -> dict:
        self.normalize()
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "AthleteProfile":
        profile = AthleteProfile(
            profile_id=str(data.get("profile_id", "")),
            profile_name=str(data.get("profile_name", "")),
            sex_category=str(data.get("sex_category", "Other")),
            date_of_birth=data.get("date_of_birth"),
            age_years=data.get("age_years"),
            height_cm=float(data.get("height_cm", 0.0)),
            weight_kg=float(data.get("weight_kg", 0.0)),
            bodyfat_percent=data.get("bodyfat_percent"),
            has_disability=data.get("has_disability"),
            drug_tested_preference=bool(data.get("drug_tested_preference", False)),
            competition_date=data.get("competition_date"),
        )
        profile.normalize()
        return profile


@dataclass
class AthleteProfileStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default() -> "AthleteProfileStore":
        root = Path(__file__).resolve().parent.parent / "profiles"
        return AthleteProfileStore(root=root)

    @property
    def _active_profile_file(self) -> Path:
        return self.root / "active_profile_id.txt"

    def _sanitize_profile_id(self, profile_id: str) -> str:
        safe = "".join(ch for ch in str(profile_id) if ch.isalnum() or ch in ("-", "_"))
        if not safe:
            raise ValueError("Invalid profile_id")
        return safe

    def path_for(self, profile_id: str) -> Path:
        safe = self._sanitize_profile_id(profile_id)
        return self.root / f"{safe}.json"

    def list_profiles(self) -> list[AthleteProfile]:
        profiles: list[AthleteProfile] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                profile = AthleteProfile.from_dict(data)
                if profile.profile_id:
                    profiles.append(profile)
            except Exception:
                continue
        profiles.sort(key=lambda p: p.profile_name.lower())
        return profiles

    def create(
        self,
        profile_name: str,
        date_of_birth: str,
        sex_category: str = "Male",
        age_years: Optional[int] = None,
        height_cm: float = 175.0,
        weight_kg: float = 80.0,
        bodyfat_percent: Optional[float] = None,
        has_disability: Optional[bool] = None,
        drug_tested_preference: bool = True,
        competition_date: Optional[str] = None,
    ) -> AthleteProfile:
        profile = AthleteProfile(
            profile_id=uuid.uuid4().hex,
            profile_name=profile_name,
            sex_category=sex_category,
            date_of_birth=date_of_birth,
            age_years=age_years,
            height_cm=height_cm,
            weight_kg=weight_kg,
            bodyfat_percent=bodyfat_percent,
            has_disability=has_disability,
            drug_tested_preference=drug_tested_preference,
            competition_date=competition_date,
        )
        profile.normalize()
        self.save(profile)
        if not self.get_active_profile_id():
            self.set_active_profile_id(profile.profile_id)
        return profile

    def load(self, profile_id: str) -> AthleteProfile:
        path = self.path_for(profile_id)
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return AthleteProfile.from_dict(data)

    def save(self, profile: AthleteProfile) -> None:
        profile.normalize()
        path = self.path_for(profile.profile_id)
        path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    def delete(self, profile_id: str) -> None:
        path = self.path_for(profile_id)
        if path.exists():
            path.unlink()
        if self.get_active_profile_id() == profile_id:
            profiles = self.list_profiles()
            if profiles:
                self.set_active_profile_id(profiles[0].profile_id)
            else:
                self.clear_active_profile_id()

    def set_active_profile_id(self, profile_id: str) -> None:
        path = self.path_for(profile_id)
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_id}")
        self._active_profile_file.write_text(self._sanitize_profile_id(profile_id), encoding="utf-8")

    def clear_active_profile_id(self) -> None:
        self._active_profile_file.unlink(missing_ok=True)

    def get_active_profile_id(self) -> Optional[str]:
        if not self._active_profile_file.exists():
            return None
        value = self._active_profile_file.read_text(encoding="utf-8").strip()
        if not value:
            return None
        return self._sanitize_profile_id(value)

    def load_active_profile(self) -> Optional[AthleteProfile]:
        active_id = self.get_active_profile_id()
        if active_id:
            try:
                return self.load(active_id)
            except (FileNotFoundError, ValueError):
                self.clear_active_profile_id()
        profiles = self.list_profiles()
        if not profiles:
            return None
        self.set_active_profile_id(profiles[0].profile_id)
        return profiles[0]

    def save_active_profile(self, profile: AthleteProfile) -> None:
        self.save(profile)
        self.set_active_profile_id(profile.profile_id)


def build_demo_profile_name(existing: list[AthleteProfile], base_name: str = "Demo Athlete") -> str:
    names = {p.profile_name.lower() for p in existing}
    candidate = base_name
    suffix = 2
    while candidate.lower() in names:
        candidate = f"{base_name} {suffix}"
        suffix += 1
    return candidate


def iso_today() -> str:
    return datetime.now().date().isoformat()
