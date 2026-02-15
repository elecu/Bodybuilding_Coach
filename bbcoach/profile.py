from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from .utils.cachepi import mirror_file
Federation = Literal["WNBF_UK", "PCA", "UKBFF", "BNBF", "NABBA"]
TTSBackend = Literal["piper_bin", "espeak", "auto"]


class ContestResult(BaseModel):
    date: str  # YYYY-MM-DD
    federation: Federation
    categories: list[str] = Field(default_factory=list)
    first_timers: bool = True
    placing: Optional[str] = None  # e.g. "1st", "Top 5", "DNF"
    notes: Optional[str] = None


class PrepPhase(BaseModel):
    mode: Literal["bulking", "cutting", "maintenance"] = "maintenance"
    start_date: Optional[str] = None
    start_weight_kg: Optional[float] = None


class CompetitionPlan(BaseModel):
    competition_date: Optional[str] = None  # YYYY-MM-DD
    target_categories: list[str] = Field(default_factory=lambda: ["Mens Physique"])
    first_timers: bool = True


class AudioConfig(BaseModel):
    mic_prefer: Optional[str] = None
    mic_fallback: Optional[str] = None


class VideoConfig(BaseModel):
    depth_min_m: Optional[float] = None
    depth_max_m: Optional[float] = None


class UserProfile(BaseModel):
    # Keep "schema" in the saved JSON for forward compatibility, but avoid
    # shadowing BaseModel.schema (Pydantic warning).
    model_config = ConfigDict(populate_by_name=True)
    schema_version: int = Field(default=1, alias="schema")
    name: str
    federation: Federation = "PCA"
    selected_categories: list[str] = Field(default_factory=lambda: ["Mens Physique"])
    first_timers: bool = True

    prep: PrepPhase = Field(default_factory=PrepPhase)
    plan: CompetitionPlan = Field(default_factory=CompetitionPlan)

    bodyweight_log: list[dict] = Field(default_factory=list)  # {date, weight_kg}
    contest_history: list[ContestResult] = Field(default_factory=list)

    # Templates store pose features for personal matching
    templates: dict[str, dict] = Field(default_factory=dict)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    coach_voice: bool = False
    tts_backend: TTSBackend = "piper_bin"


@dataclass
class ProfileStore:
    root: Path

    @staticmethod
    def default() -> "ProfileStore":
        root = Path(__file__).resolve().parent.parent / "config" / "profiles"
        root.mkdir(parents=True, exist_ok=True)
        return ProfileStore(root=root)

    def path_for(self, name: str) -> Path:
        safe = "".join(c for c in name.lower() if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
        return self.root / f"{safe}.json"

    def create(self, name: str) -> UserProfile:
        p = self.path_for(name)
        if p.exists():
            raise FileExistsError(f"Profile already exists: {p}")
        prof = UserProfile(name=name)
        self.save(prof)
        return prof

    def load(self, name: str) -> UserProfile:
        p = self.path_for(name)
        if not p.exists():
            # Create on first use
            return self.create(name)
        data = json.loads(p.read_text(encoding="utf-8"))
        return UserProfile.model_validate(data)

    def save(self, profile: UserProfile) -> None:
        p = self.path_for(profile.name)
        p.write_text(profile.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        mirror_file(p)

    def edit_interactive(self, name: str) -> None:
        prof = self.load(name)
        print("--- Edit profile (press Enter to keep current value) ---")

        fed = input(f"Federation [WNBF_UK/UKBFF/PCA/BNBF/NABBA] ({prof.federation}): ").strip()
        if fed in ("WNBF_UK", "UKBFF", "PCA", "BNBF", "NABBA"):
            prof.federation = fed  # type: ignore

        ft = input(f"First Timers? [y/n] ({'y' if prof.first_timers else 'n'}): ").strip().lower()
        if ft in ("y", "n"):
            prof.first_timers = (ft == "y")
            prof.plan.first_timers = prof.first_timers

        cats = input(f"Categories (comma) ({', '.join(prof.selected_categories)}): ").strip()
        if cats:
            prof.selected_categories = [c.strip() for c in cats.split(",") if c.strip()]
            prof.plan.target_categories = list(prof.selected_categories)

        comp = input(f"Competition date YYYY-MM-DD ({prof.plan.competition_date or ''}): ").strip()
        if comp:
            prof.plan.competition_date = comp

        mode = input(f"Prep mode [bulking/cutting/maintenance] ({prof.prep.mode}): ").strip().lower()
        if mode in ("bulking", "cutting", "maintenance"):
            prof.prep.mode = mode  # type: ignore

        sd = input(f"Prep start date YYYY-MM-DD ({prof.prep.start_date or ''}): ").strip()
        if sd:
            prof.prep.start_date = sd

        sw = input(f"Prep start weight kg ({prof.prep.start_weight_kg or ''}): ").strip()
        if sw:
            try:
                prof.prep.start_weight_kg = float(sw)
            except ValueError:
                pass

        self.save(prof)
        print(f"Saved: {self.path_for(name)}")
