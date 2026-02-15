from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from bbcoach.athlete_profiles import AthleteProfile
from bbcoach.app_state import AppStateStore
from bbcoach.federations.specs import SelectedDivisionRef


class AppStateStoreTests(unittest.TestCase):
    def test_selected_divisions_persist_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "app_state.json"
            state = AppStateStore(persist_path=persist_path)
            refs = [
                SelectedDivisionRef(federation_id="ukbff", division_id="mens_physique", class_id="up_to_171"),
                SelectedDivisionRef(federation_id="wnbf_uk", division_id="classic_physique"),
            ]
            state.set_selected_divisions(refs)

            reloaded = AppStateStore(persist_path=persist_path)
            keys = [ref.key() for ref in reloaded.state.selected_divisions]
            self.assertEqual(keys, [ref.key() for ref in refs])

    def test_invalid_entries_in_persisted_file_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "app_state.json"
            persist_path.write_text(
                '{"selected_divisions":[{"federation_id":"ukbff","division_id":"mens_bodybuilding"},null,"bad"]}',
                encoding="utf-8",
            )
            reloaded = AppStateStore(persist_path=persist_path)
            keys = [ref.key() for ref in reloaded.state.selected_divisions]
            self.assertEqual(keys, ["ukbff:mens_bodybuilding"])

    def test_selected_divisions_are_scoped_per_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "app_state.json"
            state = AppStateStore(persist_path=persist_path)

            profile_a = AthleteProfile(
                profile_id="profile_a",
                profile_name="Athlete A",
                sex_category="Male",
                date_of_birth="1990-01-01",
                height_cm=175.0,
                weight_kg=80.0,
            )
            profile_b = AthleteProfile(
                profile_id="profile_b",
                profile_name="Athlete B",
                sex_category="Male",
                date_of_birth="1992-01-01",
                height_cm=178.0,
                weight_kg=82.0,
            )

            refs_a = [SelectedDivisionRef(federation_id="ukbff", division_id="mens_physique")]
            refs_b = [SelectedDivisionRef(federation_id="wnbf_uk", division_id="classic_physique")]

            state.set_active_profile(profile_a)
            state.set_selected_divisions(refs_a)
            self.assertEqual([ref.key() for ref in state.state.selected_divisions], [ref.key() for ref in refs_a])

            state.set_active_profile(profile_b)
            self.assertEqual([ref.key() for ref in state.state.selected_divisions], [])
            state.set_selected_divisions(refs_b)
            self.assertEqual([ref.key() for ref in state.state.selected_divisions], [ref.key() for ref in refs_b])

            state.set_active_profile(profile_a)
            self.assertEqual([ref.key() for ref in state.state.selected_divisions], [ref.key() for ref in refs_a])

            reloaded = AppStateStore(persist_path=persist_path)
            reloaded.set_active_profile(profile_a)
            self.assertEqual([ref.key() for ref in reloaded.state.selected_divisions], [ref.key() for ref in refs_a])
            reloaded.set_active_profile(profile_b)
            self.assertEqual([ref.key() for ref in reloaded.state.selected_divisions], [ref.key() for ref in refs_b])


if __name__ == "__main__":
    unittest.main()
