from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import unittest

from bbcoach.athlete_profiles import (
    AthleteProfileStore,
    build_age_eligibility_snapshot,
    calculate_age_years,
)


class AthleteProfileStoreTests(unittest.TestCase):
    def test_profile_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AthleteProfileStore(root=Path(tmpdir) / "profiles")
            created = store.create(
                profile_name="Alice",
                sex_category="Female",
                date_of_birth="2000-02-20",
                height_cm=165.5,
                weight_kg=58.2,
                bodyfat_percent=15.0,
                has_disability=True,
                drug_tested_preference=True,
                competition_date="2026-07-12",
            )

            profile_path = store.path_for(created.profile_id)
            self.assertTrue(profile_path.exists())

            loaded = store.load(created.profile_id)
            self.assertEqual(loaded.profile_name, "Alice")
            self.assertEqual(loaded.sex_category, "Female")
            self.assertEqual(loaded.date_of_birth, "2000-02-20")
            self.assertEqual(loaded.height_cm, 165.5)
            self.assertEqual(loaded.weight_kg, 58.2)
            self.assertEqual(loaded.bodyfat_percent, 15.0)
            self.assertTrue(loaded.has_disability)
            self.assertTrue(loaded.drug_tested_preference)
            self.assertEqual(loaded.competition_date, "2026-07-12")
            self.assertEqual(loaded.age_years, calculate_age_years("2000-02-20"))

    def test_delete_active_profile_switches_to_available_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AthleteProfileStore(root=Path(tmpdir) / "profiles")
            first = store.create("First Athlete", date_of_birth="1994-01-10")
            second = store.create("Second Athlete", date_of_birth="1998-03-22")
            store.set_active_profile_id(second.profile_id)

            store.delete(second.profile_id)

            active = store.load_active_profile()
            self.assertIsNotNone(active)
            assert active is not None
            self.assertEqual(active.profile_id, first.profile_id)


class AgeCalculationTests(unittest.TestCase):
    def test_age_calculation_uses_birthday_boundary(self) -> None:
        today = date(2026, 2, 13)
        self.assertEqual(calculate_age_years("2000-02-13", today=today), 26)
        self.assertEqual(calculate_age_years("2000-02-14", today=today), 25)

    def test_age_calculation_rejects_future_or_invalid_dates(self) -> None:
        today = date(2026, 2, 13)
        self.assertIsNone(calculate_age_years("not-a-date", today=today))
        self.assertIsNone(calculate_age_years("2035-01-01", today=today))

    def test_age_eligibility_snapshot_uses_competition_date_when_present(self) -> None:
        today = date(2026, 2, 13)
        snapshot = build_age_eligibility_snapshot(
            date_of_birth="2000-03-20",
            competition_date="2026-07-10",
            today=today,
        )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertTrue(snapshot.uses_competition_date)
        self.assertEqual(snapshot.reference_date.isoformat(), "2026-07-10")
        self.assertEqual(snapshot.age_today, 25)
        self.assertEqual(snapshot.age_on_competition_date, 26)
        self.assertEqual(snapshot.competition_year_turns[23], 2023)
        self.assertEqual(snapshot.competition_year_turns[60], 2060)

    def test_age_eligibility_snapshot_defaults_to_today_without_competition_date(self) -> None:
        today = date(2026, 2, 13)
        snapshot = build_age_eligibility_snapshot(
            date_of_birth="2000-01-15",
            competition_date=None,
            today=today,
        )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertFalse(snapshot.uses_competition_date)
        self.assertEqual(snapshot.reference_date, today)
        self.assertEqual(snapshot.age_today, 26)
        self.assertEqual(snapshot.age_on_competition_date, 26)


if __name__ == "__main__":
    unittest.main()
