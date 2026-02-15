from __future__ import annotations

from datetime import date, timedelta
import unittest

from bbcoach.athlete_profiles import AthleteProfile
from bbcoach.federations.eligibility import evaluate_division_eligibility, matches_age_window
from bbcoach.federations.specs import (
    DivisionClassSpec,
    DivisionEligibilityRules,
    DivisionSpec,
    SelectedDivisionRef,
    load_federation_specs,
)


class EligibilityEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lib = load_federation_specs()

    def _division(self, federation_id: str, division_id: str):
        ref = SelectedDivisionRef(federation_id=federation_id, division_id=division_id)
        div = self.lib.get_division(ref)
        self.assertIsNotNone(div, f"Division not found: {federation_id}/{division_id}")
        return div

    def _class(self, federation_id: str, division_id: str, class_id: str):
        ref = SelectedDivisionRef(federation_id=federation_id, division_id=division_id, class_id=class_id)
        cls = self.lib.get_division_class(ref)
        self.assertIsNotNone(cls, f"Class not found: {federation_id}/{division_id}/{class_id}")
        return cls

    def test_year_end_junior_rule(self) -> None:
        division = self._division("ukbff", "junior_general_16_23")

        eligible_profile = AthleteProfile(
            profile_id="p1",
            profile_name="Eligible Junior",
            sex_category="Male",
            date_of_birth="2002-05-01",
            height_cm=175.0,
            weight_kg=80.0,
            competition_date="2025-08-10",
        )
        result_ok = evaluate_division_eligibility(eligible_profile, division)
        self.assertTrue(result_ok.eligible)

        ineligible_profile = AthleteProfile(
            profile_id="p2",
            profile_name="Too Old Junior",
            sex_category="Male",
            date_of_birth="2001-01-01",
            height_cm=175.0,
            weight_kg=80.0,
            competition_date="2025-08-10",
        )
        result_bad = evaluate_division_eligibility(ineligible_profile, division)
        self.assertFalse(result_bad.eligible)

    def test_year_turn_boundaries_use_competition_date(self) -> None:
        division = self._division("ukbff", "junior_general_16_23")
        # year_turns_23 = 2002 + 23 = 2025
        on_last_eligible_day = AthleteProfile(
            profile_id="p_boundary_ok",
            profile_name="Boundary OK",
            sex_category="Male",
            date_of_birth="2002-06-15",
            height_cm=175.0,
            weight_kg=80.0,
            competition_date="2025-12-31",
        )
        after_last_eligible_day = AthleteProfile(
            profile_id="p_boundary_bad",
            profile_name="Boundary BAD",
            sex_category="Male",
            date_of_birth="2002-06-15",
            height_cm=175.0,
            weight_kg=80.0,
            competition_date="2026-01-01",
        )
        result_ok = evaluate_division_eligibility(on_last_eligible_day, division)
        result_bad = evaluate_division_eligibility(after_last_eligible_day, division)
        self.assertTrue(result_ok.eligible)
        self.assertFalse(result_bad.eligible)

    def test_height_weight_cap_reports_max_allowed_as_advisory_before_competition(self) -> None:
        division = self._division("ukbff", "mens_classic_bodybuilding")
        class_spec = self._class("ukbff", "mens_classic_bodybuilding", "cb_u180")
        future_competition_date = (date.today() + timedelta(days=90)).isoformat()

        profile = AthleteProfile(
            profile_id="p3",
            profile_name="Classic Candidate",
            sex_category="Male",
            date_of_birth="1995-01-01",
            height_cm=180.0,
            weight_kg=88.0,
            competition_date=future_competition_date,
        )
        result = evaluate_division_eligibility(profile, division, class_spec=class_spec)
        self.assertTrue(result.eligible)
        self.assertTrue(result.needs_user_confirmation)
        joined = " ".join(result.reasons)
        self.assertIn("Max allowed weight is", joined)
        self.assertIn("87.0", joined)
        self.assertIn("advisory until competition day", joined)

    def test_height_weight_cap_blocks_on_competition_day(self) -> None:
        division = self._division("ukbff", "mens_classic_bodybuilding")
        class_spec = self._class("ukbff", "mens_classic_bodybuilding", "cb_u180")
        profile = AthleteProfile(
            profile_id="p3b",
            profile_name="Classic Candidate Strict Day",
            sex_category="Male",
            date_of_birth="1995-01-01",
            height_cm=180.0,
            weight_kg=88.0,
            competition_date=date.today().isoformat(),
        )
        result = evaluate_division_eligibility(profile, division, class_spec=class_spec)
        self.assertFalse(result.eligible)
        self.assertFalse(result.needs_user_confirmation)
        joined = " ".join(result.reasons)
        self.assertIn("Max allowed weight is", joined)

    def test_novice_restriction_requires_confirmation_without_history(self) -> None:
        division = self._division("nabba", "mens_bodybuilding_classes")
        profile = AthleteProfile(
            profile_id="p4",
            profile_name="No History",
            sex_category="Male",
            date_of_birth="1990-01-01",
            height_cm=178.0,
            weight_kg=85.0,
            competition_date="2026-07-01",
        )
        result = evaluate_division_eligibility(profile, division)
        self.assertTrue(result.eligible)
        self.assertTrue(result.needs_user_confirmation)

    def test_unclear_age_rule_semantics_do_not_auto_block(self) -> None:
        # Division has age-rule text but no machine turn-age value.
        division = self._division("ukbff", "masters_general")
        profile = AthleteProfile(
            profile_id="p5",
            profile_name="Masters Candidate",
            sex_category="Male",
            date_of_birth="1990-01-01",
            height_cm=178.0,
            weight_kg=85.0,
            competition_date="2026-07-01",
        )
        result = evaluate_division_eligibility(profile, division)
        self.assertTrue(result.eligible)
        self.assertTrue(result.needs_user_confirmation)

    def test_teens_class_does_not_accept_adult_age(self) -> None:
        division = self._division("nabba", "mens_bodybuilding_classes")
        teen_class = self._class("nabba", "mens_bodybuilding_classes", "teens")
        profile = AthleteProfile(
            profile_id="p6",
            profile_name="Adult Athlete",
            sex_category="Male",
            date_of_birth="1989-03-10",
            height_cm=178.0,
            weight_kg=88.0,
            competition_date=(date.today() + timedelta(days=30)).isoformat(),
        )
        result = evaluate_division_eligibility(profile, division, class_spec=teen_class)
        self.assertFalse(result.eligible)
        joined = " ".join(result.reasons).lower()
        self.assertIn("teen", joined)

    def test_age_window_excludes_teens_for_adult(self) -> None:
        division = self._division("nabba", "mens_bodybuilding_classes")
        teen_class = self._class("nabba", "mens_bodybuilding_classes", "teens")
        profile = AthleteProfile(
            profile_id="p7",
            profile_name="Adult For Window",
            sex_category="Male",
            date_of_birth="1989-03-10",
            height_cm=178.0,
            weight_kg=88.0,
            competition_date=(date.today() + timedelta(days=30)).isoformat(),
        )
        self.assertFalse(matches_age_window(profile, division, class_spec=teen_class, years_ahead=5))

    def test_over_45s_class_rejects_younger_athlete(self) -> None:
        division = self._division("nabba", "mens_physique_open_over45")
        over_45 = self._class("nabba", "mens_physique_open_over45", "over_45s")
        competition_date = date.today() + timedelta(days=30)
        dob_44 = date(competition_date.year - 44, competition_date.month, min(competition_date.day, 28)).isoformat()
        profile = AthleteProfile(
            profile_id="p_over45_young",
            profile_name="Too Young Over45",
            sex_category="Male",
            date_of_birth=dob_44,
            height_cm=178.0,
            weight_kg=84.0,
            competition_date=competition_date.isoformat(),
        )
        result = evaluate_division_eligibility(profile, division, class_spec=over_45)
        self.assertFalse(result.eligible)
        self.assertIn("minimum age", " ".join(result.reasons).lower())

    def test_over_45s_class_accepts_age_45(self) -> None:
        division = self._division("nabba", "mens_physique_open_over45")
        over_45 = self._class("nabba", "mens_physique_open_over45", "over_45s")
        competition_date = date.today() + timedelta(days=30)
        dob_45 = date(competition_date.year - 45, competition_date.month, min(competition_date.day, 28)).isoformat()
        profile = AthleteProfile(
            profile_id="p_over45_ok",
            profile_name="Age 45 Over45",
            sex_category="Male",
            date_of_birth=dob_45,
            height_cm=178.0,
            weight_kg=84.0,
            competition_date=competition_date.isoformat(),
        )
        result = evaluate_division_eligibility(profile, division, class_spec=over_45)
        self.assertTrue(result.eligible)

    def test_over_90kg_weight_class_not_treated_as_age_gate(self) -> None:
        division = self._division("ukbff", "mens_bodybuilding")
        over_90 = self._class("ukbff", "mens_bodybuilding", "mens_bb_o90")
        profile = AthleteProfile(
            profile_id="p_over90_weight",
            profile_name="Over90 Weight Class",
            sex_category="Male",
            date_of_birth="1989-03-10",
            height_cm=178.0,
            weight_kg=95.0,
            competition_date=(date.today() + timedelta(days=60)).isoformat(),
        )
        result = evaluate_division_eligibility(profile, division, class_spec=over_90)
        self.assertTrue(result.eligible)
        self.assertNotIn("minimum age", " ".join(result.reasons).lower())

    def test_women_master_class_rejects_male_profile(self) -> None:
        division = self._division("ukbff", "masters_general")
        women_over_35 = self._class("ukbff", "masters_general", "masters_women_over35")
        profile = AthleteProfile(
            profile_id="p_male_womens_class",
            profile_name="Male In Womens Class",
            sex_category="Male",
            date_of_birth="1989-06-19",
            height_cm=169.0,
            weight_kg=63.5,
            competition_date="2026-11-01",
        )
        result = evaluate_division_eligibility(profile, division, class_spec=women_over_35)
        self.assertFalse(result.eligible)
        self.assertIn("women-specific", " ".join(result.reasons).lower())

    def test_text_only_height_rule_blocks_if_under_minimum(self) -> None:
        division = DivisionSpec(
            division_id="height_text_division",
            division_name="Height Text Division",
            classes=[
                DivisionClassSpec(
                    class_id="over_170_cm",
                    class_name="Over 170 cm",
                    notes="Text-only class threshold.",
                )
            ],
            eligibility=DivisionEligibilityRules(),
        )
        profile = AthleteProfile(
            profile_id="p_height_text_block",
            profile_name="Height Text Block",
            sex_category="Male",
            date_of_birth="1989-06-19",
            height_cm=169.0,
            weight_kg=70.0,
            competition_date="2026-11-01",
        )
        result = evaluate_division_eligibility(profile, division, class_spec=division.classes[0])
        self.assertFalse(result.eligible)
        self.assertIn("requires height over 170.0 cm", " ".join(result.reasons).lower())

    def test_mixed_disability_requires_profile_flag(self) -> None:
        division = self._division("pca", "mixed_disability")
        profile = AthleteProfile(
            profile_id="p_disability_missing",
            profile_name="Missing Disability Field",
            sex_category="Male",
            date_of_birth="1989-03-10",
            height_cm=178.0,
            weight_kg=88.0,
            competition_date="2026-11-01",
            has_disability=None,
        )
        result = evaluate_division_eligibility(profile, division)
        self.assertTrue(result.eligible)
        self.assertTrue(result.needs_user_confirmation)
        self.assertIn("disability status is required", " ".join(result.reasons).lower())

    def test_mixed_disability_blocks_when_profile_says_no(self) -> None:
        division = self._division("pca", "mixed_disability")
        profile = AthleteProfile(
            profile_id="p_disability_no",
            profile_name="No Disability",
            sex_category="Female",
            date_of_birth="1992-04-11",
            height_cm=165.0,
            weight_kg=60.0,
            competition_date="2026-11-01",
            has_disability=False,
        )
        result = evaluate_division_eligibility(profile, division)
        self.assertFalse(result.eligible)
        self.assertIn("disabled athletes only", " ".join(result.reasons).lower())

    def test_mixed_disability_allows_when_profile_says_yes(self) -> None:
        division = self._division("pca", "mixed_disability")
        profile = AthleteProfile(
            profile_id="p_disability_yes",
            profile_name="Has Disability",
            sex_category="Female",
            date_of_birth="1992-04-11",
            height_cm=165.0,
            weight_kg=60.0,
            competition_date="2026-11-01",
            has_disability=True,
        )
        result = evaluate_division_eligibility(profile, division)
        self.assertTrue(result.eligible)

    def test_text_only_height_rule_allows_when_within_limit(self) -> None:
        division = DivisionSpec(
            division_id="height_text_division_2",
            division_name="Height Text Division 2",
            classes=[
                DivisionClassSpec(
                    class_id="upto_170_cm",
                    class_name="Up to and including 170 cm",
                    notes="Text-only class threshold.",
                )
            ],
            eligibility=DivisionEligibilityRules(),
        )
        profile = AthleteProfile(
            profile_id="p_height_text_ok",
            profile_name="Height Text OK",
            sex_category="Male",
            date_of_birth="1989-06-19",
            height_cm=169.0,
            weight_kg=70.0,
            competition_date="2026-11-01",
        )
        result = evaluate_division_eligibility(profile, division, class_spec=division.classes[0])
        self.assertTrue(result.eligible)

    def test_text_only_over_up_to_height_rule_blocks_below_lower_bound(self) -> None:
        division = DivisionSpec(
            division_id="height_text_division_3",
            division_name="Height Text Division 3",
            classes=[
                DivisionClassSpec(
                    class_id="over_168_upto_171",
                    class_name="Over 168 cm up to and including 171 cm",
                    notes="Text-only range threshold.",
                )
            ],
            eligibility=DivisionEligibilityRules(),
        )
        profile = AthleteProfile(
            profile_id="p_height_text_over_upto_block",
            profile_name="Height Text Over/Upto Block",
            sex_category="Male",
            date_of_birth="1989-06-19",
            height_cm=167.0,
            weight_kg=70.0,
            competition_date="2026-11-01",
        )
        result = evaluate_division_eligibility(profile, division, class_spec=division.classes[0])
        self.assertFalse(result.eligible)
        self.assertIn("requires height over 168.0 cm and up to 171.0 cm", " ".join(result.reasons).lower())

    def test_text_only_over_up_to_height_rule_allows_inside_range(self) -> None:
        division = DivisionSpec(
            division_id="height_text_division_4",
            division_name="Height Text Division 4",
            classes=[
                DivisionClassSpec(
                    class_id="over_168_upto_171_ok",
                    class_name="Over 168 cm up to and including 171 cm",
                    notes="Text-only range threshold.",
                )
            ],
            eligibility=DivisionEligibilityRules(),
        )
        profile = AthleteProfile(
            profile_id="p_height_text_over_upto_ok",
            profile_name="Height Text Over/Upto OK",
            sex_category="Male",
            date_of_birth="1989-06-19",
            height_cm=169.0,
            weight_kg=70.0,
            competition_date="2026-11-01",
        )
        result = evaluate_division_eligibility(profile, division, class_spec=division.classes[0])
        self.assertTrue(result.eligible)

    def test_ukbff_classic_u171_rejects_under_168(self) -> None:
        division = self._division("ukbff", "mens_classic_physique")
        class_spec = self._class("ukbff", "mens_classic_physique", "cp_u171")
        profile = AthleteProfile(
            profile_id="p_ukbff_u171_block",
            profile_name="UKBFF U171 Block",
            sex_category="Male",
            date_of_birth="1989-06-19",
            height_cm=167.0,
            weight_kg=70.0,
            competition_date="2026-11-01",
        )
        result = evaluate_division_eligibility(profile, division, class_spec=class_spec)
        self.assertFalse(result.eligible)
        self.assertIn("below class minimum", " ".join(result.reasons).lower())

    def test_age_window_includes_over40_when_reachable_in_five_years(self) -> None:
        division = self._division("ukbff", "masters_general")
        over_40 = self._class("ukbff", "masters_general", "masters_men_over40")
        base = date.today()
        dob = date(base.year - 39, base.month, max(1, min(base.day, 28))).isoformat()
        profile = AthleteProfile(
            profile_id="p8",
            profile_name="Near Over40",
            sex_category="Male",
            date_of_birth=dob,
            height_cm=178.0,
            weight_kg=88.0,
            competition_date=base.isoformat(),
        )
        self.assertTrue(matches_age_window(profile, division, class_spec=over_40, years_ahead=5))


if __name__ == "__main__":
    unittest.main()
